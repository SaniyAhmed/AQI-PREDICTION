import os
import requests
import pandas as pd
import hopsworks
import joblib
import shutil
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR 
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

# --- CONFIG ---
KARACHI_LAT, KARACHI_LON = 24.8607, 67.0011
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

def get_forecast_features(trained_columns):
    print("🌐 Fetching 72-hour Forecast Data...")
    params = {
        "latitude": KARACHI_LAT, "longitude": KARACHI_LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,dew_point_2m,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "forecast_days": 3
    }
    res = requests.get(FORECAST_URL, params=params).json()
    df_forecast = pd.DataFrame(res["hourly"])
    df_forecast['time'] = pd.to_datetime(df_forecast['time'])
    prep = pd.DataFrame({
        'year': df_forecast['time'].dt.year.astype('int64'), 
        'month': df_forecast['time'].dt.month.astype('int64'),
        'day': df_forecast['time'].dt.day.astype('int64'), 
        'hour': df_forecast['time'].dt.hour.astype('int64')
    })
    name_map = {
        'pm2_5': 'pm25', 'pm10': 'pm10', 'carbon_monoxide': 'co', 
        'nitrogen_dioxide': 'no2', 'sulphur_dioxide': 'so2', 'ozone': 'o3', 
        'temperature_2m': 'temperature', 'relative_humidity_2m': 'humidity', 
        'wind_speed_10m': 'wind_speed', 'dew_point_2m': 'dew_point'
    }
    for api, local in name_map.items():
        if api in df_forecast.columns: 
            prep[local] = df_forecast[api].astype('float64')
    
    for col in trained_columns:
        if col not in prep.columns: 
            prep[col] = 0.0
            
    return prep[trained_columns], df_forecast['time']

def run_pipeline():
    # --- LOGIN ---
    api_key = os.getenv('MY_HOPSWORK_KEY')
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    
    # --- 1. THE ULTIMATE BYPASS: DOWNLOAD RAW PARQUET ---
    print("📥 Bypassing ALL Feature Store Services... Downloading raw data files.")
    
    # Path logic: On Hopsworks, data is stored in:
    # /Projects/[ProjectName]/Resources/FeatureStore/[FSName]/[FGName]_[Version]/
    # We will grab the Offline Training Data directly.
    dataset_api = project.get_dataset_api()
    
    # This path is standard for Hopsworks Offline Storage
    remote_path = f"Resources/FeatureStore/{fs.name}/karachi_aqi_4"
    local_dir = "./raw_data"
    
    if os.path.exists(local_dir): shutil.rmtree(local_dir)
    os.makedirs(local_dir)

    print(f"📂 Searching for data in: {remote_path}")
    try:
        # Download the directory containing the parquet files
        dataset_api.download(remote_path, local_path=local_dir, overwrite=True)
        
        # Find the .parquet file (Hudi/Spark usually names it with a UUID)
        parquet_files = []
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                if file.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, file))
        
        if not parquet_files:
            raise Exception("No Parquet files found in the downloaded directory.")
            
        print(f"📄 Found {len(parquet_files)} data files. Loading into Pandas...")
        data_list = [pd.read_parquet(f) for f in parquet_files]
        data = pd.concat(data_list, ignore_index=True)
        
        # Clean up Hudi/System columns if they exist
        system_cols = ['_hoodie_commit_time', '_hoodie_commit_seqno', '_hoodie_record_key', '_hoodie_partition_path', '_hoodie_file_name']
        data = data.drop(columns=[c for c in system_cols if c in data.columns])

    except Exception as e:
        print(f"❌ Direct download failed: {e}")
        print("💡 Suggestion: Go to Hopsworks UI -> Data Sets -> Resources and verify 'karachi_aqi_4' exists.")
        return

    print(f"✅ Data loaded: {len(data)} rows.")

    # --- 2. PREP & TOURNAMENT ---
    y = data[['aqi']]
    X = data.drop(columns=['aqi'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base_models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "SVR": SVR(kernel='rbf', C=1.0)
    }

    print("\n🏆 STARTING TOURNAMENT...")
    best_model, best_rmse, best_model_name = None, float('inf'), ""

    for name, model in base_models.items():
        print(f"🔍 Training {name}...")
        model.fit(X_train_scaled, y_train.values.ravel())
        preds = model.predict(X_test_scaled)
        test_rmse = root_mean_squared_error(y_test, preds)
        print(f"    📊 {name:12} -> RMSE: {test_rmse:.4f}")

        model_dir = f"model_dir_{name.lower()}"
        if os.path.exists(model_dir): shutil.rmtree(model_dir)
        os.makedirs(model_dir)
        joblib.dump(model, f"{model_dir}/karachi_aqi_model.pkl")
        joblib.dump(scaler, f"{model_dir}/scaler.pkl")

        current_model = mr.python.create_model(
            name=f"karachi_aqi_{name.lower()}", 
            metrics={"rmse": float(test_rmse)}, 
            description=f"Direct Download Training"
        )
        current_model.save(model_dir)

        if test_rmse < best_rmse:
            best_rmse, best_model, best_model_name = test_rmse, model, name

    # --- 3. FORECAST & UPLOAD ---
    X_f, times = get_forecast_features(X_train.columns.tolist())
    preds = best_model.predict(scaler.transform(X_f))
    forecast_df = X_f[['year', 'month', 'day', 'hour']].copy()
    forecast_df['predicted_aqi'] = preds.round(2).astype('float64')
    forecast_df['prediction_timestamp'] = times.dt.strftime('%Y-%m-%d %H:%M:%S')

    fg_forecast = fs.get_or_create_feature_group(
        name="karachi_aqi_forecast", version=1, 
        primary_key=['year', 'month', 'day', 'hour'], online_enabled=True
    )
    fg_forecast.insert(forecast_df, write_options={"wait_for_job": False})
    print(f"✅ Pipeline Completed via Dataset API Bypass!")

if __name__ == "__main__":
    run_pipeline()
