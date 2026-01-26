import os
import requests
import pandas as pd
import hopsworks
import joblib
import shutil
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR 
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import root_mean_squared_error
from hsml.model_schema import ModelSchema

# --- CONFIG ---
KARACHI_LAT, KARACHI_LON = 24.8607, 67.0011
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

def get_forecast_features(trained_columns):
    print("🌐 Fetching 72-hour Forecast Data...")
    params = {
        "latitude": KARACHI_LAT, "longitude": KARACHI_LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,dew_point_2m,pm2_5,pm10,carbon_monoxide",
        "forecast_days": 3
    }
    res = requests.get(FORECAST_URL, params=params).json()
    df_forecast = pd.DataFrame(res["hourly"])
    df_forecast['time'] = pd.to_datetime(df_forecast['time'])
    
    prep = pd.DataFrame({
        'year': df_forecast['time'].dt.year.astype('int64'), 
        'month': df_forecast['time'].dt.month.astype('int64'),
        'day': df_forecast['time'].dt.day.astype('int64'), 
        'hour': df_forecast['time'].dt.hour.astype('int64'),
        'weekday': df_forecast['time'].dt.weekday.astype('float64')
    })

    name_map = {
        'pm2_5': 'pm25', 'pm10': 'pm10', 'carbon_monoxide': 'co', 
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
    # 1. LOGIN (Hardened for GitHub Actions)
    print("🔑 Logging into Hopsworks...")
    project = hopsworks.login(
        api_key_value=os.getenv('MY_HOPSWORK_KEY'),
        # Force disable Flight here to prevent connection drops
        project=os.getenv('HOPSWORKS_PROJECT_NAME') # Optional: specify project name if known
    )
    fs = project.get_feature_store()
    
    # 2. GET FEATURE VIEW
    feature_view = fs.get_feature_view(name="karachi_aqi_view", version=2)
    
    # 3. FETCH DATA (Restricted for Stability)
    print("📥 Pulling training data...")
    try:
        # We use train_test_split but we can also try to limit the data if it's too large
        X_train, X_test, y_train, y_test = feature_view.train_test_split(
            test_size=0.2,
            # If your data is huge, consider using a training dataset created in the UI:
            # training_dataset_version=1 
        )
    except Exception as e:
        print(f"⚠️ Initial fetch failed: {e}. Retrying with direct batch read...")
        # Fallback to standard read if split fails
        data_df = feature_view.get_batch_data()
        from sklearn.model_selection import train_test_split
        y = data_df[['aqi']]
        X = data_df.drop(columns=['aqi'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Clean data
    X_train = X_train.dropna()
    y_train = y_train.loc[X_train.index]
    X_test = X_test.dropna()
    y_test = y_test.loc[X_test.index]

    # Scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. TOURNAMENT
    print("\n🏆 STARTING TOURNAMENT...")
    models = {
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10)
    }

    best_model, best_rmse = None, float('inf')
    best_model_name = ""

    for name, model in models.items():
        model.fit(X_train_scaled, y_train.values.ravel())
        preds = model.predict(X_test_scaled)
        rmse = root_mean_squared_error(y_test, preds)
        print(f"📊 {name:12} -> TEST RMSE: {rmse:.4f}")
        
        if rmse < best_rmse:
            best_rmse, best_model, best_model_name = rmse, model, name

    # 5. UPLOAD MODEL
    model_dir = "aqi_model_dir"
    if os.path.exists(model_dir): shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    
    joblib.dump(best_model, f"{model_dir}/karachi_aqi_model.pkl")
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")

    mr = project.get_model_registry()
    karachi_model = mr.python.create_model(
        name="karachi_aqi_model", 
        metrics={"rmse": best_rmse}, 
        input_schema=ModelSchema(X_train),
        output_schema=ModelSchema(y_train),
        description=f"Winner: {best_model_name}"
    )
    karachi_model.save(model_dir)

    # 6. FORECAST & UPLOAD
    X_f, times = get_forecast_features(X_train.columns.tolist())
    X_f_scaled = scaler.transform(X_f)
    preds = best_model.predict(X_f_scaled)
    
    forecast_df = X_f[['year', 'month', 'day', 'hour']].copy()
    forecast_df['predicted_aqi'] = preds.round(2).astype('float64')
    forecast_df['prediction_timestamp'] = times.dt.strftime('%Y-%m-%d %H:%M:%S')

    print("🚀 Updating Hopsworks Feature Group...")
    forecast_fg = fs.get_feature_group(name="karachi_aqi_forecast", version=1)
    # Using 'wait_for_job=False' helps prevent timeout errors in GitHub Actions
    forecast_fg.insert(forecast_df, write_options={"wait_for_job": False})
    
    print(f"✅ SUCCESS! Best model ({best_model_name}) deployed.")

if __name__ == "__main__":
    run_pipeline()
