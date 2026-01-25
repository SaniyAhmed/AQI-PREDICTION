import os
# --- THE GITHUB ACTIONS FIX ---
os.environ["HSFS_DISABLE_FLIGHT_CLIENT"] = "True"

import requests
import pandas as pd
import hopsworks
import joblib
import shutil
import time
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR 
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

# --- CONFIG ---
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011
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

    prep = pd.DataFrame()
    prep['year'] = df_forecast['time'].dt.year.astype('int64')
    prep['month'] = df_forecast['time'].dt.month.astype('int64')
    prep['day'] = df_forecast['time'].dt.day.astype('int64')
    prep['hour'] = df_forecast['time'].dt.hour.astype('int64')
    
    name_map = {
        'pm2_5': 'pm25', 'pm10': 'pm10', 'carbon_monoxide': 'co',
        'nitrogen_dioxide': 'no2', 'sulphur_dioxide': 'so2', 'ozone': 'o3',
        'temperature_2m': 'temperature', 'relative_humidity_2m': 'humidity',
        'wind_speed_10m': 'wind_speed', 'dew_point_2m': 'dew_point'
    }
    
    for api_name, local_name in name_map.items():
        if api_name in df_forecast.columns:
            prep[local_name] = df_forecast[api_name].astype('float64')

    for col in trained_columns:
        if col not in prep.columns:
            prep[col] = 0.0

    return prep[trained_columns], df_forecast['time']

def run_pipeline():
    # 1. LOGIN
    api_key = os.getenv('MY_HOPSWORK_KEY') 
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()

    # 2. FETCH DATA FROM VERSION 3 VIEW
    print("📥 Accessing Feature View V3...")
    feature_view = fs.get_feature_view(name="karachi_aqi_view", version=3)
    
    # Logic fix: Use get_batch_data() which works for Feature Views
    try:
        print("尝试进行数据读取 (Local Split Strategy)...")
        # read_options forces a direct download which is safer for GitHub
        full_df = feature_view.get_batch_data() 
        
        target = "aqi"
        y = full_df[[target]]
        X = full_df.drop(columns=[target])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"✅ Data loaded successfully. Rows: {len(full_df)}")
    except Exception as e:
        print(f"❌ Critical failure reading data: {e}")
        return

    # 3. CLEAN & SCALE
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. MODEL TOURNAMENT
    print("🏆 Training Model Tournament...")
    models = {
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6),
        "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=12),
        "SVR": SVR(kernel='rbf', C=15, epsilon=0.1) 
    }

    best_model, best_rmse, best_model_name = None, float('inf'), ""
    for name, model in models.items():
        model.fit(X_train_scaled, y_train.values.ravel())
        preds = model.predict(X_test_scaled)
        rmse = root_mean_squared_error(y_test, preds)
        print(f"   - {name} RMSE: {rmse:.4f}")
        if rmse < best_rmse:
            best_rmse, best_model, best_model_name = rmse, model, name

    # 5. SAVE WINNING MODEL
    print(f"📦 Saving {best_model_name} (RMSE: {best_rmse:.2f})")
    model_dir = "aqi_model_dir"
    if os.path.exists(model_dir): shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    
    joblib.dump(best_model, f"{model_dir}/karachi_aqi_model.pkl")
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")

    mr = project.get_model_registry()
    karachi_model = mr.python.create_model(
        name="karachi_aqi_model", 
        metrics={"rmse": best_rmse},
        description=f"Winner V3: {best_model_name}"
    )
    karachi_model.save(model_dir)

    # 6. FORECAST
    X_forecast, timestamps = get_forecast_features(X_train.columns.tolist())
    X_forecast_scaled = scaler.transform(X_forecast)
    future_preds = best_model.predict(X_forecast_scaled)

    # 7. PREPARE DATAFRAME
    forecast_df = X_forecast[['year', 'month', 'day', 'hour']].copy()
    forecast_df['predicted_aqi'] = future_preds.round(2).astype('float64')
    forecast_df['prediction_timestamp'] = timestamps.dt.strftime('%Y-%m-%d %H:%M:%S')

    # 8. INSERTION WITH RETRY
    max_retries = 3
    for attempt in range(max_retries):
        try:
            forecast_fg = fs.get_or_create_feature_group(
                name="karachi_aqi_forecast",
                version=1,
                primary_key=['year', 'month', 'day', 'hour'],
                description="3-Day Predicted AQI",
                online_enabled=True
            )
            forecast_fg.insert(forecast_df, write_options={"wait_for_job": False})
            print(f"✅ SUCCESS! Best model {best_model_name} synced.")
            break 
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e}")
            time.sleep(10)

if __name__ == "__main__":
    run_pipeline()
