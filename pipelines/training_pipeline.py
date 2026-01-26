import os
# Force global settings to disable the crash-prone client
os.environ["HSFS_DISABLE_FLIGHT_CLIENT"] = "True"

import requests
import pandas as pd
import hopsworks
import joblib
import shutil
import time
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR 
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

# --- THE HOPSWORKS INTERCEPTOR ---
# This prevents the SDK from even looking for the Arrow Flight service
import hsfs
from hsfs.core import arrow_flight_client
def mock_init(self, *args, **kwargs): return
arrow_flight_client.ArrowFlightClient.__init__ = mock_init

# --- CONFIG ---
KARACHI_LAT, KARACHI_LON = 24.8607, 67.0011
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

def get_forecast_features(trained_columns):
    print("🌐 Fetching 72-hour forecast...")
    params = {
        "latitude": KARACHI_LAT, "longitude": KARACHI_LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "forecast_days": 3
    }
    res = requests.get(FORECAST_URL, params=params).json()
    df = pd.DataFrame(res["hourly"])
    df['time'] = pd.to_datetime(df['time'])
    prep = pd.DataFrame({
        'year': df['time'].dt.year.astype('int64'), 
        'month': df['time'].dt.month.astype('int64'),
        'day': df['time'].dt.day.astype('int64'), 
        'hour': df['time'].dt.hour.astype('int64')
    })
    name_map = {'pm2_5': 'pm25', 'pm10': 'pm10', 'carbon_monoxide': 'co', 'nitrogen_dioxide': 'no2', 'sulphur_dioxide': 'so2', 'ozone': 'o3', 'temperature_2m': 'temperature', 'relative_humidity_2m': 'humidity', 'wind_speed_10m': 'wind_speed', 'dew_point_2m': 'dew_point'}
    for api_name, local_name in name_map.items():
        if api_name in df.columns: prep[local_name] = df[api_name].astype('float64')
    for col in trained_columns:
        if col not in prep.columns: prep[col] = 0.0
    return prep[trained_columns], df['time']

def run_pipeline():
    api_key = os.getenv('MY_HOPSWORK_KEY')
    # 1. Login
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    
    print("📥 RETRIEVING DATA VIA SDK (REST API MODE)...")
    # 2. Get the Feature Group
    fg = fs.get_feature_group(name="karachi_aqi", version=1)
    
    # THE CRITICAL STEP: use_api=True tells Hopsworks to use Port 443 (REST)
    # instead of Port 443/5005 (Arrow Flight). 
    # Because of our 'mock_init' above, it cannot crash.
    full_df = fg.select_all().read(read_options={"use_api": True})
    
    if full_df is None or full_df.empty:
        print("❌ CRITICAL ERROR: Feature Group is empty or could not be read.")
        return

    print(f"✅ SUCCESS! Retrieved {len(full_df)} rows.")

    # --- ML TOURNAMENT ---
    target = 'pm25' if 'pm25' in full_df.columns else full_df.columns[-1]
    X = full_df.drop(columns=[target])
    y = full_df[[target]]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fast tournament for GitHub Actions stability
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train.values.ravel())
    rmse = root_mean_squared_error(y_test, model.predict(X_test_scaled))
    print(f"📊 Model Trained. RMSE: {rmse:.4f}")

    # Save locally and Register
    model_dir = "model_files"
    if os.path.exists(model_dir): shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    joblib.dump(model, f"{model_dir}/karachi_aqi_model.pkl")
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")
    
    h_model = mr.python.create_model(name="karachi_aqi_model", metrics={"rmse": rmse})
    h_model.save(model_dir)

    # --- FORECAST & UPLOAD ---
    X_f, times = get_forecast_features(X_train.columns.tolist())
    predictions = model.predict(scaler.transform(X_f))
    
    forecast_df = pd.DataFrame({
        'year': X_f['year'].astype('int64'),
        'month': X_f['month'].astype('int64'),
        'day': X_f['day'].astype('int64'),
        'hour': X_f['hour'].astype('int64'),
        'predicted_aqi': predictions.round(2).astype('float64'),
        'prediction_timestamp': times.dt.strftime('%Y-%m-%d %H:%M:%S')
    })

    fg_forecast = fs.get_or_create_feature_group(
        name="karachi_aqi_forecast", version=1, 
        primary_key=['year', 'month', 'day', 'hour'], 
        online_enabled=True
    )
    # Insert works via standard HTTP POST
    fg_forecast.insert(forecast_df, write_options={"wait_for_job": False})
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_pipeline()
