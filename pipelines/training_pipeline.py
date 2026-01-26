import os
# Force disable high-speed flight client to prevent the crash in GitHub Actions
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

# --- CONFIG ---
KARACHI_LAT, KARACHI_LON = 24.8607, 67.0011
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

def get_forecast_features(trained_columns):
    """Fetch 72-hour forecast from Open-Meteo API"""
    print("🌐 Fetching 72-hour forecast...")
    params = {
        "latitude": KARACHI_LAT, 
        "longitude": KARACHI_LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,dew_point_2m,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "forecast_days": 3
    }
    res = requests.get(FORECAST_URL, params=params, timeout=30).json()
    df = pd.DataFrame(res["hourly"])
    df['time'] = pd.to_datetime(df['time'])
    
    prep = pd.DataFrame({
        'year': df['time'].dt.year.astype('int64'), 
        'month': df['time'].dt.month.astype('int64'),
        'day': df['time'].dt.day.astype('int64'), 
        'hour': df['time'].dt.hour.astype('int64')
    })
    
    name_map = {
        'pm2_5': 'pm25', 'pm10': 'pm10', 'carbon_monoxide': 'co',
        'nitrogen_dioxide': 'no2', 'sulphur_dioxide': 'so2', 'ozone': 'o3',
        'temperature_2m': 'temperature', 'relative_humidity_2m': 'humidity',
        'wind_speed_10m': 'wind_speed', 'dew_point_2m': 'dew_point'
    }
    
    for api_name, local_name in name_map.items():
        if api_name in df.columns:
            prep[local_name] = df[api_name].astype('float64')
    
    for col in trained_columns:
        if col not in prep.columns:
            prep[col] = 0.0
    
    return prep[trained_columns], df['time']

def run_pipeline():
    print("=" * 70)
    print("🚀 KARACHI AQI TRAINING PIPELINE (REST API VERSION)")
    print("=" * 70)
    
    # 1. Login to Hopsworks
    project = hopsworks.login(api_key_value=os.getenv('MY_HOPSWORK_KEY'))
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    print("✅ Connected to Hopsworks!")
    
    # 2. READ DATA USING THE REST API ENGINE (The Fix)
    print("📥 Downloading data via REST API (Port 443)...")
    fg = fs.get_feature_group(name="karachi_aqi", version=1)
    
    # We use .read(read_options={"use_api": True}) which is the official
    # way to bypass the Arrow Flight Query Service in restricted environments.
    full_df = fg.select_all().read(read_options={"use_api": True})
    
    # SAFETY SHIELD: Stop if data is empty to avoid the "n_samples=0" crash
    if full_df is None or full_df.empty:
        print("❌ CRITICAL ERROR: Feature Group returned 0 rows.")
        print("Ensure 'karachi_aqi' has data in the Hopsworks UI before running.")
        return

    print(f"✅ SUCCESS! Retrieved {len(full_df)} rows.")

    # 3. PREPARE TARGET
    # Look for pm25 or aqi as target
    target = 'aqi' if 'aqi' in full_df.columns else 'pm25'
    print(f"🎯 Target identified: {target}")

    X = full_df.drop(columns=[target])
    y = full_df[[target]]
    
    # 4. SPLIT AND SCALE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. TRAINING TOURNAMENT
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        "SVR": SVR(kernel='rbf', C=10)
    }
    
    best_model, best_rmse, best_name = None, float('inf'), ""
    
    for name, model in models.items():
        print(f"🔄 Training {name}...")
        model.fit(X_train_scaled, y_train.values.ravel())
        preds = model.predict(X_test_scaled)
        rmse = root_mean_squared_error(y_test, preds)
        print(f"   ✅ {name} RMSE: {rmse:.4f}")
        
        # Save locally
        model_dir = f"model_{name.lower()}"
        if os.path.exists(model_dir): shutil.rmtree(model_dir)
        os.makedirs(model_dir)
        joblib.dump(model, f"{model_dir}/karachi_aqi_model.pkl")
        joblib.dump(scaler, f"{model_dir}/scaler.pkl")
        
        # Register to Registry
        hops_model = mr.python.create_model(name="karachi_aqi_model", metrics={"rmse": rmse})
        hops_model.save(model_dir)

        if rmse < best_rmse:
            best_rmse, best_model, best_name = rmse, model, name

    print(f"🏆 WINNER: {best_name} (RMSE: {best_rmse:.4f})")

    # 6. FORECAST
    X_f, times = get_forecast_features(X_train.columns.tolist())
    predictions = best_model.predict(scaler.transform(X_f))
    
    forecast_df = pd.DataFrame({
        'year': X_f['year'].astype('int64'),
        'month': X_f['month'].astype('int64'),
        'day': X_f['day'].astype('int64'),
        'hour': X_f['hour'].astype('int64'),
        'predicted_aqi': predictions.round(2).astype('float64'),
        'prediction_timestamp': times.dt.strftime('%Y-%m-%d %H:%M:%S')
    })

    # 7. UPLOAD
    fg_forecast = fs.get_or_create_feature_group(
        name="karachi_aqi_forecast", version=1, 
        primary_key=['year', 'month', 'day', 'hour'], 
        online_enabled=True
    )
    fg_forecast.insert(forecast_df, write_options={"wait_for_job": False})
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_pipeline()
