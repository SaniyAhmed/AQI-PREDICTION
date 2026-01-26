import os
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
    # Login normally for Registry and Feature Group metadata
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    
    print("📥 DOWNLOADING DATA VIA DIRECT HTTPS BYPASS...")
    # Get metadata for IDs
    fg = fs.get_feature_group(name="karachi_aqi", version=1)
    
    # CONSTRUCT DIRECT REST API URL
    # This bypasses the Arrow Flight Client entirely.
    base_url = "https://c.app.hopsworks.ai/hopsworks-api/api"
    url = f"{base_url}/project/{project.id}/featurestores/{fs.id}/featuregroups/{fg.id}/data?isOnline=false&n=10000"
    headers = {"Authorization": f"ApiKey {api_key}"}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data via REST: {response.status_code} - {response.text}")
    
    # Load JSON response into DataFrame
    data_items = response.json().get("items", [])
    full_df = pd.DataFrame(data_items)
    
    if full_df.empty:
        print("❌ CRITICAL ERROR: No data found in Feature Group.")
        return

    print(f"✅ SUCCESS! Retrieved {len(full_df)} rows via REST API.")

    # --- ML LOGIC ---
    target = 'pm25' if 'pm25' in full_df.columns else full_df.columns[-1]
    X = full_df.drop(columns=[target])
    y = full_df[[target]]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Tournament (Simplified for 100% stability)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train.values.ravel())
    rmse = root_mean_squared_error(y_test, model.predict(X_test_scaled))
    print(f"📊 Model Trained. RMSE: {rmse:.4f}")

    # Save and Register
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

    # Upload via Feature Group Insert (Usually works as it uses simple POST)
    fg_forecast = fs.get_or_create_feature_group(
        name="karachi_aqi_forecast", version=1, 
        primary_key=['year', 'month', 'day', 'hour'], 
        online_enabled=True
    )
    fg_forecast.insert(forecast_df, write_options={"wait_for_job": False})
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_pipeline()
