import os
# Keeping this to ensure stability on the free tier
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
from sklearn.model_selection import train_test_split, RandomizedSearchCV

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
    
    # --- 1. GET DATA FROM VERSION 4 ---
    print("📥 Accessing Feature Group Version 4...")
    fg = fs.get_feature_group(name="karachi_aqi", version=4)
    
    # We use the feature view to create the training data
    # If the view doesn't exist for V4, we create it
    try:
        feature_view = fs.get_feature_view(name="karachi_aqi_view", version=4)
    except:
        print("🏗️ Creating Feature View Version 4...")
        feature_view = fs.create_feature_view(
            name="karachi_aqi_view",
            version=4,
            query=fg.select_all(),
            labels=["aqi"]
        )

    # Read data (bypassing the server-side split to avoid query service timeouts)
    data = feature_view.read()
    
    y = data[['aqi']]
    X = data.drop(columns=['aqi'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, y_train = X_train.dropna(), y_train.loc[X_train.dropna().index]
    X_test, y_test = X_test.dropna(), y_test.loc[X_test.dropna().index]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 2. TOURNAMENT LOGIC (RF, XGB, SVR) ---
    # Simplified grids for a stable first run; you can expand these later
    param_grids = {
        "RandomForest": {"n_estimators": [100], "max_depth": [10]},
        "XGBoost": {"n_estimators": [100], "learning_rate": [0.1]},
        "SVR": {"C": [1.0], "epsilon": [0.1]}
    }
    base_models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "SVR": SVR(kernel='rbf')
    }

    print("\n🏆 STARTING TOURNAMENT (All models will be saved to registry)...")
    print("-" * 50)
    best_model, best_rmse, best_model_name = None, float('inf'), ""

    for name, model in base_models.items():
        print(f"🔍 Training {name}...")
        # Fitting with simple search to ensure completion
        model.fit(X_train_scaled, y_train.values.ravel())
        
        preds = model.predict(X_test_scaled)
        test_rmse = root_mean_squared_error(y_test, preds)
        print(f"    📊 {name:12} -> RMSE: {test_rmse:.4f}")

        # --- REGISTRY LOGIC ---
        model_dir = f"model_dir_{name.lower()}"
        if os.path.exists(model_dir): shutil.rmtree(model_dir)
        os.makedirs(model_dir)
        
        joblib.dump(model, f"{model_dir}/karachi_aqi_model.pkl")
        joblib.dump(scaler, f"{model_dir}/scaler.pkl")

        # Log each model to the registry
        current_model = mr.python.create_model(
            name=f"karachi_aqi_{name.lower()}", 
            metrics={"rmse": float(test_rmse)}, 
            description=f"Tournament Participant: {name}"
        )
        current_model.save(model_dir)

        # Winner Selection
        if test_rmse < best_rmse:
            best_rmse, best_model, best_model_name = test_rmse, model, name

    print("-" * 50)
    print(f"⭐ TOURNAMENT WINNER: {best_model_name} (RMSE: {best_rmse:.4f})")

    # --- 3. FORECAST GENERATION (Using Winner) ---
    X_f, times = get_forecast_features(X_train.columns.tolist())
    preds = best_model.predict(scaler.transform(X_f))
    
    forecast_df = X_f[['year', 'month', 'day', 'hour']].copy()
    forecast_df['predicted_aqi'] = preds.round(2).astype('float64')
    forecast_df['prediction_timestamp'] = times.dt.strftime('%Y-%m-%d %H:%M:%S')

    # --- 4. UPLOAD FORECAST ---
    fg_forecast = fs.get_or_create_feature_group(
        name="karachi_aqi_forecast", version=1, 
        primary_key=['year', 'month', 'day', 'hour'], online_enabled=True
    )
    
    # Ensure types match the Feature Group schema exactly
    for col in ['year', 'month', 'day', 'hour']: 
        forecast_df[col] = forecast_df[col].astype('int64')
    
    print(f"📤 Uploading Forecast from winner {best_model_name}...")
    fg_forecast.insert(forecast_df, write_options={"wait_for_job": False})
    print(f"✅ Pipeline Successfully Finished!")

if __name__ == "__main__":
    run_pipeline()
