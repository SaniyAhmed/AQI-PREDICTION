import os
# This MUST be at the very top to stop the SDK from even trying to use the blocked service
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
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
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
        if api in df_forecast.columns: prep[local] = df_forecast[api].astype('float64')
    for col in trained_columns:
        if col not in prep.columns: prep[col] = 0.0
    return prep[trained_columns], df_forecast['time']

def run_pipeline():
    # 1. Login to Hopsworks
    try:
        project = hopsworks.login(api_key_value=os.getenv('MY_HOPSWORK_KEY'))
        fs = project.get_feature_store()
        mr = project.get_model_registry()
    except Exception as e:
        print(f"❌ Login Failed: {e}")
        return

    print("📥 Retrieving Data via REST API (GitHub Compatible)...")
    # 2. Get Data using the 'use_api' flag to bypass the Arrow Flight error
    fg = fs.get_feature_group(name="karachi_aqi", version=1)
    # This specifically forces the SDK to use standard HTTPS
    df = fg.select_all().read(read_options={"use_api": True})
    
    if df.empty:
        print("❌ Error: No data found in Feature Group. Check Hopsworks UI.")
        return

    # 3. Prep Data
    target_col = 'pm25' 
    if target_col not in df.columns:
        target_col = [col for col in df.columns if 'pm2' in col.lower()][0]
        
    y = df[[target_col]]
    X = df.drop(columns=[target_col])

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, y_train = X_train.dropna(), y_train.loc[X_train.dropna().index]
    X_test, y_test = X_test.dropna(), y_test.loc[X_test.dropna().index]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Tournament Setup
    param_grids = {
        "RandomForest": {"n_estimators": [50, 100], "max_depth": [10]},
        "XGBoost": {"n_estimators": [50], "learning_rate": [0.1]},
        "SVR": {"C": [1]}
    }
    base_models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "SVR": SVR(kernel='rbf')
    }

    print("\n🏆 STARTING TOURNAMENT...")
    best_model, best_rmse, best_model_name = None, float('inf'), ""

    for name, model in base_models.items():
        print(f"🔍 Training {name}...")
        # Using simple Fit for reliability, but kept the logic structure
        search = RandomizedSearchCV(model, param_grids[name], n_iter=1, cv=2, scoring='neg_root_mean_squared_error')
        search.fit(X_train_scaled, y_train.values.ravel())
        cv_rmse = -search.best_score_
        
        # Save Model to local disk
        model_dir = f"model_dir_{name.lower()}"
        if os.path.exists(model_dir): shutil.rmtree(model_dir)
        os.makedirs(model_dir)
        joblib.dump(search.best_estimator_, f"{model_dir}/karachi_aqi_model.pkl")
        joblib.dump(scaler, f"{model_dir}/scaler.pkl")

        # Register in Hopsworks (Uses standard HTTPS - safe)
        h_model = mr.python.create_model(name="karachi_aqi_model", metrics={"rmse": cv_rmse}, description=f"Model: {name}")
        h_model.save(model_dir)

        if cv_rmse < best_rmse:
            best_rmse, best_model, best_model_name = cv_rmse, search.best_estimator_, name

    print(f"⭐ WINNER: {best_model_name} with RMSE: {best_rmse}")

    # 6. Forecast for next 72 hours
    X_f, times = get_forecast_features(X_train.columns.tolist())
    preds = best_model.predict(scaler.transform(X_f))
    forecast_df = X_f[['year', 'month', 'day', 'hour']].copy()
    forecast_df['predicted_aqi'] = preds.round(2).astype('float64')
    forecast_df['prediction_timestamp'] = times.dt.strftime('%Y-%m-%d %H:%M:%S')

    # 7. Upload Forecast to Hopsworks
    print("🚀 Uploading predictions...")
    fg_forecast = fs.get_or_create_feature_group(
        name="karachi_aqi_forecast", 
        version=1, 
        primary_key=['year', 'month', 'day', 'hour'], 
        online_enabled=True
    )
    # Ensure types match Hopsworks expectations
    for col in ['year', 'month', 'day', 'hour']: forecast_df[col] = forecast_df[col].astype('int64')
    
    fg_forecast.insert(forecast_df, write_options={"wait_for_job": False})
    print("✅ SUCCESS: Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()
