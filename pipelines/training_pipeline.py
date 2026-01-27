import os
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
    trained_columns = [str(col) for col in trained_columns]  # ✅ Ensure plain strings
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
        if api in df_forecast.columns: prep[local] = df_forecast[api].astype('float64')
    for col in trained_columns:
        if col not in prep.columns: prep[col] = 0.0
    return prep[trained_columns], df_forecast['time']

def run_pipeline():
    project = hopsworks.login(api_key_value=os.getenv('MY_HOPSWORK_KEY'))
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    feature_view = fs.get_feature_view(name="karachi_aqi_view", version=3)
    
    print("📥 Retrieving Training Data...")
    X_train, X_test, y_train, y_test = feature_view.train_test_split(test_size=0.2)

    # ✅ Convert all columns to plain strings to prevent BinderError
    X_train.columns = [str(col) for col in X_train.columns]
    X_test.columns = [str(col) for col in X_test.columns]

    X_train, y_train = X_train.dropna(), y_train.loc[X_train.dropna().index]
    X_test, y_test = X_test.dropna(), y_test.loc[X_test.dropna().index]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 🏆 TOURNAMENT SETUP
    param_grids = {
        "RandomForest": {"n_estimators": [50, 100], "max_depth": [10, 20], "min_samples_split": [2, 5]},
        "XGBoost": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
        "SVR": {"C": [1, 10], "epsilon": [0.1]}
    }
    base_models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "SVR": SVR(kernel='rbf')
    }

    print("\n🏆 STARTING TOURNAMENT (All models will be registered)...")
    print("-" * 50)
    best_model, best_rmse, best_model_name = None, float('inf'), ""

    for name, model in base_models.items():
        print(f"🔍 Tuning {name}...")
        search = RandomizedSearchCV(
            model, param_grids[name], n_iter=3, cv=3, 
            scoring='neg_root_mean_squared_error', n_jobs=-1
        )
        search.fit(X_train_scaled, y_train.values.ravel())
        cv_rmse = -search.best_score_
        
        test_preds = search.best_estimator_.predict(X_test_scaled)
        test_rmse = root_mean_squared_error(y_test, test_preds)
        
        print(f"   📊 {name:12} -> CV RMSE: {cv_rmse:.4f} | TEST RMSE: {test_rmse:.4f}")

        # --- LOGIC TO STORE EACH MODEL ---
        iter_model_dir = f"model_dir_{name.lower()}"
        if os.path.exists(iter_model_dir): shutil.rmtree(iter_model_dir)
        os.makedirs(iter_model_dir)
        
        joblib.dump(search.best_estimator_, f"{iter_model_dir}/karachi_aqi_model.pkl", compress=3)
        joblib.dump(scaler, f"{iter_model_dir}/scaler.pkl")

        # Register model version
        current_model = mr.python.create_model(
            name="karachi_aqi_model", 
            metrics={"cv_rmse": cv_rmse, "test_rmse": test_rmse}, 
            description=f"Tournament Participant: {name}"
        )
        current_model.save(iter_model_dir)
        print(f"✅ {name} registered as a new version.")

        if cv_rmse < best_rmse:
            best_rmse, best_model, best_model_name = cv_rmse, search.best_estimator_, name

    print("-" * 50)
    print(f"⭐ TOURNAMENT WINNER: {best_model_name} (CV RMSE: {best_rmse:.4f})")
    print("✅ All models synced to Registry!")

    # 6. FORECAST GENERATION (Uses the tournament winner)
    X_f, times = get_forecast_features(X_train.columns.tolist())
    preds = best_model.predict(scaler.transform(X_f))
    forecast_df = X_f[['year', 'month', 'day', 'hour']].copy()
    forecast_df['predicted_aqi'] = preds.round(2).astype('float64')
    forecast_df['prediction_timestamp'] = times.dt.strftime('%Y-%m-%d %H:%M:%S')

    # 7. FORECAST UPLOAD (WITH RESILIENCE)
    print("🚀 Preparing Forecast Upload...")
    fg = fs.get_or_create_feature_group(
        name="karachi_aqi_forecast", version=1, 
        primary_key=['year', 'month', 'day', 'hour'], online_enabled=True
    )
    
    for col in ['year', 'month', 'day', 'hour']: 
        forecast_df[col] = forecast_df[col].astype('int64')
    
    for attempt in range(3):
        try:
            print(f"📤 Uploading Forecast (Attempt {attempt+1})...")
            fg.insert(forecast_df, write_options={"start_offline_materialization": False, "wait_for_job": False})
            print(f"✅ SUCCESS! Karachi forecast is live.")
            break
        except Exception as e:
            print(f"⚠️ Upload attempt failed: {e}")
            if attempt < 2: 
                print("⏳ Retrying in 10 seconds...")
                time.sleep(10)
            else:
                print("❌ Final attempt failed. Check Hopsworks UI.")

if __name__ == "__main__":
    run_pipeline()
