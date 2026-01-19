import hopsworks
import pandas as pd
import requests
import joblib
import os
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error

# --- CONFIGURATION ---
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

def get_aqi_status(aqi):
    if aqi <= 50: return "🟢 Good"
    elif aqi <= 100: return "🟡 Moderate"
    elif aqi <= 150: return "🟠 Unhealthy (Sensitive)"
    elif aqi <= 200: return "🔴 Unhealthy"
    else: return "🟣 Very Unhealthy"

def get_forecast_features():
    params = {
        "latitude": KARACHI_LAT, "longitude": KARACHI_LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,dew_point_2m",
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
    prep['weekday'] = df_forecast['time'].dt.weekday.astype('float64')
    prep['dew_point'] = df_forecast['dew_point_2m'].astype('float64')
    prep['wind_speed'] = df_forecast['wind_speed_10m'].astype('float64')
    
    return prep, df_forecast['time']

def run_pipeline():
    # Load API Key securely
    api_key = os.getenv('MY_HOPSWORK_KEY') 
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()

    # 1. FETCH DATA (Using version 2 as you defined)
    feature_view = fs.get_feature_view(name="karachi_aqi_view", version=2)
    X_train, X_test, y_train, y_test = feature_view.train_test_split(test_size=0.2)

    # 2. MODEL SELECTION
    models = {
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10),
        "Ridge": Ridge(alpha=1.0)
    }

    best_model, best_rmse = None, float('inf')
    for name, model in models.items():
        model.fit(X_train, y_train.values.ravel())
        preds = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, preds)
        if rmse < best_rmse:
            best_rmse, best_model = rmse, model

    # 3. GENERATE 3-DAY FORECAST
    X_forecast, timestamps = get_forecast_features()
    last_known = X_train.iloc[-1]
    
    # Use last known pollutants to fill forecast features
    for col in ['pm25', 'pm10', 'co', 'aqi_lag_1', 'aqi_lag_2', 'pm25_lag_1']:
        X_forecast[col] = last_known[col]
    X_forecast['aqi_change_rate'] = 0.0

    # Ensure schema order
    hopsworks_features = [
        'weekday', 'month', 'dew_point', 'aqi_lag_1', 'year', 'aqi_lag_2', 
        'hour', 'co', 'aqi_change_rate', 'pm10', 'pm25_lag_1', 
        'day', 'pm25', 'wind_speed'
    ]
    preds = best_model.predict(X_forecast[hopsworks_features])

    # 4. PREPARE DATAFRAME
    forecast_df = X_forecast[['year', 'month', 'day', 'hour']].copy()
    forecast_df['predicted_aqi'] = preds.round(2).astype('float64')
    forecast_df['prediction_timestamp'] = timestamps.dt.strftime('%Y-%m-%d %H:%M:%S')

    # 5. INSERT INTO EXISTING FEATURE GROUP (NO CREATE)
    try:
        print("🚀 Accessing existing forecast feature group...")
        forecast_fg = fs.get_feature_group(name="karachi_aqi_forecast", version=1)
        
        # We add wait_for_job=False to prevent the connection timeout error
        print("📥 Inserting predictions...")
        forecast_fg.insert(forecast_df, write_options={"wait_for_job": False})
        print("✅ Data sent to Hopsworks! (Background job is processing)")
        
    except Exception as e:
        print(f"❌ Error during Hopsworks insertion: {e}")

if __name__ == "__main__":
    run_pipeline()
