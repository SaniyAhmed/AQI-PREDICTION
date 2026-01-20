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
from hsml.model_schema import ModelSchema

# --- CONFIGURATION ---
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AQICN_TOKEN = "c6a73cfca3b6bb6d9930dabdd8c0eea057e29278"
AQICN_URL = f"https://api.waqi.info/feed/geo:{KARACHI_LAT};{KARACHI_LON}/?token={AQICN_TOKEN}"

def get_forecast_features():
    print("🌐 Fetching Weather (Open-Meteo) + Pollutant Forecasts (AQICN)...")
    w_params = {"latitude": KARACHI_LAT, "longitude": KARACHI_LON, "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,dew_point_2m", "forecast_days": 3}
    w_res = requests.get(FORECAST_URL, params=w_params).json()
    df_forecast = pd.DataFrame(w_res["hourly"])
    df_forecast['time'] = pd.to_datetime(df_forecast['time'])

    aq_res = requests.get(AQICN_URL).json()
    pm25_daily_forecast = aq_res['data']['forecast']['daily']['pm25']
    
    def map_daily_pm25(row_time):
        date_str = row_time.strftime('%Y-%m-%d')
        for day_data in pm25_daily_forecast:
            if day_data['day'] == date_str: return float(day_data['avg'])
        return float(pm25_daily_forecast[0]['avg'])

    prep = pd.DataFrame()
    prep['year'] = df_forecast['time'].dt.year.astype('int64')
    prep['month'] = df_forecast['time'].dt.month.astype('int64')
    prep['day'] = df_forecast['time'].dt.day.astype('int64')
    prep['hour'] = df_forecast['time'].dt.hour.astype('int64')
    prep['weekday'] = df_forecast['time'].dt.weekday.astype('float64')
    prep['dew_point'] = df_forecast['dew_point_2m'].astype('float64')
    prep['wind_speed'] = df_forecast['wind_speed_10m'].astype('float64')
    prep['pm25'] = df_forecast['time'].apply(map_daily_pm25).astype('float64')
    return prep, df_forecast['time']

def run_pipeline():
    # 1. LOGIN
    api_key = os.getenv('MY_HOPSWORK_KEY') 
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # 2. FETCH DATA (THE ROBUST SQL WAY)
    print("📥 Pulling training data via SQL (Bypassing Arrow)...")
    
    # Get the Feature Group directly instead of the Feature View to avoid Arrow initialization
    fg = fs.get_feature_group(name="karachi_aqi", version=1)
    
    # Read using Hive SQL - this is the most compatible mode for GitHub Runners
    df = fg.read(read_options={"use_hive": True})
    
    # Manually split since we are reading raw data to be safe
    df = df.sort_values(['year', 'month', 'day', 'hour']).dropna()
    
    # Define features and target based on your schema
    target = 'aqi'
    # Drop target and metadata for X
    X = df.drop(columns=[target], errors='ignore')
    y = df[[target]]

    # Split 80/20
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"✅ Data Loaded. Training on {len(X_train)} rows.")

    # 3. TOURNAMENT
    models = {"XGBoost": XGBRegressor(n_estimators=100), "RandomForest": RandomForestRegressor(n_estimators=100), "Ridge": Ridge(alpha=1.0)}
    best_model, best_rmse, best_name = None, float('inf'), ""
    
    for name, model in models.items():
        model.fit(X_train, y_train.values.ravel())
        rmse = root_mean_squared_error(y_test, model.predict(X_test))
        print(f"Model: {name} | RMSE: {rmse:.4f}")
        if rmse < best_rmse: best_rmse, best_model, best_name = rmse, model, name

    # 4. SAVE MODEL TO REGISTRY
    print(f"🌟 Registering Best Model: {best_name}")
    model_dir = "karachi_aqi_model"
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    joblib.dump(best_model, f"{model_dir}/model.pkl")

    karachi_model = mr.python.create_model(
        name="karachi_aqi_model", 
        metrics={"rmse": best_rmse},
        description=f"Best model ({best_name}) trained on daily update."
    )
    karachi_model.save(model_dir)
    
    # 5. FORECAST & INSERT
    X_forecast, timestamps = get_forecast_features()
    last_known = X_train.iloc[-1]
    
    # Ensure forecast has all necessary columns from X_train
    for col in X_train.columns:
        if col not in X_forecast.columns:
            X_forecast[col] = last_known[col]
            
    X_forecast = X_forecast[X_train.columns] # Match order
    
    preds = best_model.predict(X_forecast)
    forecast_df = X_forecast[['year', 'month', 'day', 'hour']].copy()
    forecast_df['predicted_aqi'] = preds.round(2).astype('float64')
    forecast_df['prediction_timestamp'] = timestamps.dt.strftime('%Y-%m-%d %H:%M:%S')

    try:
        forecast_fg = fs.get_feature_group(name="karachi_aqi_forecast", version=1)
        forecast_fg.insert(forecast_df, write_options={"wait_for_job": False})
        print(f"✅ SUCCESS! Forecast and Model Version {karachi_model.version} stored.")
    except Exception as e:
        print(f"❌ Error during insertion: {e}")

if __name__ == "__main__":
    run_pipeline()
