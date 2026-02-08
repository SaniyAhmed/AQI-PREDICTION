import os
os.environ["HSFS_DISABLE_FLIGHT_CLIENT"] = "True"

import requests
import pandas as pd
import hopsworks
import joblib
import shutil
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error

# --- CONFIG ---
KARACHI_LAT, KARACHI_LON = 24.8607, 67.0011
OWM_API_KEY = os.getenv('OWM_API_KEY') 

def get_forecast_features(trained_columns, latest_actuals):
    """Fetches weather from Open-Meteo and pollutants from OpenWeatherMap."""
    
    # 1. Fetch Weather Forecast (Open-Meteo)
    weather_res = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": KARACHI_LAT, "longitude": KARACHI_LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "forecast_days": 3
    }).json()
    
    # 2. Fetch Pollutant Forecast (OpenWeatherMap)
    pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={KARACHI_LAT}&lon={KARACHI_LON}&appid={OWM_API_KEY}"
    pollution_res = requests.get(pollution_url).json()
    
    # Parse Weather
    df_w = pd.DataFrame(weather_res["hourly"])
    df_w['time'] = pd.to_datetime(df_w['time'])
    
    # Parse Pollutants (OWM returns data in a 'list')
    pollutant_list = []
    for entry in pollution_res['list']:
        pollutant_list.append({
            'time': pd.to_datetime(entry['dt'], unit='s'),
            'pm25': float(entry['components']['pm2_5']),
            'pm10': float(entry['components']['pm10'])
        })
    df_p = pd.DataFrame(pollutant_list)
    
    # Merge on nearest time (OWM might have different timestamps)
    df_f = pd.merge_asof(df_w.sort_values('time'), df_p.sort_values('time'), on='time', direction='nearest')
    
    prep = pd.DataFrame({
        'year': df_f['time'].dt.year.astype('int64'), 
        'month': df_f['time'].dt.month.astype('int64'),
        'day': df_f['time'].dt.day.astype('int64'), 
        'hour': df_f['time'].dt.hour.astype('int64'),
        'weekday': df_f['time'].dt.weekday.astype('int64'),
        'pm25': df_f['pm25'].astype('float64'),
        'pm10': df_f['pm10'].astype('float64'),
        'wind_speed': df_f['wind_speed_10m'].astype('float64')
    })

    # Robust Imputation
    prep = prep.ffill().bfill().fillna(0.0)
    
    for c in trained_columns:
        if c not in prep.columns:
            prep[c] = latest_actuals.get(c, 0.0)
            
    return prep[trained_columns], df_f['time']

def run_pipeline():
    project = hopsworks.login(api_key_value=os.getenv('MY_HOPSWORK_KEY'))
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    
    print("🎬 Reading Data from Feature Group: karachi_aqi (v4)...")
    fg = fs.get_feature_group(name="karachi_aqi", version=4)
    full_df = fg.read()
    
    full_df = full_df.sort_values(['year', 'month', 'day', 'hour']).dropna()
    latest_row = full_df.iloc[-1]
    latest_actuals = latest_row.to_dict()
    current_aqi = float(latest_actuals.get('aqi'))

    X = full_df.drop(columns=["aqi"])
    y = full_df["aqi"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    feature_names = X_train.columns.tolist()
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("\n🔍 TUNING INDIVIDUAL MODELS...")
    xgb_grid = GridSearchCV(XGBRegressor(random_state=42), {"n_estimators": [300], "max_depth": [3, 4]}, cv=3).fit(X_train_s, y_train)
    svr_grid = GridSearchCV(SVR(), {"C": [1, 10], "kernel": ["rbf"]}, cv=3).fit(X_train_s, y_train)
    rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), {"n_estimators": [500], "max_depth": [8]}, cv=3).fit(X_train_s, y_train)

    print("\n🤝 TRAINING ENSEMBLE...")
    ensemble_model = VotingRegressor([('xgb', xgb_grid.best_estimator_), ('svr', svr_grid.best_estimator_), ('rf', rf_grid.best_estimator_)], weights=[2, 2, 1])
    ensemble_model.fit(X_train_s, y_train)
    
    test_rmse = root_mean_squared_error(y_test, ensemble_model.predict(X_test_s))

    model_dir = "karachi_ensemble_model"
    if os.path.exists(model_dir): shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    joblib.dump(ensemble_model, f"{model_dir}/model.pkl")
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")

    aqi_model = mr.python.create_model(name="karachi_aqi_model", metrics={"test_rmse": test_rmse}, description="Ensemble with OWM data.")
    aqi_model.save(model_dir)

    print("\n🔮 Generating 3-day Forecast...")
    X_f_base, times = get_forecast_features(feature_names, latest_actuals)
    predictions = []
    moving_state_aqi = current_aqi 

    for i in range(len(X_f_base)):
        row = X_f_base.iloc[[i]].copy()
        if 'aqi_lag_1' in row.columns: row['aqi_lag_1'] = moving_state_aqi
        suggestion = ensemble_model.predict(scaler.transform(row))[0]
        next_step = (moving_state_aqi * 0.85) + (suggestion * 0.15)
        predictions.append(float(next_step))
        moving_state_aqi = next_step 

    forecast_df = X_f_base[['year', 'month', 'day', 'hour']].copy()
    forecast_df['predicted_aqi'] = [round(p, 2) for p in predictions]
    forecast_df['prediction_timestamp'] = pd.to_datetime(times).dt.strftime('%Y-%m-%d %H:%M:%S')

    fg_forecast = fs.get_or_create_feature_group(name="karachi_aqi_forecast", version=1, primary_key=['year', 'month', 'day', 'hour'], online_enabled=True)
    fg_forecast.insert(forecast_df, write_options={"wait_for_job": False})
    
    print("\n✅ Professional Pipeline complete using OpenWeatherMap.")

if __name__ == "__main__":
    run_pipeline()
