import os
os.environ["HSFS_DISABLE_FLIGHT_CLIENT"] = "True"

import requests
import pandas as pd
import hopsworks
import joblib
import shutil
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error

# --- CONFIG ---
KARACHI_LAT, KARACHI_LON = 24.8607, 67.0011

def get_forecast_features(trained_columns, latest_actuals):
    res = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": KARACHI_LAT, "longitude": KARACHI_LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,pm2_5,pm10",
        "forecast_days": 3
    }).json()
    
    df_f = pd.DataFrame(res["hourly"])
    df_f['time'] = pd.to_datetime(df_f['time'])
    
    prep = pd.DataFrame({
        'year': df_f['time'].dt.year.astype('int64'), 
        'month': df_f['time'].dt.month.astype('int64'),
        'day': df_f['time'].dt.day.astype('int64'), 
        'hour': df_f['time'].dt.hour.astype('int64'),
        'weekday': df_f['time'].dt.weekday.astype('int64'),
        'pm25': df_f['pm2_5'].ffill().fillna(0).astype('float64'),
        'pm10': df_f['pm10'].ffill().fillna(0).astype('float64'),
        'wind_speed': df_f['wind_speed_10m'].ffill().fillna(0).astype('float64')
    })
    
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
    
    latest_row = full_df.sort_values(['year', 'month', 'day', 'hour']).iloc[-1]
    latest_actuals = latest_row.to_dict()
    current_aqi = float(latest_actuals.get('aqi'))
    
    X = full_df.drop(columns=["aqi"])
    y = full_df["aqi"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    feature_names = X_train.columns.tolist()
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train.dropna())
    X_test_s = scaler.transform(X_test.dropna())
    y_train = y_train.loc[X_train.dropna().index]
    y_test = y_test.loc[X_test.dropna().index]

    # --- REGULARIZED MODEL TOURNAMENT ---
    models_to_train = [
        {
            "name": "RandomForest",
            "estimator": RandomForestRegressor(random_state=42, bootstrap=True),
            "params": {
                "n_estimators": [500], 
                "max_depth": [6, 8], 
                "min_samples_split": [15], 
                "min_samples_leaf": [10],
                "max_features": ["sqrt"],
                "max_samples": [0.7],
                "min_impurity_decrease": [0.01]
            }
        }
    ]

    best_overall_model = None
    best_rmse = float('inf')

    for m in models_to_train:
        grid = GridSearchCV(m["estimator"], m["params"], cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
        grid.fit(X_train_s, y_train)
        best_overall_model = grid.best_estimator_
        best_rmse = root_mean_squared_error(y_test, grid.predict(X_test_s))

    # --- RECURSIVE FORECAST ---
    X_f_base, times = get_forecast_features(feature_names, latest_actuals)
    predictions = []
    moving_state_aqi = current_aqi 

    for i in range(len(X_f_base)):
        row = X_f_base.iloc[[i]].copy()
        if 'aqi_lag_1' in row.columns: row['aqi_lag_1'] = moving_state_aqi
        
        suggestion = best_overall_model.predict(scaler.transform(row))[0]
        next_step = (moving_state_aqi * 0.85) + (suggestion * 0.15)
        predictions.append(float(next_step))
        moving_state_aqi = next_step 

    # --- PREPARING DATA FOR UPLOAD ---
    forecast_df = X_f_base[['year', 'month', 'day', 'hour']].copy()
    forecast_df['predicted_aqi'] = [round(p, 2) for p in predictions]
    forecast_df['prediction_timestamp'] = pd.to_datetime(times)

    # 1. Prepare Hourly Data (Convert timestamp to string for Hopsworks)
    hourly_upload_df = forecast_df.copy()
    hourly_upload_df['prediction_timestamp'] = hourly_upload_df['prediction_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # 2. Prepare Daily Average Data
    daily_stats = forecast_df.groupby(forecast_df['prediction_timestamp'].dt.date).agg({
        'predicted_aqi': 'mean',
        'year': 'first',
        'month': 'first',
        'day': 'first'
    }).reset_index()
    daily_stats.rename(columns={'prediction_timestamp': 'date', 'predicted_aqi': 'daily_avg_aqi'}, inplace=True)
    daily_stats['date'] = daily_stats['date'].astype(str) # Primary key compatibility

    # --- HOPSWORKS UPLOADS ---
    
    # Upload Hourly Predictions
    fg_forecast = fs.get_or_create_feature_group(
        name="karachi_aqi_forecast", version=1, 
        primary_key=['year', 'month', 'day', 'hour'], online_enabled=True
    )
    print("📤 Uploading hourly forecast...")
    fg_forecast.insert(hourly_upload_df, write_options={"wait_for_job": False})

    # Upload Daily Averages (The missing part!)
    fg_daily = fs.get_or_create_feature_group(
        name="karachi_aqi_daily_summary", version=1,
        description="Daily aggregated average AQI forecasts for Karachi.",
        primary_key=['date'], online_enabled=True
    )
    print("📊 Uploading daily average summary...")
    fg_daily.insert(daily_stats, write_options={"wait_for_job": False})
    
    print("\n✅ Hourly and Daily data successfully pushed to Hopsworks.")

if __name__ == "__main__":
    run_pipeline()
