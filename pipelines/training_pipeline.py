import os
os.environ["HSFS_DISABLE_FLIGHT_CLIENT"] = "True"

import requests
import pandas as pd
import hopsworks
import joblib
import shutil
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# --- CONFIG ---
KARACHI_LAT, KARACHI_LON = 24.8607, 67.0011

def get_forecast_features(trained_columns, latest_actuals):
    # Fetch weather + air quality forecast
    res = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": KARACHI_LAT, "longitude": KARACHI_LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,pm2_5,pm10",
        "forecast_days": 3
    }).json()
    
    df_f = pd.DataFrame(res["hourly"])
    df_f['time'] = pd.to_datetime(df_f['time'])
    
    # Map API names to your Feature Group names
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
    
    # Fill any missing trained columns with latest actuals
    for c in trained_columns:
        if c not in prep.columns:
            prep[c] = latest_actuals.get(c, 0.0)
            
    return prep[trained_columns], df_f['time']

def run_pipeline():
    project = hopsworks.login(api_key_value=os.getenv('MY_HOPSWORK_KEY'))
    fs = project.get_feature_store()
    
    print("\n📥 Loading Feature Group (Version 4)...")
    fg = fs.get_feature_group(name="karachi_aqi", version=4)
    full_df = fg.read()
    
    # Get the "Anchor"
    latest_row = full_df.sort_values(['year', 'month', 'day', 'hour']).iloc[-1]
    latest_actuals = latest_row.to_dict()
    current_aqi = float(latest_actuals.get('aqi', 150.0))
    avg_aqi = float(full_df['aqi'].mean()) # The "Normal" state for Karachi

    X = full_df.drop(columns=["aqi"])
    y = full_df[["aqi"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    feature_names = X_train.columns.tolist()
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train.dropna())
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=4)
    model.fit(X_train_s, y_train.loc[X_train.dropna().index].values.ravel())

    # --- RECURSIVE FORECAST WITH NAN PROTECTION ---
    X_f_base, times = get_forecast_features(feature_names, latest_actuals)
    
    predictions = []
    current_lag_aqi = current_aqi

    print(f"🚀 Starting Forecast from: {current_aqi:.2f} (Karachi Avg: {avg_aqi:.2f})")

    for i in range(len(X_f_base)):
        row = X_f_base.iloc[[i]].copy()
        
        # 1. Update the Lag
        if 'aqi_lag_1' in row.columns:
            row['aqi_lag_1'] = current_lag_aqi
        
        # 2. Predict using model
        step_s = scaler.transform(row)
        pred = model.predict(step_s)[0]
        
        # 3. ANTI-FREEZE LOGIC:
        # Every hour, pull the prediction 5% closer to the Karachi Average.
        # This prevents the 150 "Flat Line" while avoiding NaN errors.
        if i > 0:
            pred = (pred * 0.90) + (avg_aqi * 0.10)

        predictions.append(float(pred))
        current_lag_aqi = pred 

    forecast_df = X_f_base[['year', 'month', 'day', 'hour']].copy()
    forecast_df['predicted_aqi'] = [round(p, 2) for p in predictions]
    forecast_df['predicted_aqi'] = forecast_df['predicted_aqi'].astype('float64')
    forecast_df['prediction_timestamp'] = times.dt.strftime('%Y-%m-%d %H:%M:%S')

    print("\n📊 FINAL CLEAN RESULTS:")
    print(forecast_df[['prediction_timestamp', 'predicted_aqi']].head(10))

    # Upload
    fg_forecast = fs.get_or_create_feature_group(
        name="karachi_aqi_forecast", version=1, 
        primary_key=['year', 'month', 'day', 'hour'], online_enabled=True
    )
    fg_forecast.insert(forecast_df, write_options={"wait_for_job": False})
    print("\n✅ Forecast success. No more NaNs!")

if __name__ == "__main__":
    run_pipeline()
