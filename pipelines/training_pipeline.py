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
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

# --- CONFIG ---
KARACHI_LAT, KARACHI_LON = 24.8607, 67.0011

def get_forecast_features(trained_columns, latest_actuals):
    res = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": KARACHI_LAT, "longitude": KARACHI_LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,dew_point_2m,pm2_5,pm10,carbon_monoxide",
        "forecast_days": 3
    }).json()
    
    df_f = pd.DataFrame(res["hourly"])
    df_f['time'] = pd.to_datetime(df_f['time'])
    
    prep = pd.DataFrame({
        'year': df_f['time'].dt.year.astype('int64'), 
        'month': df_f['time'].dt.month.astype('int64'),
        'day': df_f['time'].dt.day.astype('int64'), 
        'hour': df_f['time'].dt.hour.astype('int64'),
        'weekday': df_f['time'].dt.weekday.astype('int64')
    })
    
    name_map = {
        'pm2_5':'pm25', 'pm10':'pm10', 'carbon_monoxide':'co',
        'temperature_2m':'temperature', 'relative_humidity_2m':'humidity',
        'wind_speed_10m':'wind_speed', 'dew_point_2m':'dew_point'
    }
    
    for api, loc in name_map.items(): 
        if api in df_f.columns: prep[loc] = df_f[api].astype('float64')
    
    # Initialize missing columns with latest known actuals
    for c in trained_columns:
        if c not in prep.columns:
            prep[c] = latest_actuals.get(c, 0.0)
            if c == "dew_pointt" and "dew_point" in prep.columns:
                prep["dew_pointt"] = prep["dew_point"]
        
    return prep[trained_columns], df_f['time']

def run_pipeline():
    project = hopsworks.login(api_key_value=os.getenv('MY_HOPSWORK_KEY'))
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    
    print("\n📥 Fetching Data from Version 4...")
    fg = fs.get_feature_group(name="karachi_aqi", version=4)
    full_df = fg.read()
    
    latest_row = full_df.sort_values(['year', 'month', 'day', 'hour']).iloc[-1]
    latest_actuals = latest_row.to_dict()
    current_aqi = latest_actuals.get('aqi')
    print(f"📍 CURRENT AQI: {current_aqi:.2f}")

    X = full_df.drop(columns=["aqi"])
    y = full_df[["aqi"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    feature_names = X_train.columns.tolist()
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train.dropna())
    y_train_s = y_train.loc[X_train.dropna().index].values.ravel()

    # We use XGBoost as the champion for this demonstration
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train_s, y_train_s)

    # --- RECURSIVE FORECAST ---
    print("\n🔮 Running Recursive Forecast...")
    X_f_base, times = get_forecast_features(feature_names, latest_actuals)
    
    predictions = []
    # Start with the real current values
    last_aqi = current_aqi
    last_pm25 = latest_actuals.get('pm25', last_aqi * 0.8) # Rough estimate if missing

    for i in range(len(X_f_base)):
        current_step = X_f_base.iloc[[i]].copy()
        
        # Update lag features with the PREVIOUS prediction
        if 'aqi_lag_1' in current_step.columns: current_step['aqi_lag_1'] = last_aqi
        if 'pm25_lag_1' in current_step.columns: current_step['pm25_lag_1'] = last_pm25
        
        # Predict
        step_s = scaler.transform(current_step)
        pred = model.predict(step_s)[0]
        
        # Store and update for next hour
        predictions.append(float(pred))
        last_aqi = pred
        # If your model uses pm25 as a feature, we update it too
        if 'pm25' in current_step.columns: last_pm25 = current_step['pm25'].values[0]

    forecast_df = X_f_base[['year', 'month', 'day', 'hour']].copy()
    forecast_df['predicted_aqi'] = [round(p, 2) for p in predictions]
    forecast_df['predicted_aqi'] = forecast_df['predicted_aqi'].astype('float64')
    forecast_df['prediction_timestamp'] = times.dt.strftime('%Y-%m-%d %H:%M:%S')

    print("\n📊 RECURSIVE RESULTS (First 5 hours):")
    print(forecast_df[['prediction_timestamp', 'predicted_aqi']].head())

    # Save
    fg_forecast = fs.get_or_create_feature_group(
        name="karachi_aqi_forecast", version=1, 
        primary_key=['year', 'month', 'day', 'hour'], online_enabled=True
    )
    fg_forecast.insert(forecast_df, write_options={"wait_for_job": False})
    print("\n✅ Recursive forecast successfully uploaded.")

if __name__ == "__main__":
    run_pipeline()
