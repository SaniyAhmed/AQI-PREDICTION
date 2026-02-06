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
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

# --- CONFIG ---
KARACHI_LAT, KARACHI_LON = 24.8607, 67.0011

def get_forecast_features(trained_columns, latest_actuals):
    res = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": KARACHI_LAT, "longitude": KARACHI_LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,dew_point_2m,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
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
        'nitrogen_dioxide':'no2', 'sulphur_dioxide':'so2', 'ozone':'o3',
        'temperature_2m':'temperature', 'relative_humidity_2m':'humidity',
        'wind_speed_10m':'wind_speed', 'dew_point_2m':'dew_point'
    }
    
    for api, loc in name_map.items(): 
        if api in df_f.columns: prep[loc] = df_f[api].astype('float64')
    
    for c in trained_columns:
        if c not in prep.columns:
            prep[c] = latest_actuals.get(c, 0.0)
            if c == "dew_pointt" and "dew_point" in prep.columns:
                prep["dew_pointt"] = prep["dew_point"]
        
    return prep[trained_columns].ffill().bfill(), df_f['time']

def run_pipeline():
    project = hopsworks.login(api_key_value=os.getenv('MY_HOPSWORK_KEY'))
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    
    print("\n📥 Fetching Latest Data...")
    fg_list = fs.get_feature_groups(name="karachi_aqi")
    latest_fg = sorted(fg_list, key=lambda x: x.version)[-1]
    full_df = latest_fg.read()
    
    latest_row = full_df.sort_values(['year', 'month', 'day', 'hour']).iloc[-1]
    latest_actuals = latest_row.to_dict()
    print(f"📍 CURRENT AQI (Latest Record): {latest_actuals.get('aqi'):.2f}")

    X = full_df.drop(columns=["aqi"])
    y = full_df[["aqi"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, y_train = X_train.dropna(), y_train.loc[X_train.dropna().index]
    X_test, y_test = X_test.dropna(), y_test.loc[X_test.dropna().index]

    feature_names = X_train.columns.tolist()
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Note: Added 'reg:squarederror' to avoid base_score issues
    base_models = {
        "RandomForest": RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, objective='reg:squarederror', random_state=42)
    }

    best_m, best_score, best_name = None, float('inf'), ""

    for name, model in base_models.items():
        model.fit(X_train_s, y_train.values.ravel())
        t_rmse = root_mean_squared_error(y_test, model.predict(X_test_s))
        if t_rmse < best_score:
            best_score, best_m, best_name = t_rmse, model, name

    print(f"\n🏆 CHAMPION MODEL: {best_name} (Test RMSE: {best_score:.4f})")

    # --- FORECAST GENERATION ---
    X_f, times = get_forecast_features(feature_names, latest_actuals)
    preds = best_m.predict(scaler.transform(X_f))
    
    forecast_df = X_f[['year', 'month', 'day', 'hour']].copy()
    
    # CRITICAL FIX: Cast to float64 to match Hopsworks 'double' requirement
    forecast_df['predicted_aqi'] = preds.astype('float64').round(2)
    forecast_df['prediction_timestamp'] = times.dt.strftime('%Y-%m-%d %H:%M:%S')

    print("\n🔮 3-DAY AQI FORECAST (Summary):")
    print("=" * 45)
    summary_print = forecast_df.iloc[::6] 
    for _, row in summary_print.iterrows():
        print(f"📅 {row['prediction_timestamp']} | Predicted AQI: {row['predicted_aqi']:>6}")
    print("=" * 45)

    # Save to Hopsworks
    fg_forecast = fs.get_or_create_feature_group(
        name="karachi_aqi_forecast", version=1, 
        primary_key=['year', 'month', 'day', 'hour'], online_enabled=True
    )
    
    # Insert with explicit schema check bypass if it still complains, 
    # but the astype('float64') usually solves it!
    fg_forecast.insert(forecast_df, write_options={"wait_for_job": False})
    print("✅ Forecast uploaded successfully.")

if __name__ == "__main__":
    run_pipeline()
