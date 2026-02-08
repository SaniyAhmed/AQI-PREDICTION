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
    # ✅ YES: Taking next 3 days forecast from Open-Meteo
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
    
    # ✅ YES: Pulling training data from Feature Group 'karachi_aqi' version 4
    print("🎬 Reading Data from Feature Group: karachi_aqi (v4)...")
    fg = fs.get_feature_group(name="karachi_aqi", version=4)
    full_df = fg.read()
    
    latest_row = full_df.sort_values(['year', 'month', 'day', 'hour']).iloc[-1]
    latest_actuals = latest_row.to_dict()
    current_aqi = float(latest_actuals.get('aqi'))
    print(f"📡 Dynamically Fetched Current AQI: {current_aqi:.2f}")

    X = full_df.drop(columns=["aqi"])
    y = full_df["aqi"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    feature_names = X_train.columns.tolist()
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train.dropna())
    X_test_s = scaler.transform(X_test.dropna())
    y_train = y_train.loc[X_train.dropna().index]
    y_test = y_test.loc[X_test.dropna().index]

    # ✅ YES: TOURNAMENT with 3 Models & Deep Random Forest Tuning
    models_to_train = [
        {
            "name": "RandomForest",
            "estimator": RandomForestRegressor(random_state=42),
            "params": {
                "n_estimators": [100, 300], 
                "max_depth": [10, 20],
                "max_features": ["sqrt"]
            }
        },
        {
            "name": "XGBoost",
            "estimator": XGBRegressor(random_state=42),
            "params": {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}
        },
        {
            "name": "SVR",
            "estimator": SVR(),
            "params": {"C": [1, 10, 100], "epsilon": [0.1, 0.2]}
        }
    ]

    best_overall_model = None
    best_rmse = float('inf')
    winning_model_name = ""

    print("\n🏆 STARTING EXTENDED MODEL TOURNAMENT...")
    for m in models_to_train:
        grid = GridSearchCV(m["estimator"], m["params"], cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
        grid.fit(X_train_s, y_train)
        
        train_rmse = root_mean_squared_error(y_train, grid.predict(X_train_s))
        test_rmse = root_mean_squared_error(y_test, grid.predict(X_test_s))
        
        print(f"--- {m['name']} ---")
        print(f"   Best Params: {grid.best_params_}")
        print(f"   Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
        print(f"   Overfitting Gap: {abs(train_rmse - test_rmse):.4f}")

        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_overall_model = grid.best_estimator_
            winning_model_name = m["name"]

    print(f"\n🥇 TOURNAMENT WINNER: {winning_model_name} (RMSE: {best_rmse:.4f})")

    # ✅ Save Local Files
    model_dir = "karachi_aqi_model"
    if os.path.exists(model_dir): shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    joblib.dump(best_overall_model, f"{model_dir}/model.pkl")
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")

    # ✅ REGISTER MODEL (Using standard schema inference)
    aqi_model = mr.python.create_model(
        name="karachi_aqi_model",
        metrics={"test_rmse": best_rmse},
        description=f"Winner: {winning_model_name} trained on FG v4."
    )
    aqi_model.save(model_dir)

    # 2. RECURSIVE FORECAST
    X_f_base, times = get_forecast_features(feature_names, latest_actuals)
    predictions = []
    moving_state_aqi = current_aqi 

    for i in range(len(X_f_base)):
        row = X_f_base.iloc[[i]].copy()
        if 'aqi_lag_1' in row.columns:
            row['aqi_lag_1'] = moving_state_aqi
        
        suggestion = best_overall_model.predict(scaler.transform(row))[0]
        momentum = 0.85 
        next_step = (moving_state_aqi * momentum) + (suggestion * (1 - momentum))
        predictions.append(float(next_step))
        moving_state_aqi = next_step 

    # 3. ANALYSIS & AVERAGES
    forecast_df = X_f_base[['year', 'month', 'day', 'hour']].copy()
    forecast_df['predicted_aqi'] = [round(p, 2) for p in predictions]
    forecast_df['prediction_timestamp'] = pd.to_datetime(times)

    daily_stats = forecast_df.groupby(forecast_df['prediction_timestamp'].dt.date)['predicted_aqi'].mean()
    total_avg = daily_stats.mean()

    print("\n📅 DAILY AQI AVERAGES (Forecasted):")
    for date, avg in daily_stats.items():
        print(f"   {date}: {avg:.2f}")
    
    print(f"\n⭐ TOTAL 3-DAY AVERAGE FORECAST: {total_avg:.2f}")

    # ✅ FINAL UPLOAD
    forecast_df['prediction_timestamp'] = forecast_df['prediction_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    fg_forecast = fs.get_or_create_feature_group(
        name="karachi_aqi_forecast", version=1, 
        primary_key=['year', 'month', 'day', 'hour'], online_enabled=True
    )
    
    print("\n📤 Uploading forecast to Hopsworks...")
    # Using simple insert options to avoid remote timeout errors
    fg_forecast.insert(forecast_df, write_options={"wait_for_job": False})
    
    print("\n✅ Professional Pipeline complete.")

if __name__ == "__main__":
    run_pipeline()
