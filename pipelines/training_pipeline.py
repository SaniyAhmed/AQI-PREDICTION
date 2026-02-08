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
    # FIXED URL: api.open-meteo.com
    weather_res = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": KARACHI_LAT, "longitude": KARACHI_LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "forecast_days": 3
    }).json()
    
    pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={KARACHI_LAT}&lon={KARACHI_LON}&appid={OWM_API_KEY}"
    pollution_res = requests.get(pollution_url).json()
    
    df_w = pd.DataFrame(weather_res["hourly"])
    df_w['time'] = pd.to_datetime(df_w['time'])
    
    pollutant_list = []
    for entry in pollution_res['list']:
        pollutant_list.append({
            'time': pd.to_datetime(entry['dt'], unit='s'),
            'pm25': float(entry['components']['pm2_5']),
            'pm10': float(entry['components']['pm10'])
        })
    df_p = pd.DataFrame(pollutant_list)
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

    prep = prep.ffill().bfill().fillna(0.0)
    for c in trained_columns:
        if c not in prep.columns:
            prep[c] = latest_actuals.get(c, 0.0)
            
    return prep[trained_columns], df_f['time']

def run_pipeline():
    project = hopsworks.login(api_key_value=os.getenv('MY_HOPSWORK_KEY'))
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    
    # 1. FETCH DATA
    print("🎬 Reading Data from Feature Group: karachi_aqi (v4)...")
    fg = fs.get_feature_group(name="karachi_aqi", version=4)
    full_df = fg.read().sort_values(['year', 'month', 'day', 'hour']).dropna()
    
    latest_actuals = full_df.iloc[-1].to_dict()
    current_aqi = float(latest_actuals.get('aqi'))

    X = full_df.drop(columns=["aqi"])
    y = full_df["aqi"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    feature_names = X_train.columns.tolist()
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 2. TOURNAMENT: HEAVY REGULARIZATION TO STOP RF OVERFITTING
    print("\n🏆 STARTING REGULARIZED MODEL TOURNAMENT...")
    model_configs = {
        'RandomForest': (RandomForestRegressor(random_state=42), {
            "n_estimators": [500], 
            "max_depth": [6, 8],          # Reduced depth to fight overfitting
            "min_samples_leaf": [10, 20],  # Increased to generalize better
            "max_features": ["sqrt"],
            "ccp_alpha": [0.05, 0.1]       # Stronger pruning
        }),
        'XGBoost': (XGBRegressor(random_state=42), {
            "n_estimators": [300], 
            "max_depth": [3, 4], 
            "learning_rate": [0.05],
            "reg_lambda": [1, 10]
        }),
        'SVR': (SVR(), {
            "C": [0.1, 1], 
            "epsilon": [0.1, 0.2]
        })
    }

    results, best_estimators = [], []
    for name, (model, params) in model_configs.items():
        grid = GridSearchCV(model, params, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1).fit(X_train_s, y_train)
        best_m = grid.best_estimator_
        best_estimators.append((name.lower(), best_m))
        
        tr_rmse = root_mean_squared_error(y_train, best_m.predict(X_train_s))
        te_rmse = root_mean_squared_error(y_test, best_m.predict(X_test_s))
        gap = abs(tr_rmse - te_rmse)
        results.append({'Model': name, 'Train RMSE': tr_rmse, 'Test RMSE': te_rmse, 'Gap': gap})
        print(f"   {name} -> Train RMSE: {tr_rmse:.4f} | Test RMSE: {te_rmse:.4f} | Gap: {gap:.4f}")

    res_df = pd.DataFrame(results).sort_values('Test RMSE')
    winner_name = res_df.iloc[0]['Model']
    winner_rmse = res_df.iloc[0]['Test RMSE']
    print(f"\n🥇 TOURNAMENT WINNER: {winner_name} (RMSE: {winner_rmse:.4f})")

    # 4. ENSEMBLE TRAINING
    ensemble_model = VotingRegressor(best_estimators, weights=[1, 2, 2])
    ensemble_model.fit(X_train_s, y_train)
    ens_test_rmse = root_mean_squared_error(y_test, ensemble_model.predict(X_test_s))

    # 5. MODEL REGISTRY
    model_dir = "karachi_ensemble_model"
    if os.path.exists(model_dir): shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    joblib.dump(ensemble_model, f"{model_dir}/model.pkl")
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")

    aqi_model = mr.python.create_model(
        name="karachi_aqi_model",
        metrics={"test_rmse": ens_test_rmse, "winner_rmse": winner_rmse},
        description=f"Ensemble Model. Winner: {winner_name} (RMSE: {winner_rmse:.4f})."
    )
    aqi_model.save(model_dir)

    # 6. FORECAST GENERATION
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

    # 7. HOURLY DATA PREP
    forecast_df = X_f_base[['year', 'month', 'day', 'hour']].copy()
    forecast_df['predicted_aqi'] = [round(p, 2) for p in predictions]
    forecast_df['prediction_timestamp'] = pd.to_datetime(times).dt.strftime('%Y-%m-%d %H:%M:%S')

    # 8. DAILY & GRAND SUMMARY PREP (Corrected iteration)
    full_prediction_series = pd.Series(predictions)
    grand_avg = round(full_prediction_series.mean(), 2)
    times_dt = pd.to_datetime(times)
    daily_groups = full_prediction_series.groupby(times_dt.dt.date)
    
    summary_data = []
    for date, group in daily_groups:
        summary_data.append({
            "date": str(date),
            "daily_avg_aqi": round(group.mean(), 2),
            "grand_avg_aqi": grand_avg,
            "forecast_type": "3-day-tournament-ensemble"
        })
    summary_df = pd.DataFrame(summary_data)

    # 9. UPLOAD TO HOPSWORKS
    # A. Save Hourly
    fg_forecast = fs.get_or_create_feature_group(name="karachi_aqi_forecast", version=1, primary_key=['year', 'month', 'day', 'hour'], online_enabled=True)
    fg_forecast.insert(forecast_df, write_options={"wait_for_job": False})

    # B. Save Daily Summaries
    target_version = 1
    try:
        fg_summary = fs.get_or_create_feature_group(
            name="karachi_aqi_daily_summary", version=target_version,
            primary_key=['date'], online_enabled=True,
            description="Daily and 3-day Grand Average AQI forecasts."
        )
        fg_summary.insert(summary_df, write_options={"wait_for_job": False})
    except Exception as e:
        print(f"⚠️ Version {target_version} fallback. Creating/Updating next version.")
        fg_summary = fs.get_or_create_feature_group(
            name="karachi_aqi_daily_summary", version=target_version + 1,
            primary_key=['date'], online_enabled=True
        )
        fg_summary.insert(summary_df, write_options={"wait_for_job": False})
    
    print(f"\n📊 3-DAY GRAND AVERAGE: {grand_avg}")
    print("\n✅ Pipeline complete. API URL fixed and summaries stored.")

if __name__ == "__main__":
    run_pipeline()
