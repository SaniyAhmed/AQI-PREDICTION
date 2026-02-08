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

def get_forecast_features(trained_columns, latest_actuals):
    # 1. TAKING NEXT THREE DAYS FORECAST FROM OPEN METEO
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
    
    # 2. TAKING DATA FROM HOPSWORK FEATURE GROUP karachi_aqi (v4)
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

    # 3. HYPERPARAMETER TUNING FOR ALL THREE MODELS
    print("\n🔍 TUNING INDIVIDUAL MODELS...")
    
    # XGBoost Tuning
    xgb_grid = GridSearchCV(
        XGBRegressor(objective='reg:squarederror', random_state=42),
        {"n_estimators": [300], "max_depth": [3, 4], "learning_rate": [0.05], "subsample": [0.8]},
        cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1
    ).fit(X_train_s, y_train)

    # SVR Tuning
    svr_grid = GridSearchCV(
        SVR(),
        {"C": [1, 10], "epsilon": [0.1, 0.2], "kernel": ["rbf"]},
        cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1
    ).fit(X_train_s, y_train)

    # Random Forest Tuning
    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        {"n_estimators": [500], "max_depth": [8, 10], "min_samples_leaf": [10], "max_features": ["sqrt"]},
        cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1
    ).fit(X_train_s, y_train)

    # 4. TRAINING ENSEMBLE WITH TUNED MODELS
    print("\n🤝 TRAINING ENSEMBLE (VOTING REGRESSOR)...")
    ensemble_model = VotingRegressor(
        estimators=[
            ('xgb', xgb_grid.best_estimator_), 
            ('svr', svr_grid.best_estimator_), 
            ('rf', rf_grid.best_estimator_)
        ],
        weights=[2, 2, 1] 
    )
    ensemble_model.fit(X_train_s, y_train)
    
    # 5. PRINTING TRAIN/TEST RMSE & OVERFITTING GAP
    train_rmse = root_mean_squared_error(y_train, ensemble_model.predict(X_train_s))
    test_rmse = root_mean_squared_error(y_test, ensemble_model.predict(X_test_s))
    
    print(f"--- Ensemble Performance ---")
    print(f"   Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
    print(f"   Overfitting Gap: {abs(train_rmse - test_rmse):.4f}")

    # 6. STORING ENSEMBLE IN MODEL REGISTRY karachi_aqi_model
    model_dir = "karachi_ensemble_model"
    if os.path.exists(model_dir): shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    joblib.dump(ensemble_model, f"{model_dir}/model.pkl")
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")

    aqi_model = mr.python.create_model(
        name="karachi_aqi_model",
        metrics={"test_rmse": test_rmse},
        description="Best-in-class Ensemble (Tuned XGB, SVR, RF)."
    )
    aqi_model.save(model_dir)

    # 7. RECURSIVE FORECAST
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

    # 8. STORING PREDICTIONS IN karachi_aqi_forecast v1
    forecast_df = X_f_base[['year', 'month', 'day', 'hour']].copy()
    forecast_df['predicted_aqi'] = [round(p, 2) for p in predictions]
    forecast_df['prediction_timestamp'] = pd.to_datetime(times).dt.strftime('%Y-%m-%d %H:%M:%S')

    fg_forecast = fs.get_or_create_feature_group(
        name="karachi_aqi_forecast", version=1, 
        primary_key=['year', 'month', 'day', 'hour'], online_enabled=True
    )
    
    print("\n📤 Uploading forecast to Hopsworks...")
    fg_forecast.insert(forecast_df, write_options={"wait_for_job": False})
    
    print("\n✅ Professional Ensemble Pipeline complete.")

if __name__ == "__main__":
    run_pipeline()
