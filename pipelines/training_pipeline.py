import os
import requests
import pandas as pd
import hopsworks
import joblib
import shutil
import time
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR 
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingRandomSearchCV

# Force environment variables for CI/CD stability
os.environ["HSFS_DISABLE_FLIGHT_CLIENT"] = "True"
os.environ["HOPSWORKS_DISABLE_ARROW"] = "True"

# --- CONFIG ---
KARACHI_LAT, KARACHI_LON = 24.8607, 67.0011

def get_forecast_features(trained_columns):
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
        'weekday': df_f['time'].dt.dayofweek.astype('int64')
    })
    
    name_map = {
        'pm2_5':'pm25','pm10':'pm10','carbon_monoxide':'co',
        'nitrogen_dioxide':'no2','sulphur_dioxide':'so2','ozone':'o3',
        'temperature_2m':'temperature', 'relative_humidity_2m':'humidity',
        'wind_speed_10m':'wind_speed','dew_point_2m':'dew_point'
    }
    
    for api, loc in name_map.items(): 
        if api in df_f.columns: prep[loc] = df_f[api].astype('float64')
    
    for c in trained_columns:
        if c not in prep.columns: prep[c] = 0.0
        
    return prep[trained_columns].ffill().bfill(), df_f['time']

def run_pipeline():
    api_key = os.getenv('MY_HOPSWORK_KEY')
    if not api_key:
        raise ValueError("MY_HOPSWORK_KEY environment variable is not set!")

    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    
    print("📥 Fetching Feature View...")
    fv = fs.get_feature_view(name="karachi_aqi_view", version=5)
    
    print("📥 Fetching Training Data (This may take a moment)...")
    # Using read() instead of train_test_split directly can sometimes bypass BinderErrors
    # but here we follow your original logic with added safety
    try:
        X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)
    except Exception as e:
        print(f"Direct split failed: {e}. Attempting full read.")
        df_full = fv.read()
        # Custom split if the API call fails
        df_full = df_full.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(df_full) * 0.8)
        train_df = df_full.iloc[:split_idx]
        test_df = df_full.iloc[split_idx:]
        
        y_train = train_df[['pm25']] # Change target if different
        X_train = train_df.drop(columns=['pm25'])
        y_test = test_df[['pm25']]
        X_test = test_df.drop(columns=['pm25'])

    X_train = X_train.dropna()
    y_train = y_train.loc[X_train.index]
    X_test = X_test.dropna()
    y_test = y_test.loc[X_test.index]

    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    param_grids = {
        "RandomForest": {
            "n_estimators": [500], 
            "max_depth": [20, 25],
            "max_features": ["sqrt", 0.8]
        },
        "XGBoost": {
            "n_estimators": [100],
            "learning_rate": [0.07],
            "max_depth": [5]
        }, 
        "SVR": {
            "C": [10.0],
            "kernel": ['rbf']
        } 
    }
    
    base_models = {
        "RandomForest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(random_state=42, n_jobs=-1, tree_method='hist'),
        "SVR": SVR(cache_size=1000)
    }

    print("\n🏆 TOURNAMENT STARTING")
    best_m, best_score, best_name = None, float('inf'), ""

    for name, model in base_models.items():
        print(f"🔍 Tuning {name}...")
        search = HalvingRandomSearchCV(
            model, param_grids[name], factor=3, cv=3,
            n_candidates='exhaust', scoring='neg_root_mean_squared_error', 
            n_jobs=-1, random_state=42, verbose=0
        )
        search.fit(X_train_s, y_train.values.ravel())
        
        final_model = search.best_estimator_
        test_preds = final_model.predict(X_test_s)
        test_rmse = root_mean_squared_error(y_test, test_preds)
        
        print(f" -> {name} RMSE: {test_rmse:.4f}")

        m_dir = f"model_dir_{name.lower()}"
        if os.path.exists(m_dir): shutil.rmtree(m_dir)
        os.makedirs(m_dir)
        joblib.dump(final_model, f"{m_dir}/karachi_aqi_model.pkl")
        joblib.dump(scaler, f"{m_dir}/scaler.pkl")
        
        mr.python.create_model(
            name=f"karachi_aqi_{name.lower()}", 
            metrics={"test_rmse": float(test_rmse)},
            description=f"Automated Training"
        ).save(m_dir)

        if test_rmse < best_score:
            best_score, best_m, best_name = test_rmse, final_model, name

    print(f"\n🏆 CHAMPION: {best_name}")
    
    # Forecast
    X_f, times = get_forecast_features(X_train.columns.tolist())
    preds = best_m.predict(scaler.transform(X_f))
    
    forecast_df = X_f[['year', 'month', 'day', 'hour']].copy()
    forecast_df['predicted_aqi'] = preds.round(2).astype('float64')
    forecast_df['prediction_timestamp'] = times.dt.strftime('%Y-%m-%d %H:%M:%S')

    for col in ['year', 'month', 'day', 'hour']:
        forecast_df[col] = forecast_df[col].astype('int64')

    fg = fs.get_or_create_feature_group(
        name="karachi_aqi_forecast", version=1, 
        primary_key=['year', 'month', 'day', 'hour'], online_enabled=True
    )
    
    print("🚀 Uploading forecast...")
    fg.insert(forecast_df, write_options={"wait_for_job": False})
    print("✅ SUCCESS!")

if __name__ == "__main__":
    run_pipeline()
