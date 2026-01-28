import os
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
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingRandomSearchCV, train_test_split

# --- CONFIG ---
KARACHI_LAT, KARACHI_LON = 24.8607, 67.0011
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

def get_forecast_features(trained_columns):
    print("🌐 Fetching 72-hour Forecast Data...")
    params = {
        "latitude": KARACHI_LAT, "longitude": KARACHI_LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,dew_point_2m,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "forecast_days": 3
    }
    res = requests.get(FORECAST_URL, params=params).json()
    df_f = pd.DataFrame(res["hourly"])
    df_f['time'] = pd.to_datetime(df_f['time'])
    
    prep = pd.DataFrame({
        'year': df_f['time'].dt.year.astype('int64'), 
        'month': df_f['time'].dt.month.astype('int64'),
        'day': df_f['time'].dt.day.astype('int64'), 
        'hour': df_f['time'].dt.hour.astype('int64')
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
    # --- LOGIN ---
    api_key = os.getenv('MY_HOPSWORK_KEY')
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    
    # --- 1. DATA BYPASS (STRICTLY MAINTAINED FOR GITHUB ACTIONS) ---
    print("📥 Bypassing Feature Store Services... Downloading raw parquet files.")
    dataset_api = project.get_dataset_api()
    
    # Note: We try to find the folder dynamically. 
    # Usually it is Resources/FeatureStore/{fs_name}/{fg_name}_{version}
    # Based on your previous error, let's try to locate karachi_aqi_4 or karachi_aqi_5
    remote_path = f"Resources/FeatureStore/{fs.name}/karachi_aqi_4" 
    local_dir = "./raw_data"
    
    if os.path.exists(local_dir): shutil.rmtree(local_dir)
    os.makedirs(local_dir)

    print(f"📂 Attempting download from: {remote_path}")
    try:
        dataset_api.download(remote_path, local_path=local_dir, overwrite=True)
    except Exception as e:
        print(f"⚠️ Failed to find path {remote_path}. Trying fallback path 'karachi_aqi_5'...")
        remote_path = f"Resources/FeatureStore/{fs.name}/karachi_aqi_5"
        dataset_api.download(remote_path, local_path=local_dir, overwrite=True)

    parquet_files = []
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            if file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))
    
    if not parquet_files:
        raise Exception("No Parquet files found. Pipeline cannot continue.")
        
    print(f"📄 Found {len(parquet_files)} data files. Loading...")
    data_list = [pd.read_parquet(f) for f in parquet_files]
    data = pd.concat(data_list, ignore_index=True)
    
    # Clean system columns
    system_cols = ['_hoodie_commit_time', '_hoodie_commit_seqno', '_hoodie_record_key', '_hoodie_partition_path', '_hoodie_file_name']
    data = data.drop(columns=[c for c in system_cols if c in data.columns]).dropna()
    print(f"✅ Data loaded: {len(data)} rows.")

    # --- 2. PREP ---
    y = data[['aqi']]
    X = data.drop(columns=['aqi'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- 3. TOURNAMENT TUNING (FROM CODE 1) ---
    param_grids = {
        "RandomForest": {
            "n_estimators": [300, 500], 
            "max_features": [1.0], 
            "max_depth": [30, None],
            "bootstrap": [True]
        },
        "XGBoost": {"n_estimators": [100], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}, 
        "SVR": {"C": [1.0, 10.0], "epsilon": [0.1]} 
    }
    
    base_models = {
        "RandomForest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(random_state=42, n_jobs=-1),
        "SVR": SVR(kernel='rbf')
    }

    print("\n🏆 STARTING TOURNAMENT...")
    best_m, best_score, best_name = None, float('inf'), ""

    for name, model in base_models.items():
        print(f"🔍 Tuning {name}...")
        n_cands = 10 if name == "RandomForest" else 4
        
        search = HalvingRandomSearchCV(
            model, 
            param_grids[name], 
            factor=3, 
            cv=3, 
            n_candidates=n_cands,
            min_resources='exhaust',
            scoring='neg_root_mean_squared_error', 
            n_jobs=-1, 
            random_state=42
        )
        search.fit(X_train_s, y_train.values.ravel())
        
        final_model = search.best_estimator_
        test_rmse = root_mean_squared_error(y_test, final_model.predict(X_test_s))
        cv_report_rmse = abs(search.best_score_)
        
        print(f"    📊 {name:12} -> CV RMSE: {cv_report_rmse:.2f} | FINAL TEST RMSE: {test_rmse:.4f}")

        # Save and Register
        m_dir = f"model_dir_{name.lower()}"
        if os.path.exists(m_dir): shutil.rmtree(m_dir)
        os.makedirs(m_dir)
        joblib.dump(final_model, f"{m_dir}/karachi_aqi_model.pkl")
        joblib.dump(scaler, f"{m_dir}/scaler.pkl")
        
        mr.python.create_model(
            name=f"karachi_aqi_{name.lower()}", 
            metrics={"test_rmse": float(test_rmse), "cv_rmse": float(cv_report_rmse)},
            description=f"GitHub Actions Tournament: {name}"
        ).save(m_dir)

        if test_rmse < best_score:
            best_score, best_m, best_name = test_rmse, final_model, name

    print("-" * 50 + f"\n⭐ OVERALL WINNER: {best_name} (Test RMSE: {best_score:.4f})\n" + "-" * 50)
    
    # --- 4. FORECAST & UPLOAD ---
    X_f, times = get_forecast_features(X_train.columns.tolist())
    preds = best_m.predict(scaler.transform(X_f))
    
    forecast_df = X_f[['year', 'month', 'day', 'hour']].copy()
    forecast_df['predicted_aqi'] = preds.round(2).astype('float64')
    forecast_df['prediction_timestamp'] = times.dt.strftime('%Y-%m-%d %H:%M:%S')

    # Ensure bigint compatibility
    for col in ['year', 'month', 'day', 'hour']:
        forecast_df[col] = forecast_df[col].astype('int64')

    fg = fs.get_or_create_feature_group(
        name="karachi_aqi_forecast", version=1, 
        primary_key=['year', 'month', 'day', 'hour'], online_enabled=True
    )
    
    print("🚀 Preparing Forecast Upload...")
    fg.insert(forecast_df, write_options={"wait_for_job": False})
    print(f"✅ SUCCESS! {best_name} forecast uploaded.")

if __name__ == "__main__":
    run_pipeline()
