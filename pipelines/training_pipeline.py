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
from sklearn.svm import SVR 
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingRandomSearchCV, train_test_split

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
    project = hopsworks.login(api_key_value=os.getenv('MY_HOPSWORK_KEY'))
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    
    print("📥 Fetching Data...")
    
    # ✅ CRITICAL FIX: Read directly from Feature Group instead of Feature View
    # Feature Views don't work in GitHub Actions, but Feature Groups do!
    try:
        # Try to use Feature View (works in VS Code)
        fv = fs.get_feature_view(name="karachi_aqi_view", version=5)
        X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)
        print("✅ Loaded data using Feature View")
    except Exception as e:
        print(f"⚠️ Feature View failed (normal in GitHub Actions): {str(e)[:100]}")
        print("🔄 Switching to Feature Group direct read...")
        
        # Fallback: Read from Feature Group (works in GitHub Actions)
        fg = fs.get_feature_group(name="karachi_aqi", version=1)
        full_df = fg.read()
        
        # Manual train/test split
        target = "aqi"
        X = full_df.drop(columns=[target])
        y = full_df[[target]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"✅ Loaded {len(full_df)} rows from Feature Group")
    
    X_train, y_train = X_train.dropna(), y_train.loc[X_train.dropna().index]
    X_test, y_test = X_test.dropna(), y_test.loc[X_test.dropna().index]

    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- ULTRA-OPTIMIZED PARAMETERS - RandomForest DOMINANCE ---
    param_grids = {
        "RandomForest": {
            "n_estimators": [800, 1000],
            "max_depth": [20, 25, 30],
            "min_samples_leaf": [1, 2],
            "min_samples_split": [2, 3],
            "max_features": ["sqrt", 0.7, 0.8],
            "bootstrap": [True],
            "max_samples": [0.85, 0.95],
            "min_impurity_decrease": [0.0, 0.0001]
        },
        "XGBoost": {
            "n_estimators": [200, 300],
            "learning_rate": [0.05, 0.07],
            "max_depth": [4, 5],
            "min_child_weight": [5],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "gamma": [0.3],
            "reg_alpha": [0.5],
            "reg_lambda": [2.0]
        }, 
        "SVR": {
            "C": [10.0, 15.0],
            "epsilon": [0.1, 0.15],
            "kernel": ['rbf'],
            "gamma": ['scale']
        } 
    }
    
    base_models = {
        "RandomForest": RandomForestRegressor(
            random_state=42, 
            n_jobs=-1,
            warm_start=False,
            oob_score=False,
            criterion='squared_error'
        ),
        "XGBoost": XGBRegressor(
            random_state=42, 
            n_jobs=-1, 
            tree_method='hist'
        ),
        "SVR": SVR(cache_size=1000)
    }

    print("\n🏆 TOURNAMENT")
    print("=" * 60)
    best_m, best_score, best_name = None, float('inf'), ""

    for name, model in base_models.items():
        print(f"\n🔍 Tuning {name}...")
        
        param_size = 1
        for p in param_grids[name].values(): 
            param_size *= len(p)
        
        if name == "RandomForest":
            n_cands = min(30, param_size)
        else:
            n_cands = min(15, param_size)
        
        search = HalvingRandomSearchCV(
            model, 
            param_grids[name], 
            factor=3, 
            cv=5,
            n_candidates=n_cands,
            min_resources='exhaust',
            scoring='neg_root_mean_squared_error', 
            n_jobs=-1, 
            random_state=42,
            verbose=0
        )
        
        search.fit(X_train_s, y_train.values.ravel())
        
        final_model = search.best_estimator_
        
        train_preds = final_model.predict(X_train_s)
        test_preds = final_model.predict(X_test_s)
        
        train_rmse = root_mean_squared_error(y_train, train_preds)
        test_rmse = root_mean_squared_error(y_test, test_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        test_r2 = r2_score(y_test, test_preds)
        
        cv_rmse = abs(search.best_score_)
        overfit_gap = train_rmse - test_rmse
        
        print(f"    📊 {name:12}")
        print(f"       CV RMSE:    {cv_rmse:.4f}")
        print(f"       Train RMSE: {train_rmse:.4f}")
        print(f"       Test RMSE:  {test_rmse:.4f} ✓" if test_rmse < 1.0 else f"       Test RMSE:  {test_rmse:.4f} ✗")
        print(f"       Test MAE:   {test_mae:.4f}")
        print(f"       Test R²:    {test_r2:.4f}")
        print(f"       Overfit Gap: {abs(overfit_gap):.4f} {'✓ Good' if abs(overfit_gap) < 0.5 else '⚠ Check'}")
        print(f"    🔧 Best Params: {search.best_params_}")

        m_dir = f"model_dir_{name.lower()}"
        if os.path.exists(m_dir): shutil.rmtree(m_dir)
        os.makedirs(m_dir)
        joblib.dump(final_model, f"{m_dir}/karachi_aqi_model.pkl", compress=3)
        joblib.dump(scaler, f"{m_dir}/scaler.pkl")
        
        mr.python.create_model(
            name=f"karachi_aqi_{name.lower()}", 
            metrics={
                "test_rmse": float(test_rmse), 
                "cv_rmse": float(cv_rmse),
                "train_rmse": float(train_rmse),
                "test_mae": float(test_mae),
                "test_r2": float(test_r2),
                "overfit_gap": float(abs(overfit_gap))
            },
            description=f"Optimized {name} | Best Params: {search.best_params_}"
        ).save(m_dir)

        if test_rmse < best_score:
            best_score, best_m, best_name = test_rmse, final_model, name

    print("\n" + "=" * 60)
    print(f"🏆 CHAMPION: {best_name} (Test RMSE: {best_score:.4f})")
    if best_score < 1.0:
        print("✅ ELITE PERFORMANCE")
    else:
        print(f"⚠️ Close! Gap to target: {best_score - 1.0:.4f}")
    print("=" * 60)
    
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
    
    print("\n🚀 Uploading 72-hour forecast...")
    for attempt in range(3):
        try:
            fg.insert(forecast_df, write_options={"wait_for_job": False})
            print(f"✅ SUCCESS! {best_name} forecast uploaded to Hopsworks.")
            break
        except Exception as e:
            if attempt < 2:
                print(f"⚠️ Retry {attempt + 1}/3 in 5s...")
                time.sleep(5)
            else:
                raise e

if __name__ == "__main__":
    run_pipeline()
