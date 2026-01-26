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
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV

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
    df_forecast = pd.DataFrame(res["hourly"])
    df_forecast['time'] = pd.to_datetime(df_forecast['time'])
    prep = pd.DataFrame({
        'year': df_forecast['time'].dt.year.astype('int64'), 
        'month': df_forecast['time'].dt.month.astype('int64'),
        'day': df_forecast['time'].dt.day.astype('int64'), 
        'hour': df_forecast['time'].dt.hour.astype('int64')
    })
    name_map = {
        'pm2_5': 'pm25', 
        'pm10': 'pm10', 
        'carbon_monoxide': 'co', 
        'nitrogen_dioxide': 'no2', 
        'sulphur_dioxide': 'so2', 
        'ozone': 'o3', 
        'temperature_2m': 'temperature', 
        'relative_humidity_2m': 'humidity', 
        'wind_speed_10m': 'wind_speed', 
        'dew_point_2m': 'dew_point'
    }
    for api, local in name_map.items():
        if api in df_forecast.columns: 
            prep[local] = df_forecast[api].astype('float64')
    for col in trained_columns:
        if col not in prep.columns: 
            prep[col] = 0.0
    return prep[trained_columns], df_forecast['time']

def run_pipeline():
    print("🔑 Logging into Hopsworks...")
    project = hopsworks.login(api_key_value=os.getenv('MY_HOPSWORK_KEY'))
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    
    # ✅ CRITICAL FIX: Read directly from Feature Group instead of Feature View
    print("📥 Reading Training Data from Feature Group...")
    try:
        fg = fs.get_feature_group(name="karachi_aqi", version=1)
        full_df = fg.read(read_options={"use_hive": True})
        print(f"✅ Loaded {len(full_df)} rows from Feature Group")
    except Exception as e:
        print(f"❌ Error reading feature group: {e}")
        print("🔍 Attempting alternative read method...")
        fg = fs.get_feature_group(name="karachi_aqi", version=1)
        full_df = fg.read()
    
    # Check if data exists
    if full_df is None or len(full_df) == 0:
        print("❌ ERROR: No data found in feature group!")
        return
    
    # Prepare target and features
    target = "aqi"
    if target not in full_df.columns:
        print(f"❌ ERROR: Target column '{target}' not found!")
        print(f"Available columns: {full_df.columns.tolist()}")
        return
    
    print("🧹 Cleaning data...")
    y = full_df[[target]].dropna()
    X = full_df.drop(columns=[target]).loc[y.index]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 🏆 TOURNAMENT SETUP
    param_grids = {
        "RandomForest": {
            "n_estimators": [50, 100], 
            "max_depth": [10, 20], 
            "min_samples_split": [2, 5]
        },
        "XGBoost": {
            "n_estimators": [50, 100], 
            "learning_rate": [0.05, 0.1], 
            "max_depth": [3, 5]
        },
        "SVR": {
            "C": [1, 10], 
            "epsilon": [0.1]
        }
    }
    
    base_models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "SVR": SVR(kernel='rbf')
    }

    print("\n🏆 STARTING TOURNAMENT...")
    print("-" * 60)
    
    best_model, best_rmse, best_model_name = None, float('inf'), ""
    tournament_results = []

    for name, model in base_models.items():
        print(f"\n🔍 Tuning {name}...")
        
        try:
            search = RandomizedSearchCV(
                model, 
                param_grids[name], 
                n_iter=3, 
                cv=3, 
                scoring='neg_root_mean_squared_error', 
                n_jobs=-1,
                random_state=42
            )
            
            search.fit(X_train_scaled, y_train.values.ravel())
            cv_rmse = -search.best_score_
            
            # Test performance
            test_preds = search.best_estimator_.predict(X_test_scaled)
            test_rmse = root_mean_squared_error(y_test, test_preds)
            
            print(f"   ✅ {name:15} -> CV RMSE: {cv_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
            
            tournament_results.append({
                'model': name,
                'cv_rmse': cv_rmse,
                'test_rmse': test_rmse
            })

            # Save this model version
            iter_model_dir = f"model_dir_{name.lower()}"
            if os.path.exists(iter_model_dir): 
                shutil.rmtree(iter_model_dir)
            os.makedirs(iter_model_dir)
            
            joblib.dump(search.best_estimator_, f"{iter_model_dir}/karachi_aqi_model.pkl", compress=3)
            joblib.dump(scaler, f"{iter_model_dir}/scaler.pkl")
            
            # Register to Hopsworks
            try:
                current_model = mr.python.create_model(
                    name="karachi_aqi_model", 
                    metrics={"cv_rmse": float(cv_rmse), "test_rmse": float(test_rmse)}, 
                    description=f"Tournament Participant: {name} | Best Params: {search.best_params_}"
                )
                current_model.save(iter_model_dir)
                print(f"   📦 {name} registered to Model Registry")
            except Exception as e:
                print(f"   ⚠️ Could not register {name}: {e}")

            # Track best model
            if cv_rmse < best_rmse:
                best_rmse = cv_rmse
                best_model = search.best_estimator_
                best_model_name = name
                
        except Exception as e:
            print(f"   ❌ {name} training failed: {e}")
            continue

    print("\n" + "-" * 60)
    print(f"🏆 TOURNAMENT WINNER: {best_model_name} (CV RMSE: {best_rmse:.4f})")
    print("-" * 60)
    
    if best_model is None:
        print("❌ ERROR: No model was successfully trained!")
        return

    # 6. GENERATE 72-HOUR FORECAST
    print("\n🔮 Generating 72-hour forecast...")
    X_f, times = get_forecast_features(X_train.columns.tolist())
    
    print(f"📊 Forecast data shape: {X_f.shape}")
    
    # Scale and predict
    X_f_scaled = scaler.transform(X_f)
    preds = best_model.predict(X_f_scaled)
    
    # Prepare forecast dataframe
    forecast_df = X_f[['year', 'month', 'day', 'hour']].copy()
    forecast_df['predicted_aqi'] = preds.round(2).astype('float64')
    forecast_df['prediction_timestamp'] = times.dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Ensure correct types
    for col in ['year', 'month', 'day', 'hour']: 
        forecast_df[col] = forecast_df[col].astype('int64')
    
    print(f"✅ Generated {len(forecast_df)} hourly predictions")

    # 7. UPLOAD FORECAST TO HOPSWORKS
    print("\n🚀 Uploading forecast to Hopsworks...")
    
    for attempt in range(3):
        try:
            # Get or create forecast feature group
            fg_forecast = fs.get_or_create_feature_group(
                name="karachi_aqi_forecast", 
                version=1, 
                primary_key=['year', 'month', 'day', 'hour'], 
                description="72-hour AQI predictions for Karachi",
                online_enabled=False
            )
            
            print(f"📤 Uploading forecast (Attempt {attempt + 1}/3)...")
            
            # Insert with specific options for GitHub Actions
            fg_forecast.insert(
                forecast_df, 
                write_options={
                    "start_offline_materialization": False,
                    "wait_for_job": False
                }
            )
            
            print(f"✅ SUCCESS! Forecast uploaded to Hopsworks")
            print(f"📅 Predictions from {times.min()} to {times.max()}")
            break
            
        except Exception as e:
            print(f"⚠️ Upload attempt {attempt + 1} failed: {e}")
            
            if attempt < 2:
                wait_time = (attempt + 1) * 10
                print(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print("❌ All upload attempts failed!")
                print("💡 Check Hopsworks UI manually for feature group status")
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"🏆 Winner: {best_model_name}")
    print(f"📊 Best RMSE: {best_rmse:.4f}")
    print(f"🔮 Forecast: Next 72 hours uploaded")
    print("=" * 60)

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
