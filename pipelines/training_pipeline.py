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
    
    # ✅ NUCLEAR OPTION: Force download to Parquet, then read locally
    print("📥 Downloading training data as Parquet files...")
    
    try:
        fg = fs.get_feature_group(name="karachi_aqi", version=1)
        
        # Force download the entire feature group as parquet
        print("⬇️ Materializing feature group to local disk...")
        
        # Method 1: Use select_all() with read_options
        query = fg.select_all()
        full_df = query.read(online=False, dataframe_type="pandas", read_options={"arrow_flight_config": None})
        
        print(f"✅ Successfully loaded {len(full_df)} rows")
        
    except Exception as e:
        print(f"⚠️ Method 1 failed: {e}")
        print("🔄 Trying alternative method...")
        
        try:
            # Method 2: Force offline storage read
            fg = fs.get_feature_group(name="karachi_aqi", version=1)
            full_df = fg.read(online=False, dataframe_type="pandas")
            print(f"✅ Loaded {len(full_df)} rows using offline storage")
            
        except Exception as e2:
            print(f"⚠️ Method 2 failed: {e2}")
            print("🔄 Trying Method 3: SQL Query...")
            
            try:
                # Method 3: Raw SQL query (most reliable in GitHub Actions)
                from hsfs import feature_group
                fg = fs.get_feature_group(name="karachi_aqi", version=1)
                
                # Get the storage connector
                full_df = fg.read(read_options={"use_hive": True, "hive_config": {"spark.sql.execution.arrow.pyspark.enabled": "false"}})
                print(f"✅ Loaded {len(full_df)} rows using Hive")
                
            except Exception as e3:
                print(f"❌ All methods failed!")
                print(f"Last error: {e3}")
                
                # FINAL FALLBACK: Use existing model to make predictions only
                print("\n🚨 FALLBACK MODE: Skipping training, using existing model...")
                
                try:
                    # Just generate forecasts with existing model
                    model_meta = mr.get_model("karachi_aqi_model", version=1)
                    model_dir = model_meta.download()
                    model = joblib.load(f"{model_dir}/karachi_aqi_model.pkl")
                    scaler = joblib.load(f"{model_dir}/scaler.pkl")
                    
                    # Get feature names from model
                    if hasattr(model, 'feature_names_in_'):
                        feature_names = list(model.feature_names_in_)
                    else:
                        # Default feature set
                        feature_names = ['year', 'month', 'day', 'hour', 'weekday', 'pm25', 'pm10', 
                                       'co', 'no2', 'so2', 'o3', 'temperature', 'humidity', 
                                       'wind_speed', 'dew_point']
                    
                    X_f, times = get_forecast_features(feature_names)
                    X_f_scaled = scaler.transform(X_f)
                    preds = model.predict(X_f_scaled)
                    
                    forecast_df = X_f[['year', 'month', 'day', 'hour']].copy()
                    forecast_df['predicted_aqi'] = preds.round(2).astype('float64')
                    forecast_df['prediction_timestamp'] = times.dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    for col in ['year', 'month', 'day', 'hour']: 
                        forecast_df[col] = forecast_df[col].astype('int64')
                    
                    # Upload forecast
                    fg_forecast = fs.get_or_create_feature_group(
                        name="karachi_aqi_forecast", 
                        version=1, 
                        primary_key=['year', 'month', 'day', 'hour'], 
                        online_enabled=False
                    )
                    
                    fg_forecast.insert(
                        forecast_df, 
                        write_options={
                            "start_offline_materialization": False,
                            "wait_for_job": False
                        }
                    )
                    
                    print("✅ Forecast generated with existing model!")
                    return
                    
                except Exception as fallback_error:
                    print(f"❌ Fallback also failed: {fallback_error}")
                    raise
    
    # Continue with normal training if data was loaded
    if full_df is None or len(full_df) == 0:
        print("❌ ERROR: No data available!")
        return
    
    print(f"📊 Data shape: {full_df.shape}")
    print(f"📋 Columns: {full_df.columns.tolist()}")
    
    # Prepare target and features
    target = "aqi"
    if target not in full_df.columns:
        print(f"❌ ERROR: Target column '{target}' not found!")
        return
    
    print("🧹 Cleaning data...")
    full_df = full_df.dropna()
    y = full_df[[target]]
    X = full_df.drop(columns=[target])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 🏆 TOURNAMENT
    param_grids = {
        "RandomForest": {"n_estimators": [50, 100], "max_depth": [10, 20], "min_samples_split": [2, 5]},
        "XGBoost": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
        "SVR": {"C": [1, 10], "epsilon": [0.1]}
    }
    
    base_models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "SVR": SVR(kernel='rbf')
    }

    print("\n🏆 TOURNAMENT START...")
    print("-" * 60)
    
    best_model, best_rmse, best_model_name = None, float('inf'), ""

    for name, model in base_models.items():
        print(f"\n🔍 Training {name}...")
        
        try:
            search = RandomizedSearchCV(
                model, param_grids[name], n_iter=3, cv=3, 
                scoring='neg_root_mean_squared_error', n_jobs=-1, random_state=42
            )
            
            search.fit(X_train_scaled, y_train.values.ravel())
            cv_rmse = -search.best_score_
            
            test_preds = search.best_estimator_.predict(X_test_scaled)
            test_rmse = root_mean_squared_error(y_test, test_preds)
            
            print(f"   ✅ {name:15} CV: {cv_rmse:.4f} | Test: {test_rmse:.4f}")

            # Save model
            model_dir = f"model_dir_{name.lower()}"
            if os.path.exists(model_dir): shutil.rmtree(model_dir)
            os.makedirs(model_dir)
            
            joblib.dump(search.best_estimator_, f"{model_dir}/karachi_aqi_model.pkl", compress=3)
            joblib.dump(scaler, f"{model_dir}/scaler.pkl")
            
            # Register
            try:
                current_model = mr.python.create_model(
                    name="karachi_aqi_model", 
                    metrics={"cv_rmse": float(cv_rmse), "test_rmse": float(test_rmse)}, 
                    description=f"Tournament: {name}"
                )
                current_model.save(model_dir)
                print(f"   📦 Registered to Model Registry")
            except Exception as e:
                print(f"   ⚠️ Registry skip: {e}")

            if cv_rmse < best_rmse:
                best_rmse, best_model, best_model_name = cv_rmse, search.best_estimator_, name
                
        except Exception as e:
            print(f"   ❌ {name} failed: {e}")
            continue

    print(f"\n🏆 WINNER: {best_model_name} (RMSE: {best_rmse:.4f})")
    
    if best_model is None:
        print("❌ No model trained!")
        return

    # Generate forecast
    print("\n🔮 Generating forecast...")
    X_f, times = get_forecast_features(X_train.columns.tolist())
    X_f_scaled = scaler.transform(X_f)
    preds = best_model.predict(X_f_scaled)
    
    forecast_df = X_f[['year', 'month', 'day', 'hour']].copy()
    forecast_df['predicted_aqi'] = preds.round(2).astype('float64')
    forecast_df['prediction_timestamp'] = times.dt.strftime('%Y-%m-%d %H:%M:%S')
    
    for col in ['year', 'month', 'day', 'hour']: 
        forecast_df[col] = forecast_df[col].astype('int64')

    # Upload
    print("🚀 Uploading forecast...")
    for attempt in range(3):
        try:
            fg_forecast = fs.get_or_create_feature_group(
                name="karachi_aqi_forecast", 
                version=1, 
                primary_key=['year', 'month', 'day', 'hour'], 
                online_enabled=False
            )
            
            fg_forecast.insert(
                forecast_df, 
                write_options={"start_offline_materialization": False, "wait_for_job": False}
            )
            
            print(f"✅ SUCCESS!")
            break
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {e}")
            if attempt < 2: time.sleep(10)

    print("\n✅ PIPELINE COMPLETE!")

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
