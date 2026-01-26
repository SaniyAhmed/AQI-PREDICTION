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
from sklearn.model_selection import train_test_split

# --- CONFIG ---
KARACHI_LAT, KARACHI_LON = 24.8607, 67.0011
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

def get_forecast_features(trained_columns):
    """Fetch 72-hour forecast from Open-Meteo API"""
    print("🌐 Fetching 72-hour forecast...")
    params = {
        "latitude": KARACHI_LAT, 
        "longitude": KARACHI_LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,dew_point_2m,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "forecast_days": 3
    }
    res = requests.get(FORECAST_URL, params=params, timeout=30).json()
    df = pd.DataFrame(res["hourly"])
    df['time'] = pd.to_datetime(df['time'])
    
    # Build feature dataframe
    prep = pd.DataFrame({
        'year': df['time'].dt.year.astype('int64'), 
        'month': df['time'].dt.month.astype('int64'),
        'day': df['time'].dt.day.astype('int64'), 
        'hour': df['time'].dt.hour.astype('int64'),
        'weekday': df['time'].dt.weekday.astype('float64')
    })
    
    # Map API names to feature names
    name_map = {
        'pm2_5': 'pm25', 'pm10': 'pm10', 'carbon_monoxide': 'co',
        'nitrogen_dioxide': 'no2', 'sulphur_dioxide': 'so2', 'ozone': 'o3',
        'temperature_2m': 'temperature', 'relative_humidity_2m': 'humidity',
        'wind_speed_10m': 'wind_speed', 'dew_point_2m': 'dew_point'
    }
    
    for api_name, local_name in name_map.items():
        if api_name in df.columns:
            prep[local_name] = df[api_name].astype('float64')
    
    # Add missing columns with zeros
    for col in trained_columns:
        if col not in prep.columns:
            prep[col] = 0.0
    
    return prep[trained_columns], df['time']

def fetch_training_data_from_api():
    """Fetch historical data directly from Open-Meteo API (no Hopsworks read needed)"""
    print("🌐 Fetching historical training data from Open-Meteo API...")
    
    all_data = []
    
    # Fetch last 90 days of data
    for days_back in range(0, 90, 7):
        try:
            params = {
                "latitude": KARACHI_LAT,
                "longitude": KARACHI_LON,
                "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,dew_point_2m,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
                "past_days": days_back,
                "forecast_days": 0
            }
            
            res = requests.get(FORECAST_URL, params=params, timeout=30).json()
            df = pd.DataFrame(res["hourly"])
            all_data.append(df)
            print(f"   ✅ Fetched data from {days_back} days ago")
            time.sleep(1)  # Be nice to API
            
        except Exception as e:
            print(f"   ⚠️ Failed to fetch {days_back} days: {e}")
            continue
    
    if not all_data:
        raise Exception("Could not fetch any training data!")
    
    # Combine all data
    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=['time'])
    full_df['time'] = pd.to_datetime(full_df['time'])
    
    print(f"📊 Total data points: {len(full_df)}")
    
    # Build features
    features = pd.DataFrame({
        'year': full_df['time'].dt.year.astype('int64'),
        'month': full_df['time'].dt.month.astype('int64'),
        'day': full_df['time'].dt.day.astype('int64'),
        'hour': full_df['time'].dt.hour.astype('int64'),
        'weekday': full_df['time'].dt.weekday.astype('float64')
    })
    
    # Add all available features
    name_map = {
        'pm2_5': 'pm25', 'pm10': 'pm10', 'carbon_monoxide': 'co',
        'nitrogen_dioxide': 'no2', 'sulphur_dioxide': 'so2', 'ozone': 'o3',
        'temperature_2m': 'temperature', 'relative_humidity_2m': 'humidity',
        'wind_speed_10m': 'wind_speed', 'dew_point_2m': 'dew_point'
    }
    
    for api_name, local_name in name_map.items():
        if api_name in full_df.columns:
            features[local_name] = full_df[api_name].astype('float64')
    
    # Calculate AQI from PM2.5
    def calc_aqi(pm25):
        if pd.isna(pm25) or pm25 < 0:
            return np.nan
        if pm25 <= 12.0:
            return (50.0/12.0) * pm25
        elif pm25 <= 35.4:
            return ((100-51)/(35.4-12.1)) * (pm25-12.1) + 51
        elif pm25 <= 55.4:
            return ((150-101)/(55.4-35.5)) * (pm25-35.5) + 101
        elif pm25 <= 150.4:
            return ((200-151)/(150.4-55.5)) * (pm25-55.5) + 151
        else:
            return 200 + (pm25-150.4)
    
    features['aqi'] = features['pm25'].apply(calc_aqi)
    
    # Add lag features
    features['aqi_lag_1'] = features['aqi'].shift(1)
    features['aqi_lag_2'] = features['aqi'].shift(2)
    features['pm25_lag_1'] = features['pm25'].shift(1)
    features['aqi_change_rate'] = features['aqi'].diff()
    
    # Drop NaN rows
    features = features.dropna()
    
    print(f"✅ Prepared {len(features)} training samples")
    return features

def run_pipeline():
    """Main training pipeline"""
    print("=" * 70)
    print("🚀 KARACHI AQI TRAINING PIPELINE - GITHUB ACTIONS MODE")
    print("=" * 70)
    
    # Login to Hopsworks
    print("\n🔑 Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=os.getenv('MY_HOPSWORK_KEY'))
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    
    # Get training data directly from API (bypass Hopsworks read issues)
    print("\n📥 FETCHING TRAINING DATA...")
    try:
        full_df = fetch_training_data_from_api()
    except Exception as e:
        print(f"❌ Could not fetch training data: {e}")
        print("🛑 Cannot proceed without training data!")
        exit(1)
    
    # Separate features and target
    target = 'aqi'
    X = full_df.drop(columns=[target])
    y = full_df[[target]]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Training samples: {len(X_train)}")
    print(f"📊 Testing samples: {len(X_test)}")
    print(f"📋 Features: {X_train.columns.tolist()}")
    
    # Scale features
    print("\n🔧 Scaling features...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("\n🏆 TRAINING MODELS...")
    print("-" * 70)
    
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        "SVR": SVR(kernel='rbf', C=10, epsilon=0.1)
    }
    
    best_model = None
    best_rmse = float('inf')
    best_name = ""
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        try:
            model.fit(X_train_scaled, y_train.values.ravel())
            preds = model.predict(X_test_scaled)
            rmse = root_mean_squared_error(y_test, preds)
            
            print(f"   ✅ {name:15} Test RMSE: {rmse:.4f}")
            
            # Save model
            model_dir = f"model_{name.lower()}"
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            os.makedirs(model_dir)
            
            joblib.dump(model, f"{model_dir}/karachi_aqi_model.pkl")
            joblib.dump(scaler, f"{model_dir}/scaler.pkl")
            
            # Register to Hopsworks
            try:
                hops_model = mr.python.create_model(
                    name="karachi_aqi_model",
                    metrics={"test_rmse": float(rmse)},
                    description=f"Model: {name}"
                )
                hops_model.save(model_dir)
                print(f"   📦 Registered to Hopsworks Model Registry")
            except Exception as e:
                print(f"   ⚠️ Could not register: {e}")
            
            # Track best
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_name = name
                
        except Exception as e:
            print(f"   ❌ {name} failed: {e}")
            continue
    
    print("\n" + "-" * 70)
    print(f"🏆 WINNER: {best_name} (RMSE: {best_rmse:.4f})")
    print("-" * 70)
    
    if best_model is None:
        print("❌ No model was successfully trained!")
        exit(1)
    
    # Generate forecast
    print("\n🔮 GENERATING 72-HOUR FORECAST...")
    X_forecast, timestamps = get_forecast_features(X_train.columns.tolist())
    X_forecast_scaled = scaler.transform(X_forecast)
    predictions = best_model.predict(X_forecast_scaled)
    
    # Build forecast dataframe
    forecast_df = pd.DataFrame({
        'year': X_forecast['year'].astype('int64'),
        'month': X_forecast['month'].astype('int64'),
        'day': X_forecast['day'].astype('int64'),
        'hour': X_forecast['hour'].astype('int64'),
        'predicted_aqi': predictions.round(2).astype('float64'),
        'prediction_timestamp': timestamps.dt.strftime('%Y-%m-%d %H:%M:%S')
    })
    
    print(f"✅ Generated {len(forecast_df)} hourly predictions")
    print(f"📅 From: {timestamps.min()}")
    print(f"📅 To: {timestamps.max()}")
    
    # Upload to Hopsworks
    print("\n🚀 UPLOADING FORECAST TO HOPSWORKS...")
    for attempt in range(3):
        try:
            fg_forecast = fs.get_or_create_feature_group(
                name="karachi_aqi_forecast",
                version=1,
                primary_key=['year', 'month', 'day', 'hour'],
                description="72-hour AQI predictions",
                online_enabled=False
            )
            
            fg_forecast.insert(
                forecast_df,
                write_options={
                    "start_offline_materialization": False,
                    "wait_for_job": False
                }
            )
            
            print(f"✅ FORECAST UPLOADED SUCCESSFULLY!")
            break
            
        except Exception as e:
            print(f"⚠️ Upload attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(10)
            else:
                print("❌ All upload attempts failed!")
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"🏆 Best Model: {best_name}")
    print(f"📊 RMSE: {best_rmse:.4f}")
    print(f"🔮 Forecast: {len(forecast_df)} hours uploaded")
    print("=" * 70)

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
