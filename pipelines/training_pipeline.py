import os
import requests
import pandas as pd
import hopsworks
import joblib
import shutil
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR 
from sklearn.preprocessing import RobustScaler  # Switched to RobustScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

# ==========================================
# --- 1. THE ULTIMATE KILL-SWITCHES ---
# ==========================================
os.environ["HSFS_DISABLE_FLIGHT_CLIENT"] = "True"

# --- CONFIGURATION ---
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AQICN_TOKEN = "c6a73cfca3b6bb6d9930dabdd8c0eea057e29278"
AQICN_URL = f"https://api.waqi.info/feed/geo:{KARACHI_LAT};{KARACHI_LON}/?token={AQICN_TOKEN}"

def get_forecast_features():
    print("🌐 Fetching Weather + Pollutant Forecasts...")
    w_params = {"latitude": KARACHI_LAT, "longitude": KARACHI_LON, "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,dew_point_2m", "forecast_days": 3}
    w_res = requests.get(FORECAST_URL, params=w_params).json()
    df_forecast = pd.DataFrame(w_res["hourly"])
    df_forecast['time'] = pd.to_datetime(df_forecast['time'])

    aq_res = requests.get(AQICN_URL).json()
    pm25_daily_forecast = aq_res['data']['forecast']['daily']['pm25']
    
    def map_daily_pm25(row_time):
        date_str = row_time.strftime('%Y-%m-%d')
        for day_data in pm25_daily_forecast:
            if day_data['day'] == date_str: return float(day_data['avg'])
        return float(pm25_daily_forecast[0]['avg'])

    prep = pd.DataFrame()
    prep['year'] = df_forecast['time'].dt.year.astype('int64')
    prep['month'] = df_forecast['time'].dt.month.astype('int64')
    prep['day'] = df_forecast['time'].dt.day.astype('int64')
    prep['hour'] = df_forecast['time'].dt.hour.astype('int64')
    prep['weekday'] = df_forecast['time'].dt.weekday.astype('float64')
    prep['dew_point'] = df_forecast['dew_point_2m'].astype('float64')
    prep['wind_speed'] = df_forecast['wind_speed_10m'].astype('float64')
    prep['pm25'] = df_forecast['time'].apply(map_daily_pm25).astype('float64')
    return prep, df_forecast['time']

def run_pipeline():
    # 1. LOGIN
    api_key = os.getenv('MY_HOPSWORK_KEY') 
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()

    # 2. AGGRESSIVE DATA FETCH
    print("📥 Attempting to find and read Feature Group...")
    full_df = None
    possible_names = ["karachi_aqi", "karachi_aqi_1", "karachi_aqi_fg"]
    
    for name in possible_names:
        try:
            print(f"🔍 Checking for Feature Group: '{name}' (version 1)...")
            fg = fs.get_feature_group(name=name, version=1)
            full_df = fg.read(read_options={"use_hive": True})
            if full_df is not None:
                print(f"✅ Successfully loaded data from '{name}'!")
                break
        except Exception as e:
            print(f"❌ Could not read '{name}': {e}")
            continue

    if full_df is None:
        print("🛑 ERROR: Could not find ANY feature group.")
        return

    # 3. SPLIT & CLEAN
    target = "aqi"
    if target not in full_df.columns:
        print(f"🛑 ERROR: Target column '{target}' not found!")
        return

    y = full_df[[target]]
    X = full_df.drop(columns=[target])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = X_train.dropna(), X_test.dropna()
    y_train, y_test = y_train.loc[X_train.index], y_test.loc[X_test.index]

    # --- FEATURE SCALING (Robust for Outliers) ---
    scaler = RobustScaler()  # Updated to RobustScaler
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. MODEL TOURNAMENT
    print("🏆 Starting Model Tournament...")
    models = {
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10),
        "SVR": SVR(kernel='rbf', C=10, epsilon=0.1) 
    }

    best_model, best_rmse, best_model_name = None, float('inf'), ""
    for name, model in models.items():
        model.fit(X_train_scaled, y_train.values.ravel())
        preds = model.predict(X_test_scaled)
        rmse = root_mean_squared_error(y_test, preds)
        print(f"   - {name} RMSE: {rmse:.4f}")
        if rmse < best_rmse:
            best_rmse, best_model, best_model_name = rmse, model, name

    # 5. SAVE BEST MODEL
    print(f"📦 Saving {best_model_name} to Registry...")
    mr = project.get_model_registry()
    model_dir = "aqi_model_dir"
    if os.path.exists(model_dir): shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    
    joblib.dump(best_model, f"{model_dir}/karachi_aqi_model.pkl")
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")

    karachi_model = mr.python.create_model(
        name="karachi_aqi_model", 
        metrics={"rmse": best_rmse},
        description=f"Winner with Robust Scaling: {best_model_name}"
    )
    karachi_model.save(model_dir)

    # 6. GENERATE 3-DAY FORECAST
    X_forecast, timestamps = get_forecast_features()
    last_known = X_train.iloc[-1]
    cols_to_persist = ['pm10', 'co', 'aqi_lag_1', 'aqi_lag_2', 'pm25_lag_1']
    for col in cols_to_persist:
        if col in last_known: X_forecast[col] = last_known[col]
        else: X_forecast[col] = 0 
    
    X_forecast['aqi_change_rate'] = 0.0
    X_forecast = X_forecast[X_train.columns]

    X_forecast_scaled = scaler.transform(X_forecast)

    print("🔮 Generating future predictions...")
    preds = best_model.predict(X_forecast_scaled)

    # 7. PREPARE DATAFRAME
    forecast_df = X_forecast[['year', 'month', 'day', 'hour']].copy()
    forecast_df['predicted_aqi'] = preds.round(2).astype('float64')
    forecast_df['prediction_timestamp'] = timestamps.dt.strftime('%Y-%m-%d %H:%M:%S')

    # 8. INSERT INTO HOPSWORKS
    try:
        print("🚀 Inserting predictions into karachi_aqi_forecast...")
        forecast_fg = fs.get_feature_group(name="karachi_aqi_forecast", version=1)
        forecast_fg.insert(forecast_df, write_options={"wait_for_job": False})
        print(f"✅ SUCCESS! Best model was {best_model_name}")
    except Exception as e:
        print(f"❌ Error during insertion: {e}")

if __name__ == "__main__":
    run_pipeline()
