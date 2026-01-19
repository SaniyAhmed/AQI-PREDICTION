import hopsworks
import pandas as pd
import requests
import joblib
import os
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error, r2_score 

# --- CONFIGURATION ---
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

def get_aqi_status(aqi):
    if aqi <= 50: return "🟢 Good"
    elif aqi <= 100: return "🟡 Moderate"
    elif aqi <= 150: return "🟠 Unhealthy (Sensitive)"
    elif aqi <= 200: return "🔴 Unhealthy"
    else: return "🟣 Very Unhealthy"

def get_forecast_features():
    print("🌐 Fetching 3-day weather forecast for Karachi...")
    params = {
        "latitude": KARACHI_LAT, "longitude": KARACHI_LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,dew_point_2m",
        "forecast_days": 3
    }
    res = requests.get(FORECAST_URL, params=params).json()
    df_forecast = pd.DataFrame(res["hourly"])
    df_forecast['time'] = pd.to_datetime(df_forecast['time'])
    
    prep = pd.DataFrame()
    prep['year'] = df_forecast['time'].dt.year.astype('int64')
    prep['month'] = df_forecast['time'].dt.month.astype('int64')
    prep['day'] = df_forecast['time'].dt.day.astype('int64')
    prep['hour'] = df_forecast['time'].dt.hour.astype('int64')
    prep['weekday'] = df_forecast['time'].dt.weekday.astype('float64')
    prep['dew_point'] = df_forecast['dew_point_2m'].astype('float64')
    prep['wind_speed'] = df_forecast['wind_speed_10m'].astype('float64')
    
    return prep, df_forecast['time']

def run_pipeline():
    project = hopsworks.login()
    fs = project.get_feature_store()

    print("📥 Pulling data from Feature View Version 2...")
    feature_view = fs.get_feature_view(name="karachi_aqi_view", version=2)
    X_train, X_test, y_train, y_test = feature_view.train_test_split(test_size=0.2)

    # 3. MODEL COMPETITION
    models = {
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10),
        "Ridge": Ridge(alpha=1.0)
    }

    best_model, best_rmse, best_name = None, float('inf'), ""
    print("🚀 Starting Model Training...")
    for name, model in models.items():
        model.fit(X_train, y_train.values.ravel())
        preds = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, preds)
        if rmse < best_rmse:
            best_rmse, best_model, best_name = rmse, model, name

    print(f"\n🏆 Winner: {best_name} (RMSE: {best_rmse:.4f})")
    
    # 4. SAVE BEST MODEL (Local and Remote)
    model_dir = "aqi_model"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, f"{model_dir}/model.pkl")

    try:
        mr = project.get_model_registry()
        aqi_model = mr.python.create_model(
            name="karachi_aqi_model", 
            metrics={"rmse": best_rmse},
            description=f"Retrained winner: {best_name}"
        )
        aqi_model.save(model_dir)
        print("✅ Model Registry updated successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Registry upload failed, but local model is ready.")

    # 5. GENERATE 3-DAY PREDICTIONS
    X_forecast, timestamps = get_forecast_features()
    
    # We use y_train to get the last known aqi value, and X_train for the features
    last_known_features = X_train.iloc[-1]
    
    X_forecast['pm25'] = last_known_features['pm25']
    X_forecast['pm10'] = last_known_features['pm10']
    X_forecast['co'] = last_known_features['co']
    X_forecast['aqi_lag_1'] = last_known_features['aqi_lag_1']
    X_forecast['aqi_lag_2'] = last_known_features['aqi_lag_2']
    X_forecast['pm25_lag_1'] = last_known_features['pm25_lag_1']
    X_forecast['aqi_change_rate'] = 0.0 

    # 6. EXACT SCHEMA ORDER (REMOVED 'aqi' BECAUSE IT IS THE TARGET)
    # The model expects exactly these 14 columns in this order:
    hopsworks_features = [
        'weekday', 'month', 'dew_point', 'aqi_lag_1', 'year', 'aqi_lag_2', 
        'hour', 'co', 'aqi_change_rate', 'pm10', 'pm25_lag_1', 
        'day', 'pm25', 'wind_speed'
    ]
    X_forecast = X_forecast[hopsworks_features]

    print("🔮 Generating future predictions...")
    preds = best_model.predict(X_forecast)
    
    # 7. FORMAT AND SAVE
    final_df = pd.DataFrame({
        "Timestamp": timestamps,
        "Predicted_AQI": preds.round(2)
    })
    final_df['Health_Status'] = final_df['Predicted_AQI'].apply(get_aqi_status)

    print("\n📅 --- KARACHI 3-DAY SUMMARY ---")
    summary = final_df.groupby(final_df['Timestamp'].dt.date).agg({
        'Predicted_AQI': 'mean',
        'Health_Status': lambda x: x.mode()[0]
    })
    print(summary)

    final_df.to_csv("karachi_3day_forecast.csv", index=False)
    print("\n✅ Forecast saved to 'karachi_3day_forecast.csv'.")

if __name__ == "__main__":
    run_pipeline()
