import os
import requests
import pandas as pd
import hopsworks
import joblib
import shutil
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR 
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import root_mean_squared_error
from hsml.model_schema import ModelSchema

# --- CONFIG ---
KARACHI_LAT, KARACHI_LON = 24.8607, 67.0011
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

def get_forecast_features(trained_columns):
    print("🌐 Fetching 72-hour Forecast Data...")
    # Using the same Open-Meteo endpoint but ensuring we map columns correctly
    params = {
        "latitude": KARACHI_LAT, "longitude": KARACHI_LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,dew_point_2m,pm2_5,pm10,carbon_monoxide",
        "forecast_days": 3
    }
    res = requests.get(FORECAST_URL, params=params).json()
    df_forecast = pd.DataFrame(res["hourly"])
    df_forecast['time'] = pd.to_datetime(df_forecast['time'])
    
    prep = pd.DataFrame({
        'year': df_forecast['time'].dt.year.astype('int64'), 
        'month': df_forecast['time'].dt.month.astype('int64'),
        'day': df_forecast['time'].dt.day.astype('int64'), 
        'hour': df_forecast['time'].dt.hour.astype('int64'),
        'weekday': df_forecast['time'].dt.weekday.astype('float64')
    })

    # Map Open-Meteo names to your Feature Store names
    name_map = {
        'pm2_5': 'pm25', 
        'pm10': 'pm10', 
        'carbon_monoxide': 'co', 
        'temperature_2m': 'temperature', 
        'relative_humidity_2m': 'humidity', 
        'wind_speed_10m': 'wind_speed', 
        'dew_point_2m': 'dew_point'
    }
    
    for api, local in name_map.items():
        if api in df_forecast.columns: 
            prep[local] = df_forecast[api].astype('float64')
    
    # Fill missing columns (lags/change rates) with 0 or last known values to match training schema
    for col in trained_columns:
        if col not in prep.columns:
            prep[col] = 0.0
            
    return prep[trained_columns], df_forecast['time']

def run_pipeline():
    # 1. Login
    api_key = os.getenv('MY_HOPSWORK_KEY')
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    
    # 2. Get Feature View
    # Note: Using version 2 as per your working script, or version 3 if that's your new one
    feature_view = fs.get_feature_view(name="karachi_aqi_view", version=2)
    
    # 3. FETCH DATA (Using the working logic from script 1)
    print("📥 Pulling training data via train_test_split (Stable Method)...")
    # This method is more stable in GitHub Actions environment than get_batch_data()
    X_train, X_test, y_train, y_test = feature_view.train_test_split(test_size=0.2)
    
    # Clean data
    X_train = X_train.dropna()
    y_train = y_train.loc[X_train.index]
    X_test = X_test.dropna()
    y_test = y_test.loc[X_test.index]

    # Scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. TOURNAMENT
    print("\n🏆 STARTING TOURNAMENT...")
    models = {
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10),
        "SVR": SVR(kernel='rbf', C=10)
    }

    best_model, best_rmse, best_model_name = None, float('inf'), ""

    for name, model in models.items():
        model.fit(X_train_scaled, y_train.values.ravel())
        preds = model.predict(X_test_scaled)
        rmse = root_mean_squared_error(y_test, preds)
        
        print(f"📊 {name:12} -> TEST RMSE: {rmse:.4f}")
        
        if rmse < best_rmse:
            best_rmse, best_model, best_model_name = rmse, model, name

    # 5. UPLOAD MODEL TO REGISTRY
    print(f"🌟 Best Model: {best_model_name} with RMSE: {best_rmse:.4f}")
    
    model_dir = "aqi_model_dir"
    if os.path.exists(model_dir): shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    
    joblib.dump(best_model, f"{model_dir}/karachi_aqi_model.pkl")
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")

    input_schema = ModelSchema(X_train)
    output_schema = ModelSchema(y_train)

    mr = project.get_model_registry()
    karachi_model = mr.python.create_model(
        name="karachi_aqi_model", 
        metrics={"rmse": best_rmse}, 
        input_schema=input_schema,
        output_schema=output_schema,
        description=f"Winner: {best_model_name}"
    )
    karachi_model.save(model_dir)
    print("✅ Model Registry Sync Successful!")

    # 6. FORECAST & UPLOAD TO FEATURE GROUP
    X_f, times = get_forecast_features(X_train.columns.tolist())
    
    # Scale forecast features before prediction
    X_f_scaled = scaler.transform(X_f)
    preds = best_model.predict(X_f_scaled)
    
    forecast_df = X_f[['year', 'month', 'day', 'hour']].copy()
    forecast_df['predicted_aqi'] = preds.round(2).astype('float64')
    forecast_df['prediction_timestamp'] = times.dt.strftime('%Y-%m-%d %H:%M:%S')

    print("🚀 Inserting forecast into Hopsworks...")
    forecast_fg = fs.get_feature_group(name="karachi_aqi_forecast", version=1)
    forecast_fg.insert(forecast_df, write_options={"wait_for_job": False})
    
    print(f"✅ SUCCESS! Karachi forecast updated using {best_model_name}.")

if __name__ == "__main__":
    run_pipeline()
