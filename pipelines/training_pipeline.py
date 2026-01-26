import os
# Force disable Flight to prevent the 'Could not read data using Query Service' error
os.environ["HSFS_DISABLE_FLIGHT_CLIENT"] = "True"

import requests
import pandas as pd
import hopsworks
import joblib
import shutil
import time
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR 
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from hsml.model_schema import ModelSchema

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
    name_map = {'pm2_5': 'pm25', 'pm10': 'pm10', 'carbon_monoxide': 'co', 'nitrogen_dioxide': 'no2', 'sulphur_dioxide': 'so2', 'ozone': 'o3', 'temperature_2m': 'temperature', 'relative_humidity_2m': 'humidity', 'wind_speed_10m': 'wind_speed', 'dew_point_2m': 'dew_point'}
    for api, local in name_map.items():
        if api in df_forecast.columns: prep[local] = df_forecast[api].astype('float64')
    for col in trained_columns:
        if col not in prep.columns: prep[col] = 0.0
    return prep[trained_columns], df_forecast['time']

def run_pipeline():
    # 1. Login
    project = hopsworks.login(api_key_value=os.getenv('MY_HOPSWORK_KEY'))
    fs = project.get_feature_store()
    
    # 2. Get Feature View
    feature_view = fs.get_feature_view(name="karachi_aqi_view", version=3)
    
    print("📥 Retrieving Training Data via REST engine...")
    # CRITICAL CHANGE: read_options={"use_api": True} forces standard HTTP REST
    # This bypasses the Arrow Flight engine that is crashing in GitHub
    data_df = feature_view.get_batch_data(read_options={"use_api": True}) 
    
    # 3. Pre-process
    y = data_df[['aqi']]
    X = data_df.drop(columns=['aqi'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train, y_train = X_train.dropna(), y_train.loc[X_train.dropna().index]
    X_test, y_test = X_test.dropna(), y_test.loc[X_test.dropna().index]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 🏆 TOURNAMENT
    print("\n🏆 STARTING TOURNAMENT...")
    param_grids = {
        "RandomForest": {"n_estimators": [50, 100], "max_depth": [10, 20]},
        "XGBoost": {"n_estimators": [50, 100], "learning_rate": [0.1]},
        "SVR": {"C": [1, 10], "epsilon": [0.1]}
    }
    base_models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "SVR": SVR(kernel='rbf')
    }

    best_model, best_rmse, best_model_name = None, float('inf'), ""

    for name, model in base_models.items():
        search = RandomizedSearchCV(model, param_grids[name], n_iter=2, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
        search.fit(X_train_scaled, y_train.values.ravel())
        cv_rmse = -search.best_score_
        test_preds = search.best_estimator_.predict(X_test_scaled)
        test_rmse = root_mean_squared_error(y_test, test_preds)
        
        print(f"   📊 {name:12} -> CV RMSE: {cv_rmse:.4f} | TEST RMSE: {test_rmse:.4f}")
        
        if cv_rmse < best_rmse:
            best_rmse, best_model, best_model_name = cv_rmse, search.best_estimator_, name

    # 4. Schema
    input_schema = ModelSchema(X_train)
    output_schema = ModelSchema(y_train)

    # 5. UPLOAD
    model_dir = "aqi_model_dir"
    if os.path.exists(model_dir): shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    joblib.dump(best_model, f"{model_dir}/karachi_aqi_model.pkl", compress=3)
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")

    mr = project.get_model_registry()
    karachi_model = mr.python.create_model(
        name="karachi_aqi_model", 
        metrics={"cv_rmse": best_rmse}, 
        input_schema=input_schema,
        output_schema=output_schema,
        description=f"Winner: {best_model_name} (Stable REST Mode)"
    )
    karachi_model.save(model_dir)
    print("✅ Model Registry Sync Successful!")

    # 6. & 7. FORECAST & UPLOAD
    X_f, times = get_forecast_features(X_train.columns.tolist())
    preds = best_model.predict(scaler.transform(X_f))
    forecast_df = X_f[['year', 'month', 'day', 'hour']].copy()
    forecast_df['predicted_aqi'] = preds.round(2).astype('float64')
    forecast_df['prediction_timestamp'] = times.dt.strftime('%Y-%m-%d %H:%M:%S')

    fg = fs.get_or_create_feature_group(name="karachi_aqi_forecast", version=1, primary_key=['year', 'month', 'day', 'hour'], online_enabled=True)
    for col in ['year', 'month', 'day', 'hour']: forecast_df[col] = forecast_df[col].astype('int64')
    fg.insert(forecast_df, write_options={"start_offline_materialization": True, "wait_for_job": False})
    print(f"🚀 SUCCESS! Karachi forecast is live.")

if __name__ == "__main__":
    run_pipeline()
