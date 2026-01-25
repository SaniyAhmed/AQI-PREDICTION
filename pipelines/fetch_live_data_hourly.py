import os
import requests
import pandas as pd
import hopsworks
import time
from datetime import datetime

# --- CONFIG ---
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

def get_live_data():
    print("📡 Fetching current hourly data for Karachi...")
    
    # 1. Fetch ALL Pollutants (matches your Hybrid mandatory list)
    aq_params = {
        "latitude": KARACHI_LAT, 
        "longitude": KARACHI_LON, 
        "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone", 
        "past_days": 1
    }
    # Fetch Weather
    w_params = {
        "latitude": KARACHI_LAT, 
        "longitude": KARACHI_LON, 
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,dew_point_2m", 
        "past_days": 1
    }
    
    aq_res = requests.get(AQ_URL, params=aq_params).json()
    w_res = requests.get(WEATHER_URL, params=w_params).json()

    # 2. Merge and Calculate
    df = pd.merge(pd.DataFrame(aq_res["hourly"]), pd.DataFrame(w_res["hourly"]), on="time")
    df['time'] = pd.to_datetime(df['time'])
    
    def calc_aqi(pm):
        if pm <= 12: return (50/12) * pm
        elif pm <= 35.4: return ((100-51)/(35.4-12.1)) * (pm-12.1) + 51
        else: return 150

    # 3. Construct Final DataFrame matching Version 3 Schema
    live_df = pd.DataFrame()
    
    # Time Keys
    live_df['year'] = df['time'].dt.year.astype('int64')
    live_df['month'] = df['time'].dt.month.astype('int64')
    live_df['day'] = df['time'].dt.day.astype('int64')
    live_df['hour'] = df['time'].dt.hour.astype('int64')
    
    # Pollutants (Mapping API names to your local names)
    live_df['pm25'] = df['pm2_5'].astype('float64')
    live_df['pm10'] = df['pm10'].astype('float64')
    live_df['co'] = df['carbon_monoxide'].astype('float64')
    live_df['no2'] = df['nitrogen_dioxide'].astype('float64')
    live_df['so2'] = df['sulphur_dioxide'].astype('float64')
    live_df['o3'] = df['ozone'].astype('float64')
    
    # Weather
    live_df['temperature'] = df['temperature_2m'].astype('float64')
    live_df['humidity'] = df['relative_humidity_2m'].astype('float64')
    live_df['wind_speed'] = df['wind_speed_10m'].astype('float64')
    live_df['dew_point'] = df['dew_point_2m'].astype('float64')
    
    # Derived Features (Lags & Target)
    live_df['aqi'] = live_df['pm25'].apply(calc_aqi).astype('float64')
    live_df['aqi_change_rate'] = live_df['aqi'].diff().fillna(0.0).astype('float64')
    live_df['aqi_lag_1'] = live_df['aqi'].shift(1).fillna(live_df['aqi']).astype('float64')
    live_df['aqi_lag_2'] = live_df['aqi'].shift(2).fillna(live_df['aqi']).astype('float64')
    live_df['pm25_lag_1'] = live_df['pm25'].shift(1).fillna(live_df['pm25']).astype('float64')

    return live_df.tail(1)

def run_pipeline():
    api_key = os.getenv('MY_HOPSWORK_KEY') 
    
    try:
        project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store()
        
        # IMPORTANT: Connect to Version 3
        aqi_fg = fs.get_feature_group(name="karachi_aqi", version=3)
        
        # Get the schema expectations from Hopsworks
        expected_features = [f.name for f in aqi_fg.features]
        
        new_row = get_live_data()
        
        # Filter to only send what V3 expects (handles RFE dropped columns automatically)
        final_row = new_row[[col for col in new_row.columns if col in expected_features]]
        
        print(f"🚀 Syncing Hybrid Row to V3. Columns: {final_row.columns.tolist()}")
        aqi_fg.insert(final_row, write_options={"wait_for_job": False})
        
        print(f"✅ SUCCESS! Live data updated for hour {final_row['hour'].values[0]}.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_pipeline()
