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
    
    # 1. Fetch Pollutants & Weather (Past 24h to calculate lags)
    aq_params = {"latitude": KARACHI_LAT, "longitude": KARACHI_LON, 
                 "hourly": "pm2_5,pm10,carbon_monoxide", "past_days": 1}
    w_params = {"latitude": KARACHI_LAT, "longitude": KARACHI_LON, 
                "hourly": "wind_speed_10m,dew_point_2m", "past_days": 1}
    
    aq_data = requests.get(AQ_URL, params=aq_params).json()["hourly"]
    w_data = requests.get(WEATHER_URL, params=w_params).json()["hourly"]

    # 2. Merge and Calculate
    df = pd.merge(pd.DataFrame(aq_data), pd.DataFrame(w_data), on="time")
    df['time'] = pd.to_datetime(df['time'])
    
    # Define AQI Formula
    def calc_aqi(pm):
        if pm <= 12: return (50/12) * pm
        elif pm <= 35.4: return ((100-51)/(35.4-12.1)) * (pm-12.1) + 51
        else: return 150

    # 3. Construct Final DataFrame matching your 15 columns EXACTLY
    live_df = pd.DataFrame()
    
    # Time Features (Integers / BigInt)
    live_df['year'] = df['time'].dt.year.astype('int64')
    live_df['month'] = df['time'].dt.month.astype('int64')
    live_df['day'] = df['time'].dt.day.astype('int64')
    live_df['hour'] = df['time'].dt.hour.astype('int64')
    
    # Measurement Features (Floats / Double)
    live_df['weekday'] = df['time'].dt.weekday.astype('float64')
    live_df['pm25'] = df['pm2_5'].astype('float64')
    live_df['pm10'] = df['pm10'].astype('float64')
    live_df['co'] = df['carbon_monoxide'].astype('float64')
    live_df['dew_point'] = df['dew_point_2m'].astype('float64')
    live_df['wind_speed'] = df['wind_speed_10m'].astype('float64')
    
    # Derived Features
    live_df['aqi'] = live_df['pm25'].apply(calc_aqi).astype('float64')
    live_df['aqi_change_rate'] = live_df['aqi'].diff().fillna(0.0).astype('float64')
    live_df['aqi_lag_1'] = live_df['aqi'].shift(1).fillna(live_df['aqi']).astype('float64')
    live_df['aqi_lag_2'] = live_df['aqi'].shift(2).fillna(live_df['aqi']).astype('float64')
    live_df['pm25_lag_1'] = live_df['pm25'].shift(1).fillna(live_df['pm25']).astype('float64')

    # Reorder columns to match your Hopsworks list exactly (optional but safer)
    target_cols = ['weekday', 'month', 'dew_point', 'aqi_lag_1', 'year', 'aqi_lag_2', 
                   'hour', 'co', 'aqi_change_rate', 'pm10', 'aqi', 'pm25_lag_1', 
                   'day', 'pm25', 'wind_speed']
    
    return live_df[target_cols].tail(1)

def run_pipeline():
    api_key = os.getenv('MY_HOPSWORK_KEY') 
    
    # --- RETRY LOGIC (3 ATTEMPTS) ---
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"🔄 Execution attempt {attempt + 1} of {max_retries}...")
            
            project = hopsworks.login(api_key_value=api_key)
            fs = project.get_feature_store()
            aqi_fg = fs.get_feature_group(name="karachi_aqi", version=1)
            
            new_row = get_live_data()
            print("🚀 Columns verified. Preparing to insert...")
            
            # Using wait_for_job=False to prevent GitHub Action timeouts
            aqi_fg.insert(new_row, write_options={"wait_for_job": False})
            
            print(f"✅ SUCCESS! Karachi AQI for hour {new_row['hour'].values[0]} is now in Hopsworks.")
            return # Exit function on success
            
        except Exception as e:
            print(f"⚠️ Error occurred: {e}")
            if attempt < max_retries - 1:
                wait_time = 15 # Wait 15 seconds before trying again
                print(f"⏳ Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                print("❌ All retry attempts failed. Raising error.")
                raise e

if __name__ == "__main__":
    run_pipeline()
