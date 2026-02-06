import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- SETTINGS ---
# Using '.' means 'current folder', making it work anywhere on your D: drive
BASE_PATH = "." 
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011

METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# --- DATE RANGE: 365 DAYS ---
start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

print(f"📡 Fetching Karachi Data (from {start_date} to {end_date})...")

# 1. Fetch Gas/Pollutant Data
aq_params = {
    "latitude": KARACHI_LAT, "longitude": KARACHI_LON,
    "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
    "start_date": start_date, "end_date": end_date, "timezone": "auto"
}
aq_res = requests.get(AQ_URL, params=aq_params).json()
df_aq = pd.DataFrame(aq_res["hourly"])

# 2. Fetch Weather
weather_params = {
    "latitude": KARACHI_LAT, "longitude": KARACHI_LON,
    "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,dew_point_2m",
    "start_date": start_date, "end_date": end_date, "timezone": "auto"
}
weather_res = requests.get(METEO_URL, params=weather_params).json()
df_weather = pd.DataFrame(weather_res["hourly"])

# 3. Merge
df = pd.merge(df_aq, df_weather, on="time")
df['time'] = pd.to_datetime(df['time'])

# --- MAPPING TO SCHEMA ---
final_df = pd.DataFrame()

# Time features
final_df['timestamp_unix'] = df['time'].view('int64') // 10**9
final_df['year'] = df['time'].dt.year
final_df['month'] = df['time'].dt.month
final_df['day'] = df['time'].dt.day
final_df['hour'] = df['time'].dt.hour
final_df['weekday'] = df['time'].dt.weekday

# Pollutants
final_df['pm25'] = df['pm2_5']
final_df['pm10'] = df['pm10']
final_df['no2'] = df['nitrogen_dioxide']
final_df['so2'] = df['sulphur_dioxide']
final_df['co'] = df['carbon_monoxide']
final_df['o3'] = df['ozone']

# Weather
final_df['temperature'] = df['temperature_2m']
final_df['humidity'] = df['relative_humidity_2m']
final_df['pressure'] = df['surface_pressure']
final_df['wind_speed'] = df['wind_speed_10m']
final_df['dew_point'] = df['dew_point_2m']

# 4. Calculate 'aqi' (Target)
def calculate_aqi(pm):
    if pm <= 12: return (50/12) * pm
    elif pm <= 35.4: return ((100-51)/(35.4-12.1)) * (pm-12.1) + 51
    else: return 150 

# Step-by-step calculation to avoid KeyError
final_df['aqi'] = final_df['pm25'].apply(calculate_aqi)
final_df['aqi_change_rate'] = final_df['aqi'].diff().fillna(0.0)

# --- SAVE ---
# Use os.path.join for Windows-friendly paths
save_dir = os.path.join(BASE_PATH, 'data', 'raw')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'karachi_schema_data.csv')

final_df.to_csv(save_path, index=False)

print(f"✅ Success! Data saved to: {save_path}")
print(f"📊 Total Rows Processed: {len(final_df)}")
