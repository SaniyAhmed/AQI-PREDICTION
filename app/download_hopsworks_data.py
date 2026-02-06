"""
Download AI Forecast data from Hopsworks (Version 5)
Forcing legacy read paths to avoid Arrow Flight Binder Errors
"""
import os
import pandas as pd

# Force disable experimental features for stability in GitHub Actions
os.environ["HSFS_DISABLE_FLIGHT_CLIENT"] = "True"

import hopsworks

print("🔐 Logging into Hopsworks...")
try:
    project = hopsworks.login(api_key_value=os.getenv('MY_HOPSWORK_KEY'))
    fs = project.get_feature_store()
except Exception as e:
    print(f"❌ Login failed: {e}")
    exit(1)

print("📥 Fetching AI Forecast Data (Version 5)...")

try:
    # ✅ TARGETING THE CORRECT FORECAST GROUP AND VERSION
    print("Reading from Feature Group 'karachi_aqi_forecast' Version 5...")
    fg = fs.get_feature_group(name="karachi_aqi_forecast", version=5)
    
    # Read the data
    df = fg.read()
    
    if df is None or df.empty:
        raise ValueError("Dataframe is empty - Check if the Feature Group has data!")
        
    print(f"✅ Successfully loaded {len(df)} forecast records")

except Exception as e:
    print(f"❌ Primary read failed: {e}")
    print("🔄 Attempting legacy SQL read for version 5...")
    # Fallback SQL name usually follows `feature_group_name_version`
    query = "SELECT * FROM `karachi_aqi_forecast_5`"
    try:
        df = fs.sql(query)
        print("✅ Loaded data via direct SQL")
    except Exception as sql_e:
        print(f"❌ SQL Fallback also failed: {sql_e}")
        exit(1)

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

# Save to CSV for the Streamlit Frontend
output_path = 'data/forecast_data.csv'
df.to_csv(output_path, index=False)

print(f"💾 Saved {len(df)} rows to {output_path}")
print("🚀 Version 5 Forecast successfully synced!")
