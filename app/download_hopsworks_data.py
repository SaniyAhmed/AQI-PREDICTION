"""
Surgical Data Fetch for Karachi AQI Forecast v5
Optimized for hsfs 4.2.x
"""
import os
import pandas as pd
import sys

# Force disable experimental features
os.environ["HSFS_DISABLE_FLIGHT_CLIENT"] = "True"

import hopsworks

print("🔐 Logging into Hopsworks...")
try:
    project = hopsworks.login(api_key_value=os.getenv('MY_HOPSWORK_KEY'))
    fs = project.get_feature_store()
except Exception as e:
    print(f"❌ Login failed: {e}")
    sys.exit(1)

print("📥 Attempting to grab 'karachi_aqi_forecast' version 5...")

try:
    # We use get_or_create but with existing settings to force a retrieval
    fg = fs.get_or_create_feature_group(
        name="karachi_aqi_forecast",
        version=5,
        primary_key=['year', 'month', 'day', 'hour'], # Matching your known schema
        online_enabled=False
    )
    
    print(f"✅ Object found: {fg.name} v{fg.version}")
    
    # Use the most basic read possible
    print("Reading data...")
    df = fg.read()
    
    if df is None or df.empty:
        # Fallback: maybe it's version 1?
        print("⚠️ v5 empty, trying v1...")
        fg = fs.get_feature_group(name="karachi_aqi_forecast", version=1)
        df = fg.read()

    if df is not None and not df.empty:
        print(f"✅ Success! Loaded {len(df)} rows.")
        # Create directory and save
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/forecast_data.csv', index=False)
        print("💾 File saved to data/forecast_data.csv")
    else:
        print("❌ Dataframe is still empty after all attempts.")
        sys.exit(1)

except Exception as e:
    print(f"❌ Read failed with error: {str(e)}")
    print("\n💡 DEBUG TIP: Check your Hopsworks UI.")
    print("Go to Feature Groups -> Look for 'karachi_aqi_forecast'.")
    print("If it's not there, copy the EXACT name you see and tell me!")
    sys.exit(1)
