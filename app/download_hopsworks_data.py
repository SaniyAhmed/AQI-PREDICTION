"""
Download AI Forecast data from Hopsworks (Version 5)
Using explicit selection to avoid NoneType errors
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

print("📥 Fetching AI Forecast Data (Version 5)...")

try:
    # 1. Get the Feature Group object first
    fg = fs.get_feature_group(name="karachi_aqi_forecast", version=5)
    
    if fg is None:
        raise ValueError("Feature Group 'karachi_aqi_forecast' v5 not found!")

    # 2. Use select_all() - This is the most stable way to trigger a read in 4.2.x
    print("Executing select_all().read()...")
    df = fg.select_all().read()
    
    if df is None or df.empty:
        raise ValueError("Dataframe is empty. Ensure the Feature Group contains data.")
        
    print(f"✅ Successfully loaded {len(df)} forecast records")

except Exception as e:
    print(f"❌ Read failed: {e}")
    print("🔄 Attempting ultimate fallback (Feature View)...")
    try:
        # If the FG read fails, sometimes the Feature View is more accessible
        fv = fs.get_feature_view(name="karachi_aqi_view", version=1)
        df = fv.get_batch_data()
        print("✅ Loaded data via Feature View")
    except Exception as e2:
        print(f"❌ All retrieval methods failed. Error: {e2}")
        sys.exit(1)

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

# Save to CSV
output_path = 'data/forecast_data.csv'
df.to_csv(output_path, index=False)

print(f"💾 Saved {len(df)} rows to {output_path}")
print("🚀 Sync complete for Version 5!")
