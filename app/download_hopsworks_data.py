"""
Dynamic Data Discovery Script
Lists all feature groups and pulls the most relevant forecast data.
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

print("🔍 Inspecting Feature Store...")

try:
    # Get all available feature groups
    fgs = fs.get_feature_groups()
    print(f"Found {len(fgs)} feature groups:")
    
    target_fg = None
    
    # List them out so you can see them in your GitHub Action logs
    for fg in fgs:
        print(f" - {fg.name} (Version: {fg.version})")
        # Logic to pick the right one: 
        # We want 'karachi_aqi_forecast' or 'karachi_aqi_prediction'
        if "karachi_aqi" in fg.name.lower():
            # If we find version 5, that's likely our winner
            if fg.version == 5:
                target_fg = fg
            # Fallback to whatever version of 'forecast' or 'prediction' is there
            elif "forecast" in fg.name.lower() or "prediction" in fg.name.lower():
                if not target_fg or fg.version > target_fg.version:
                    target_fg = fg

    if target_fg is None:
        print("❌ Could not find a matching forecast group. Defaulting to first available...")
        target_fg = fgs[0]

    print(f"🚀 Selected for Download: {target_fg.name} (v{target_fg.version})")
    
    # Read data
    df = target_fg.select_all().read()
    
    if df is None or df.empty:
        raise ValueError("Dataframe is empty.")
        
    print(f"✅ Successfully loaded {len(df)} records")

    # Create directory and save
    os.makedirs('data', exist_ok=True)
    output_path = 'data/forecast_data.csv'
    df.to_csv(output_path, index=False)
    print(f"💾 Saved to {output_path}")

except Exception as e:
    print(f"❌ Critical Error: {e}")
    sys.exit(1)
