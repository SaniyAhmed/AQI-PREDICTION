"""
Surgical Data Fetch for Karachi AQI Forecast v5
Optimized for hsfs 4.2.x - Sorting for Recent Predictions only
"""
import os
import pandas as pd
import sys

# Force legacy read paths to avoid Arrow Flight Binder Errors
os.environ["HSFS_DISABLE_FLIGHT_CLIENT"] = "True"

import hopsworks

print("🔐 Logging into Hopsworks...")
try:
    # Authenticate using your environment variable
    project = hopsworks.login(api_key_value=os.getenv('MY_HOPSWORK_KEY'))
    fs = project.get_feature_store()
except Exception as e:
    print(f"❌ Login failed: {e}")
    sys.exit(1)

# --- TARGET CONFIGURATION ---
FG_NAME = "karachi_aqi_forecast"
FG_VERSION = 5
# ----------------------------

print(f"📥 Fetching ONLY {FG_NAME} v{FG_VERSION}...")

try:
    # Use the plural method with name filter (required for hsfs 4.x)
    fgs = fs.get_feature_groups(name=FG_NAME)
    
    # Manually extract version 5 from the results
    target_fg = next((fg for fg in fgs if fg.version == FG_VERSION), None)
    
    if target_fg is None:
        print(f"❌ Version {FG_VERSION} not found in the list for {FG_NAME}.")
        print("🔍 Available versions actually found in your project:")
        for fg in fgs:
            print(f" - Version: {fg.version}")
        
        # Fallback: Pick highest version so the dashboard still has data
        if fgs:
            target_fg = sorted(fgs, key=lambda x: x.version, reverse=True)[0]
            print(f"⚠️ Falling back to highest available: Version {target_fg.version}")
        else:
            raise ValueError(f"No Feature Groups found with name '{FG_NAME}'")

    print(f"🚀 Reading data from {target_fg.name} v{target_fg.version}...")
    
    # Read the data into a pandas dataframe
    df = target_fg.read()
    
    if df is None or df.empty:
        raise ValueError("The data returned is empty.")

    # --- NEW: SORTING AND FILTERING FOR RECENT PREDICTIONS ---
    print("🔄 Processing most recent predictions...")
    
    # 1. Create a real timestamp to sort correctly
    if all(col in df.columns for col in ['year', 'month', 'day', 'hour']):
        df['temp_ts'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        
        # 2. Sort so the newest data is at the top
        df = df.sort_values(by='temp_ts', ascending=False)
        
        # 3. Optional: If you want ONLY the latest day (24 hours), uncomment next line:
        # df = df.head(24)
        
        # Clean up temporary timestamp column
        df = df.drop(columns=['temp_ts'])
    
    # ---------------------------------------------------------

    # Success path
    print(f"✅ Successfully processed {len(df)} records.")
    
    # Save for Streamlit
    os.makedirs('data', exist_ok=True)
    output_path = 'data/forecast_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"💾 Saved {len(df)} rows to {output_path}")
    print("✅ Sync and Sort complete for Version 5!")

except Exception as e:
    print(f"❌ Final attempt failed: {e}")
    sys.exit(1)
