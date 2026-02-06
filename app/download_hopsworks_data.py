"""
Surgical Data Fetch for Karachi AQI Forecast v5
Optimized for hsfs 4.2.x - Filters for Today + Next 3 Days
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
        print(f"❌ Version {FG_VERSION} not found.")
        if fgs:
            target_fg = sorted(fgs, key=lambda x: x.version, reverse=True)[0]
            print(f"⚠️ Falling back to Version {target_fg.version}")
        else:
            raise ValueError(f"No Feature Groups found with name '{FG_NAME}'")

    print(f"🚀 Reading data from {target_fg.name} v{target_fg.version}...")
    
    # Read the data into a pandas dataframe
    df = target_fg.read()
    
    if df is None or df.empty:
        raise ValueError("The data returned is empty.")

    # --- FILTERING FOR RECENT PREDICTIONS (TODAY + 3 DAYS) ---
    print("🔄 Filtering for the most recent model run (Today + 3 Days)...")
    
    if all(col in df.columns for col in ['year', 'month', 'day', 'hour']):
        # 1. Create timestamp objects
        df['temp_ts'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        
        # 2. Identify the "Latest Run Date" available in the data
        # This ensures we pick the most recent predictions made by the model
        latest_date_in_data = df['temp_ts'].max().normalize()
        start_date = latest_date_in_data - pd.Timedelta(days=0) # Start from the latest predicted day
        end_date = start_date + pd.Timedelta(days=3)    # Go 3 days forward
        
        # 3. Apply the strict filter
        mask = (df['temp_ts'] >= start_date) & (df['temp_ts'] <= end_date)
        df = df.loc[mask]
        
        # 4. Sort chronologically for the dashboard
        df = df.sort_values(by='temp_ts', ascending=True)
        
        # Clean up temporary column
        df = df.drop(columns=['temp_ts'])
    
    # ---------------------------------------------------------

    # Success path
    print(f"✅ Filtered down to {len(df)} records (Today + 3 Days).")
    
    # Save for Streamlit
    os.makedirs('data', exist_ok=True)
    output_path = 'data/forecast_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"💾 Saved {len(df)} rows to {output_path}")
    print("✅ Sync and Filter complete for Version 5!")

except Exception as e:
    print(f"❌ Final attempt failed: {e}")
    sys.exit(1)
