import os
import pandas as pd
import sys
import hopsworks

# Force legacy read paths
os.environ["HSFS_DISABLE_FLIGHT_CLIENT"] = "True"

print("🔐 Logging into Hopsworks...")
try:
    project = hopsworks.login(api_key_value=os.getenv('MY_HOPSWORK_KEY'))
    fs = project.get_feature_store()
except Exception as e:
    print(f"❌ Login failed: {e}")
    sys.exit(1)

# TARGETING VERSION 1
FG_NAME = "karachi_aqi_forecast"
FG_VERSION = 1

print(f"📥 Fetching {FG_NAME} v{FG_VERSION}...")

try:
    # Get the specific version 1
    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)
    df = fg.read()
    
    if df is None or df.empty:
        raise ValueError("The data returned is empty.")

    print("🔄 Filtering for the most recent model run (Today + 3 Days)...")
    if all(col in df.columns for col in ['year', 'month', 'day', 'hour']):
        df['temp_ts'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        
        # Filter logic
        latest_date_in_data = df['temp_ts'].max().normalize()
        start_date = latest_date_in_data
        end_date = start_date + pd.Timedelta(days=3)
        
        mask = (df['temp_ts'] >= start_date) & (df['temp_ts'] <= end_date)
        df = df.loc[mask].sort_values(by='temp_ts', ascending=True)
        df = df.drop(columns=['temp_ts'])

    # Save for Streamlit
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/forecast_data.csv', index=False)
    print(f"✅ Successfully saved {len(df)} rows from Version 1 to data/forecast_data.csv")

except Exception as e:
    print(f"❌ Final attempt failed: {e}")
    sys.exit(1)
