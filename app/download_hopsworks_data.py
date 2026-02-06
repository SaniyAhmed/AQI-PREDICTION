"""
Download forecast data from Hopsworks Feature Store
This script runs in GitHub Actions to sync data hourly
"""
import os

# ✅ FORCE DISABLE FLIGHT CLIENT
os.environ["HSFS_DISABLE_FLIGHT_CLIENT"] = "True"

import hopsworks
import pandas as pd

print("🔐 Logging into Hopsworks...")
project = hopsworks.login(api_key_value=os.getenv('MY_HOPSWORK_KEY'))
fs = project.get_feature_store()

print("📥 Fetching Data...")

# ✅ IMPLEMENTING THE FIX:
# We use engine="hive" explicitly in the read calls to ensure 
# it doesn't use the buggy Arrow Flight Query Service.
try:
    print("Trying Feature View with Hive engine...")
    fv = fs.get_feature_view(name="karachi_aqi_view", version=5)
    # Using read() with engine="hive" is the most stable way in GitHub Actions
    df = fv.get_batch_data(read_options={"use_hive": True})
    print("✅ Loaded data using Feature View (Hive)")
except Exception as e:
    print(f"⚠️ Feature View failed: {str(e)[:100]}")
    print("🔄 Switching to Feature Group direct read (Hive)...")
    
    fg = fs.get_feature_group(name="karachi_aqi", version=1)
    # Explicitly telling the engine NOT to use the online/flight service
    df = fg.read(read_options={"use_hive": True})
    print(f"✅ Loaded {len(df)} records from Feature Group (Hive)")

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Save to CSV
output_path = 'data/forecast_data.csv'
df.to_csv(output_path, index=False)

print(f"💾 Saved data to {output_path}")
print(f"📊 Total records: {len(df)}")
print("✅ Sync complete!")
