"""
Download forecast data from Hopsworks Feature Store
This script runs in GitHub Actions to sync data hourly
"""
import os
import hopsworks
import pandas as pd

print("🔐 Logging into Hopsworks...")
api_key = os.getenv('HOPSWORKS_API_KEY')
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

print("📥 Fetching forecast data...")

# Try Feature View first, fallback to Feature Group
try:
    fv = fs.get_feature_view(name="karachi_aqi_view", version=5)
    df = fv.get_batch_data()
    print(f"✅ Loaded {len(df)} records from Feature View")
except Exception as e:
    print(f"⚠️ Feature View failed: {str(e)[:100]}")
    print("🔄 Trying Feature Group...")
    
    fg = fs.get_feature_group(name="karachi_aqi_forecast", version=1)
    df = fg.read()
    print(f"✅ Loaded {len(df)} records from Feature Group")

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Save to CSV
output_path = 'data/forecast_data.csv'
df.to_csv(output_path, index=False)

print(f"💾 Saved data to {output_path}")
print(f"📊 Total records: {len(df)}")
print("✅ Sync complete!")
