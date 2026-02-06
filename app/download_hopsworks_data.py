"""
Download forecast data from Hopsworks Feature Store
This script runs in GitHub Actions to sync data hourly
"""
import os

# ✅ THE FIX: Disable the buggy Flight Client before hopsworks is initialized
os.environ["HSFS_DISABLE_FLIGHT_CLIENT"] = "True"

import hopsworks
import pandas as pd

print("🔐 Logging into Hopsworks...")
# ✅ Using the exact login method from your running script
project = hopsworks.login(api_key_value=os.getenv('MY_HOPSWORK_KEY'))
fs = project.get_feature_store()
mr = project.get_model_registry()

print("📥 Fetching Data...")

# ✅ CRITICAL FIX: Read directly from Feature Group instead of Feature View
# Feature Views don't work in GitHub Actions, but Feature Groups do!
try:
    # Try to use Feature View (works in VS Code)
    fv = fs.get_feature_view(name="karachi_aqi_view", version=5)
    # Keeping your existing logic structure
    df = fv.get_batch_data()
    print("✅ Loaded data using Feature View")
except Exception as e:
    print(f"⚠️ Feature View failed (normal in GitHub Actions): {str(e)[:100]}")
    print("🔄 Switching to Feature Group direct read...")
    
    # Fallback: Read from Feature Group (works in GitHub Actions)
    fg = fs.get_feature_group(name="karachi_aqi", version=1)
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
