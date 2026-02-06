"""
Download forecast data from Hopsworks Feature Store
Forcing legacy read paths to avoid Arrow Flight Binder Errors
"""
import os
import pandas as pd

# Force disable experimental features
os.environ["HSFS_DISABLE_FLIGHT_CLIENT"] = "True"

import hopsworks

print("🔐 Logging into Hopsworks...")
# Using the key from your secrets
project = hopsworks.login(api_key_value=os.getenv('MY_HOPSWORK_KEY'))
fs = project.get_feature_store()

print("📥 Fetching Data...")

try:
    # 1. Try reading from Feature Group directly - usually the most stable path
    print("Reading from Feature Group 'karachi_aqi'...")
    fg = fs.get_feature_group(name="karachi_aqi", version=1)
    
    # .read() with no arguments is the standard fallback
    # We bypass the Feature View to avoid the complex SQL join that is failing
    df = fg.read()
    
    if df is None or df.empty:
        raise ValueError("Dataframe is empty")
        
    print(f"✅ Successfully loaded {len(df)} records")

except Exception as e:
    print(f"❌ Primary read failed: {e}")
    print("🔄 Attempting legacy SQL read...")
    # 2. Final fallback: Direct SQL if the object-based read fails
    query = "SELECT * FROM `karachi_aqi_1`"
    df = fs.sql(query)
    print("✅ Loaded data via direct SQL")

# Create data directory
os.makedirs('data', exist_ok=True)

# Save to CSV
output_path = 'data/forecast_data.csv'
df.to_csv(output_path, index=False)

print(f"💾 Saved {len(df)} rows to {output_path}")
print("✅ Sync complete!")
