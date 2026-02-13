import hopsworks
import pandas as pd
import os

def run_backfill_v5():
    # 1. Login
    api_key = os.getenv('MY_HOPSWORK_KEY') 
    if not api_key:
        print("❌ Error: MY_HOPSWORK_KEY not found in environment variables.")
        return

    print("🚀 Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()

    # 2. Load Data
    data_path = os.path.join("data", "processed", "processed_karachi_data.csv")
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found.")
        return
    df = pd.read_csv(data_path)

    # 3. TYPE FIX: Ensure consistency with the Version 5 Float/Double schema
    for col in df.columns:
        if col in ['year', 'month', 'day', 'hour']:
            # Primary keys must be integers
            df[col] = df[col].astype('int64')
        else:
            # Pollutants, Weather, and AQI must be float64
            df[col] = df[col].astype('float64')

    print(f"📋 Dataset ready for Version 5 update.")

    # 4. Get or Create Feature Group VERSION 5
    # Since we removed deletion, this will just 'get' the group if it exists
    try:
        print("📦 Accessing Feature Group Version 5...")
        aqi_fg = fs.get_or_create_feature_group(
            name="karachi_aqi",
            version=5, 
            primary_key=['year', 'month', 'day', 'hour'],
            description="Hybrid Schema V5: Corrected types for pollutants.",
            online_enabled=True,
            statistics_config={"enabled": True, "histograms": True, "correlations": True}
        )
        
        # 5. Insert Data
        # Using wait_for_job=True to ensure materialization completes
        print("📤 Syncing data to Version 5 (Upserting records)...")
        aqi_fg.insert(df, write_options={"wait_for_job": True})
        
        print(f"🚀 SUCCESS! Version 5 is updated and live.")
        
    except Exception as e:
        print(f"❌ Critical Error during backfill: {e}")

if __name__ == "__main__":
    run_backfill_v5()
