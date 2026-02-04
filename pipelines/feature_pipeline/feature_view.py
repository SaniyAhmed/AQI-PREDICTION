import hopsworks
import pandas as pd
import os

def run_backfill():
    # 1. Login
    api_key = os.getenv('MY_HOPSWORK_KEY') 
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    
    # 2. Load Data
    data_path = os.path.join("data", "processed", "processed_karachi_data.csv")
    df = pd.read_csv(data_path)

    # 3. TYPE FIX: Ensure pollutants are never seen as objects/dicts
    for col in df.columns:
        if col in ['year', 'month', 'day', 'hour']:
            df[col] = df[col].astype('int64')
        elif col == 'aqi':
            df[col] = df[col].astype('float64')
        else:
            # All pollutants and weather features
            df[col] = df[col].astype('float64')

    print(f"📋 Dataset ready for V4.")

    # 4. Create Feature Group VERSION 4
    try:
        print("📦 Registering Feature Group Version 4 (Clean Slate)...")
        aqi_fg = fs.get_or_create_feature_group(
            name="karachi_aqi",
            version=4, #NEW VERSION
            primary_key=['year', 'month', 'day', 'hour'],
            description="Hybrid Schema: Mandatory Pollutants + RFE Selected Supporting Features",
            online_enabled=True,
            statistics_config={"enabled": True, "histograms": True, "correlations": True}
        )
        
        # 5. Insert Data
        print("📤 Uploading Hybrid data to Hopsworks...")
        # wait_for_job=True is safer for the first time creation
        aqi_fg.insert(df, write_options={"wait_for_job": True})
        
        print(f"🚀 SUCCESS! Version 4 is now live.")
        
    except Exception as e:
        print(f"❌ Critical Error: {e}")

if __name__ == "__main__":
    run_backfill()