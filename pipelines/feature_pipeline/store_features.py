import hopsworks
import pandas as pd
import os

def run_backfill():
    # 1. Login
    api_key = os.getenv('MY_HOPSWORK_KEY') 
    if not api_key:
        print("❌ Error: MY_HOPSWORK_KEY not found in environment variables.")
        return

    print("🚀 Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    
    # 2. Load the Hybrid Processed Data
    data_path = os.path.join("data", "processed", "processed_karachi_data.csv")
    
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found. Did you run the new preprocessing.py?")
        return

    df = pd.read_csv(data_path)

    # 3. Final Type Casting (Crucial for Hopsworks 'bigint')
    # This ensures all integer columns are 64-bit
    for col in df.columns:
        if df[col].dtype == 'int64' or df[col].dtype == 'int32':
            df[col] = df[col].astype('int64')
        elif df[col].dtype == 'float32':
            df[col] = df[col].astype('float64')

    print(f"📋 Dataset ready for V3.")
    print(f"📊 Features including pollutants: {[c for c in df.columns if c not in ['year','month','day','hour','aqi']]}")

    # 4. Create Feature Group VERSION 4
    # We use Version 4 because it represents the "Hybrid Pollutant + AI" schema
    try:
        print("📦 Registering Feature Group Version 3...")
        aqi_fg = fs.get_or_create_feature_group(
            name="karachi_aqi",
            version=4, 
            primary_key=['year', 'month', 'day', 'hour'],
            description="Hybrid Schema: Mandatory Pollutants + RFE Selected Supporting Features",
            online_enabled=True,
            statistics_config={"enabled": True, "histograms": True, "correlations": True}
        )
        
        # 5. Insert Data
        print("📤 Uploading Hybrid data to Hopsworks...")
        # wait_for_job=False returns control to you immediately
        aqi_fg.insert(df, write_options={"wait_for_job": False})
        
        print(f"🚀 SUCCESS! Version 4 is now materializing in Hopsworks.")
        print("Note: All future scripts (training, live fetch) should now use version=4.")
        
    except Exception as e:
        print(f"❌ Critical Error: {e}")

if __name__ == "__main__":
    run_backfill()
