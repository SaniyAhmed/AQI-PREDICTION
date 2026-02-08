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

# TARGETING DAILY SUMMARY VERSION 2
FG_NAME = "karachi_aqi_daily_summary"
FG_VERSION = 2

print(f"📥 Fetching {FG_NAME} v{FG_VERSION}...")

try:
    # Get the daily summary feature group
    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)
    df = fg.read()
    
    if df is None or df.empty:
        raise ValueError("The data returned is empty.")
    
    print(f"✅ Successfully fetched {len(df)} rows from {FG_NAME} v{FG_VERSION}")
    
    # Ensure date column is properly formatted
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # Sort by date
    df = df.sort_values('date', ascending=True)
    
    # Display summary
    print("\n📊 Data Summary:")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"   Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Grand Average AQI: {df['grand_avg_aqi'].iloc[0] if 'grand_avg_aqi' in df.columns else 'N/A'}")
    
    # Save for Streamlit
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/forecast_data.csv', index=False)
    print(f"\n✅ Successfully saved to data/forecast_data.csv")
    
    # Display the data
    print("\n📋 Forecast Data:")
    print(df.to_string(index=False))

except Exception as e:
    print(f"❌ Failed to fetch data: {e}")
    sys.exit(1)

finally:
    try:
        hopsworks.logout()
        print("\n🔓 Logged out from Hopsworks")
    except:
        pass
