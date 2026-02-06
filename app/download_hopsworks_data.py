import os
import pandas as pd
import sys

# Force legacy read paths to avoid Arrow Flight Binder Errors
os.environ["HSFS_DISABLE_FLIGHT_CLIENT"] = "True"

import hopsworks

print("🔐 Logging into Hopsworks...")
try:
    # Use your secret key
    project = hopsworks.login(api_key_value=os.getenv('MY_HOPSWORK_KEY'))
    fs = project.get_feature_store()
except Exception as e:
    print(f"❌ Login failed: {e}")
    sys.exit(1)

# TARGET: The specific forecast group you mentioned
FG_NAME = "karachi_aqi_forecast"
FG_VERSION = 5

print(f"📥 Fetching ONLY {FG_NAME} v{FG_VERSION}...")

try:
    # Use the plural method but WITH the name argument to avoid the "missing argument" error
    fgs = fs.get_feature_groups(name=FG_NAME)
    
    # Filter for version 5 manually from the returned list
    target_fg = next((fg for fg in fgs if fg.version == FG_VERSION), None)
    
    if target_fg is None:
        print(f"❌ Version {FG_VERSION} not found in the list for {FG_NAME}.")
        print("Available versions found:")
        for fg in fgs:
            print(f" - Version: {fg.version}")
        
        # Emergency Fallback: If 5 isn't there, take the highest version available
        if fgs:
            target_fg = sorted(fgs, key=lambda x: x.version, reverse=True)[0]
            print(f"⚠️ Falling back to highest available: Version {target_fg.version}")
        else:
            raise ValueError(f"No Feature Groups found with name '{FG_NAME}'")

    print(f"🚀 Reading data from {target_fg.name} v{target_fg.version}...")
    
    # The most stable read method for HSFS 4.x
    df = target_fg.read()
    
    if df is None or df.empty:
        raise ValueError("The data returned is empty.")

    # Success path
    print(f"✅ Successfully loaded {len(df)} records.")
    os.makedirs('data', exist_ok=True)
    output_path = 'data/forecast_data.csv'
    df.to_csv(output_path, index=False)
    print(f"💾 Saved to {output_path}")

except Exception as e:
    print(f"❌ Final attempt failed: {e}")
    sys.exit(1)import os
import pandas as pd
import sys

# Force legacy read paths to avoid Arrow Flight Binder Errors
os.environ["HSFS_DISABLE_FLIGHT_CLIENT"] = "True"

import hopsworks

print("🔐 Logging into Hopsworks...")
try:
    # Use your secret key
    project = hopsworks.login(api_key_value=os.getenv('MY_HOPSWORK_KEY'))
    fs = project.get_feature_store()
except Exception as e:
    print(f"❌ Login failed: {e}")
    sys.exit(1)

# TARGET: The specific forecast group you mentioned
FG_NAME = "karachi_aqi_forecast"
FG_VERSION = 5

print(f"📥 Fetching ONLY {FG_NAME} v{FG_VERSION}...")

try:
    # Use the plural method but WITH the name argument to avoid the "missing argument" error
    fgs = fs.get_feature_groups(name=FG_NAME)
    
    # Filter for version 5 manually from the returned list
    target_fg = next((fg for fg in fgs if fg.version == FG_VERSION), None)
    
    if target_fg is None:
        print(f"❌ Version {FG_VERSION} not found in the list for {FG_NAME}.")
        print("Available versions found:")
        for fg in fgs:
            print(f" - Version: {fg.version}")
        
        # Emergency Fallback: If 5 isn't there, take the highest version available
        if fgs:
            target_fg = sorted(fgs, key=lambda x: x.version, reverse=True)[0]
            print(f"⚠️ Falling back to highest available: Version {target_fg.version}")
        else:
            raise ValueError(f"No Feature Groups found with name '{FG_NAME}'")

    print(f"🚀 Reading data from {target_fg.name} v{target_fg.version}...")
    
    # The most stable read method for HSFS 4.x
    df = target_fg.read()
    
    if df is None or df.empty:
        raise ValueError("The data returned is empty.")

    # Success path
    print(f"✅ Successfully loaded {len(df)} records.")
    os.makedirs('data', exist_ok=True)
    output_path = 'data/forecast_data.csv'
    df.to_csv(output_path, index=False)
    print(f"💾 Saved to {output_path}")

except Exception as e:
    print(f"❌ Final attempt failed: {e}")
    sys.exit(1)
