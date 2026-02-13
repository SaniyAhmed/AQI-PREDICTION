import hopsworks
import os

def create_feature_view_v5():
    # 1. Login
    api_key = os.getenv('MY_HOPSWORK_KEY')
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()

    # 2. Reference the fresh Feature Group (Version 5)
    try:
        aqi_fg = fs.get_feature_group(name="karachi_aqi", version=5)
    except Exception as e:
        print(f"❌ Could not find Version 5. Is the materialization job still running? Error: {e}")
        return

    # 3. Create the Query (Selecting all columns)
    query = aqi_fg.select_all()

    # 4. Create the Feature View
    # This defines 'aqi' as the label for your ML model
    try:
        print("📦 Creating Feature View for Version 5...")
        feature_view = fs.get_or_create_feature_view(
            name="karachi_aqi_view",
            version=5,
            labels=["aqi"], 
            query=query,
            description="ML-ready view for Karachi AQI using Version 5 data"
        )
        print(f"✅ Success! Feature View 'karachi_aqi_view' is now linked to Version 5.")
    except Exception as e:
        print(f"❌ Error creating Feature View: {e}")

if __name__ == "__main__":
    create_feature_view_v5()
