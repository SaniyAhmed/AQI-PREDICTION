from flask import Flask, jsonify
from flask_cors import CORS
import hopsworks  # Ensure this is at the very top
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app) 

# Global variables
forecast_df = pd.DataFrame()
best_model_obj = None
aqi_model = None

def initialize_backend():
    global forecast_df, best_model_obj, aqi_model
    try:
        print("Connecting to Hopsworks...")
        api_key = os.getenv("MY_HOPSWORK_KEY")
        if not api_key:
            raise ValueError("API Key 'MY_HOPSWORK_KEY' not found!")

        # Explicitly login
        project = hopsworks.login(api_key_value=api_key)
        mr = project.get_model_registry()
        fs = project.get_feature_store()

        model_types = ["karachi_aqi_randomforest", "karachi_aqi_xgboost", "karachi_aqi_svr"]
        best_model_obj = None
        lowest_rmse = float('inf')

        print("🔍 SCANNING REGISTRY: Searching for newest versions (v24+)...")
        
        for m_name in model_types:
            try:
                # Get all versions available
                versions = mr.get_models(m_name)
                if versions:
                    # Sort manually to ensure we aren't getting old cached versions
                    versions.sort(key=lambda x: x.version, reverse=True)
                    latest = versions[0]
                    
                    # Fetch metrics
                    m_metrics = getattr(latest, "training_metrics", getattr(latest, "metrics", {}))
                    rmse = float(m_metrics.get('test_rmse', 999.0))
                    
                    print(f"Found {m_name} | Version: {latest.version} | RMSE: {rmse}")

                    if rmse < lowest_rmse:
                        lowest_rmse = rmse
                        best_model_obj = latest
            except Exception as e:
                print(f"⚠️ Error checking {m_name}: {e}")

        if best_model_obj:
            print(f"🏆 CHAMPION VERIFIED: {best_model_obj.name} (v{best_model_obj.version}) with RMSE {lowest_rmse}")
            model_path = best_model_obj.download()
            aqi_model = joblib.load(os.path.join(model_path, "karachi_aqi_model.pkl"))
        else:
            print("🚨 CRITICAL: No models found in registry!")

        # Load Forecast Data
        print(f"Reading Feature Group: karachi_aqi_forecast...")
        fg = fs.get_feature_group("karachi_aqi_forecast", version=1)
        forecast_df = fg.read(read_options={"use_hive": False})
        
        # Convert timestamps for JSON
        for col in forecast_df.columns:
            if pd.api.types.is_datetime64_any_dtype(forecast_df[col]):
                forecast_df[col] = forecast_df[col].astype(str)

        print("🚀 Backend Initialization Successful!")

    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        # Dummy data to prevent UI crash
        forecast_df = pd.DataFrame([{"prediction_timestamp": "Error", "predicted_aqi": 0.0}])

# Run initialization
initialize_backend()

# --- ROUTES ---

@app.route("/forecast", methods=['GET'])
def get_forecast():
    return jsonify(forecast_df.to_dict(orient="records"))

@app.route("/model-metrics", methods=['GET'])
def get_metrics():
    """Returns a dictionary with winner info and the full leaderboard"""
    if best_model_obj:
        project = hopsworks.login(api_key_value=os.getenv("MY_HOPSWORK_KEY"))
        mr = project.get_model_registry()
        
        leaderboard = []
        model_types = ["karachi_aqi_randomforest", "karachi_aqi_xgboost", "karachi_aqi_svr"]
        
        for m_name in model_types:
            try:
                versions = mr.get_models(m_name)
                if versions:
                    versions.sort(key=lambda x: x.version, reverse=True)
                    latest = versions[0]
                    m_metrics = getattr(latest, "training_metrics", getattr(latest, "metrics", {}))
                    leaderboard.append({
                        "Model": m_name.replace("karachi_aqi_", "").title(),
                        "RMSE": round(float(m_metrics.get("test_rmse", 0.0)), 4),
                        "Status": "Champion" if latest.name == best_model_obj.name else "Challenger"
                    })
            except:
                continue

        m_metrics = getattr(best_model_obj, "training_metrics", getattr(best_model_obj, "metrics", {}))
        return jsonify({
            "test_rmse": round(float(m_metrics.get("test_rmse", 0.0)), 4),
            "test_r2": round(float(m_metrics.get("test_r2", 0.0)), 4),
            "test_mae": round(float(m_metrics.get("test_mae", 0.0)), 4),
            "winner_name": best_model_obj.name.replace("karachi_aqi_", "").replace("_", " ").title(),
            "version": best_model_obj.version,
            "leaderboard": leaderboard
        })
    return jsonify({"winner_name": "None", "test_rmse": 0.0, "leaderboard": []})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)