import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='hopsworks')
os.environ["HSFS_DISABLE_FLIGHT_CLIENT"] = "True"
os.environ["HSFS_DISABLE_HIVE_CLIENT"] = "True"

import hopsworks

st.set_page_config(page_title="Karachi AQI Sentinel", layout="wide", page_icon="🌬️")

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def load_all_data():
    df = None
    leaderboard = []
    best_model_obj = None
    
    local_file = "data/forecast_data.csv"
    if os.path.exists(local_file):
        try:
            df = pd.read_csv(local_file)
            
            # ✅ FIX: Construct timestamp from separate columns
            # Your error shows you have: ['year', 'month', 'day', 'hour']
            if all(col in df.columns for col in ['year', 'month', 'day', 'hour']):
                df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
                df = df.sort_values('timestamp')
            
            # ✅ FIX: Ensure target column exists
            # The error says 'predicted_aqi' exists, but let's be safe
            if 'predicted_aqi' not in df.columns and 'aqi' in df.columns:
                df['predicted_aqi'] = df['aqi']
                
        except Exception as e:
            st.error(f"Error processing CSV columns: {e}")

    # Model Registry Logic
    try:
        api_key = st.secrets["MY_HOPSWORK_KEY"]
        project = hopsworks.login(api_key_value=api_key)
        mr = project.get_model_registry()
        model_names = ["karachi_aqi_randomforest", "karachi_aqi_xgboost", "karachi_aqi_svr"]
        
        lowest_rmse = float('inf')
        for m_name in model_names:
            try:
                models = mr.get_models(m_name)
                if models:
                    latest = models[0]
                    metrics = latest.training_metrics if hasattr(latest, 'training_metrics') else {}
                    rmse = float(metrics.get('test_rmse', 999.0))
                    leaderboard.append({"Model": m_name.split("_")[-1].upper(), "RMSE": round(rmse, 4), "Raw": latest})
                    if rmse < lowest_rmse:
                        lowest_rmse = rmse
                        best_model_obj = latest
            except: continue
    except: pass

    return df, best_model_obj, leaderboard

df, best_model, leaderboard = load_all_data()

# --- UI LOGIC ---
st.title("🌬️ Karachi AQI Sentinel")

if df is not None and not df.empty:
    # KPI Row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Current AQI", f"{round(df['predicted_aqi'].iloc[-1], 1)}")
    with c2:
        st.metric("72h Average", f"{round(df['predicted_aqi'].mean(), 1)}")
    with c3:
        status = leaderboard[0]['Model'] if leaderboard else "Active"
        st.metric("Model Status", status)

    # Trend Chart
    st.subheader("📈 Air Quality Forecast")
    
    # ✅ FIX: Using 'timestamp' which we created above
    chart_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
    
    fig = px.line(df, x=chart_col, y="predicted_aqi", 
                  title="AQI Trend Over Time",
                  template="plotly_dark",
                  labels={chart_col: "Time", "predicted_aqi": "AQI Value"})
    
    fig.update_traces(line_color='#00d4ff', fill='tozeroy')
    st.plotly_chart(fig, use_container_width=True)

    # Tournament Table
    if leaderboard:
        st.subheader("🏆 Model Leaderboard")
        st.table(pd.DataFrame(leaderboard)[["Model", "RMSE"]])
else:
    st.warning("Data synced but schema mismatch detected. Check sidebar for debug info.")

# Sidebar Debug
st.sidebar.write("### Data Columns Detected:")
if df is not None:
    st.sidebar.write(df.columns.tolist())
