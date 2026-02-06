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
    
    # This reads the exact file created by your GitHub Action
    local_file = "data/forecast_data.csv"
    
    if os.path.exists(local_file):
        try:
            df = pd.read_csv(local_file)
            
            # Construct timestamp from separate columns
            if all(col in df.columns for col in ['year', 'month', 'day', 'hour']):
                df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
                # Sort Chronologically (Earliest to Latest) for the Chart
                df = df.sort_values('timestamp', ascending=True)
            
            # Ensure target column exists
            if 'predicted_aqi' not in df.columns and 'aqi' in df.columns:
                df['predicted_aqi'] = df['aqi']
                
        except Exception as e:
            st.error(f"Error processing CSV columns: {e}")

    # Model Registry Logic (Optional/Leaderboard)
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
                    leaderboard.append({"Model": m_name.split("_")[-1].upper(), "RMSE": round(rmse, 4)})
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
    # Get the "Current" prediction (the earliest timestamp in our filtered 4-day window)
    current_val = df['predicted_aqi'].iloc[0] 
    
    # KPI Row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Current AQI (Predicted)", f"{round(current_val, 1)}")
    with c2:
        st.metric("4-Day Forecast Average", f"{round(df['predicted_aqi'].mean(), 1)}")
    with c3:
        status = leaderboard[0]['Model'] if leaderboard else "Active"
        st.metric("Top Model", status)

    # Trend Chart
    st.subheader("📈 4-Day Air Quality Forecast")
    
    chart_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
    
    fig = px.line(df, x=chart_col, y="predicted_aqi", 
                  title="AQI Trend (Today + Next 3 Days)",
                  template="plotly_dark",
                  labels={chart_col: "Time", "predicted_aqi": "AQI Value"})
    
    fig.update_traces(line_color='#00d4ff', fill='tozeroy')
    st.plotly_chart(fig, use_container_width=True)

    # Tournament Table
    if leaderboard:
        st.subheader("🏆 Model Leaderboard")
        st.table(pd.DataFrame(leaderboard))
else:
    st.warning("Data file not found or empty. Please wait for GitHub Actions to complete.")

# Sidebar Debug
st.sidebar.write("### ℹ️ Dashboard Info")
st.sidebar.info("This dashboard shows the most recent model predictions synced from Hopsworks FG v5.")
if df is not None:
    st.sidebar.write("**Data Columns:**", df.columns.tolist())
    st.sidebar.write("**Forecast Range:**", f"{df['timestamp'].min()} to {df['timestamp'].max()}")
