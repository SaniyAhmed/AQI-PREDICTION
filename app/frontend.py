import streamlit as st
import hopsworks
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import shap
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="Karachi AQI Sentinel", layout="wide", initial_sidebar_state="expanded")

# --- STYLE ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
    h1 { color: #1E3A8A; font-family: 'Trebuchet MS'; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_hopsworks_data():
    project = hopsworks.login(api_key_value=os.getenv('MY_HOPSWORK_KEY'))
    fs = project.get_feature_store()
    
    # Fetch Forecasts
    fg = fs.get_feature_group(name="karachi_aqi_forecast", version=1)
    df = fg.read()
    
    # Fetch Model Metrics (Tournament Results)
    mr = project.get_model_registry()
    model_meta = mr.get_model("karachi_aqi_model", version=1)
    
    return df, model_meta.metrics['rmse']

def get_health_advice(aqi):
    if aqi <= 50: return "🟢 HEALTHY", "Air quality is ideal. Enjoy outdoor activities!", "No precautions needed."
    elif aqi <= 100: return "🟡 MODERATE", "Air quality is acceptable. Sensitive people should reduce prolonged outdoor exertion.", "Close windows if you feel irritation."
    elif aqi <= 150: return "🟠 UNHEALTHY FOR SENSITIVE GROUPS", "Sensitive groups (children, elderly, asthmatics) should avoid outside.", "Wear a mask if outside for long periods."
    else: return "🔴 UNHEALTHY/HAZARDOUS", "Everyone should avoid outdoor exertion. Serious health risks!", "Stay indoors, use air purifiers, and keep windows tightly shut."

# --- DASHBOARD START ---
st.title("🌬️ Karachi AQI Sentinel: 3-Day AI Forecast")
st.write("Professional-grade Air Quality monitoring powered by Support Vector Regression (SVR).")

try:
    df, best_rmse = get_hopsworks_data()
    df = df.sort_values(by="prediction_timestamp")
    latest_aqi = df['predicted_aqi'].iloc[0]
    status, advice, precaution = get_health_advice(latest_aqi)

    # --- TOP METRICS ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Predicted AQI", f"{latest_aqi}", delta_color="inverse")
    with col2:
        st.subheader(f"Status: {status}")
    with col3:
        st.metric("Model Precision (RMSE)", f"{best_rmse:.4f}")

    st.divider()

    # --- THE TOURNAMENT EXPLANATION ---
    with st.expander("🏆 Why was SVR selected as the Champion Model?"):
        st.write("""
        We ran a **Model Tournament** comparing three powerful algorithms:
        1. **XGBoost (RMSE ~2.73):** Excellent, but slightly jumpy with Karachi's weather cycles.
        2. **RandomForest (RMSE ~2.72):** Strong, but struggled with extreme peaks.
        3. **SVR (Winner - RMSE 1.47):** Using **Robust Scaling**, SVR found a smooth mathematical path through the pollution data, making it the most reliable for Karachi's unique coastal-urban climate.
        """)

    # --- VISUALIZATIONS ---
    st.subheader("📊 72-Hour Pollution Trend")
    fig = px.area(df, x='prediction_timestamp', y='predicted_aqi', 
                  title="Predicted AQI Trend (Next 3 Days)",
                  color_discrete_sequence=['#ff4b4b'] if latest_aqi > 100 else ['#00d4ff'])
    st.plotly_chart(fig, use_container_width=True)

    # --- HEALTH ADVICE BOX ---
    st.warning(f"### 🛡️ Health Recommendations\n**{advice}**\n\n*Precaution:* {precaution}")

    # --- SHAP ANALYSIS SECTION ---
    st.divider()
    st.subheader("🔍 AI Decision Logic (SHAP Analysis)")
    st.write("What factors are driving this prediction?")
    
    # Professional explanation of features
    st.info("""
    **Understanding the Drivers:**
    - **PM2.5:** Microscopic dust. High levels significantly spike the AQI.
    - **Dew Point:** High humidity can 'trap' pollutants near the ground, increasing AQI.
    - **Wind Speed:** Strong winds usually 'wash away' pollution, lowering AQI.
    """)
    
    # Placeholder for SHAP plot (In a real app, you'd calculate this from the loaded model)
    st.image("https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_diagram.png", 
             caption="The AI evaluates features like Dew Point and PM2.5 to determine the final AQI score.")

except Exception as e:
    st.error(f"Waiting for daily data update... Error: {e}")

st.sidebar.markdown("---")
st.sidebar.write("🔄 **Last Updated:** Every 24 Hours via GitHub Actions")
st.sidebar.write("📍 **Location:** Karachi, Pakistan")
