import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='hopsworks')
os.environ["HSFS_DISABLE_FLIGHT_CLIENT"] = "True"
os.environ["HSFS_DISABLE_HIVE_CLIENT"] = "True"

import hopsworks

# Page config
st.set_page_config(page_title="Karachi AQI Sentinel", layout="wide")

# Custom CSS for a professional look
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    .forecast-card { background-color: #161b22; border-radius: 10px; padding: 20px; text-align: center; border: 1px solid #30363d; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── DATA LOADING ────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_all_data():
    daily_summary_df = None
    current_aqi = None
    model_info = {}

    # Load forecast data from local CSV (fallback or primary)
    local_file = "data/forecast_data.csv"
    if os.path.exists(local_file):
        try:
            daily_summary_df = pd.read_csv(local_file)
            daily_summary_df['date'] = pd.to_datetime(daily_summary_df['date'])
            daily_summary_df = daily_summary_df.sort_values('date', ascending=True)
        except Exception as e:
            st.error(f"Error reading local CSV: {e}")

    # Fetch from Hopsworks for real-time AQI and Model Metrics
    try:
        api_key = st.secrets.get("MY_HOPSWORK_KEY") or os.getenv("MY_HOPSWORK_KEY")
        if api_key:
            project = hopsworks.login(api_key_value=api_key)
            fs = project.get_feature_store()
            mr = project.get_model_registry()

            # 1. Fetch Daily Summary (if not loaded from local)
            if daily_summary_df is None or daily_summary_df.empty:
                try:
                    fg_summary = fs.get_feature_group(name="karachi_aqi_daily_summary", version=2)
                    daily_summary_df = fg_summary.read()
                    daily_summary_df['date'] = pd.to_datetime(daily_summary_df['date'])
                    daily_summary_df = daily_summary_df.sort_values('date', ascending=True)
                except Exception as e:
                    st.warning(f"Could not fetch daily summary from Hopsworks: {e}")

            # 2. Fetch Latest Real-time AQI
            try:
                fg_historical = fs.get_feature_group(name="karachi_aqi", version=4)
                historical_df = fg_historical.read().sort_values(['year', 'month', 'day', 'hour'])
                if not historical_df.empty:
                    current_aqi = float(historical_df.iloc[-1]['aqi'])
            except Exception as e:
                pass

            # 3. Fetch Model Metrics
            try:
                models = mr.get_models("karachi_aqi_model")
                if models:
                    latest_model = models[0]
                    metrics = latest_model.training_metrics
                    
                    # UPDATED KEYS TO MATCH YOUR SCREENSHOT EXACTLY
                    model_info = {
                        "name": latest_model.name,
                        "version": latest_model.version,
                        "winner_rmse": metrics.get("winner_rmse", "Pending"),
                        "randomforest_rmse": metrics.get("randomforest_rmse", "Pending"),
                        "xgboost_rmse": metrics.get("xgboost_rmse", "Pending"),
                        "svr_rmse": metrics.get("svr_rmse", "Pending")
                    }
            except Exception:
                pass

            hopsworks.logout()
    except Exception as e:
        st.sidebar.warning(f"Hopsworks connection not available: {e}")

    return daily_summary_df, current_aqi, model_info

daily_summary_df, current_aqi, model_info = load_all_data()

# ─── UI LAYOUT ──────────────────────────────────────────────────────────────

# Sidebar
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("""
    **Karachi AQI Sentinel** monitors air quality in Karachi, Pakistan using machine learning.
    
    * **Data Source:** Hopsworks Feature Store
    * **Update Frequency:** Hourly (via GitHub Actions)
    * **Forecast Horizon:** 3 days
    * **Model:** Voting Ensemble (RF + XGBoost + SVR)
    """)
    
    st.divider()
    st.markdown("### 📊 Data Info")
    if daily_summary_df is not None:
        st.write(f"Total Records: {len(daily_summary_df)}")

# Main Header
st.markdown("""
    <div style="display: flex; align-items: center;">
        <h1 style="margin-right: 15px;">🌬️ KARACHI AQI SENTINEL</h1>
    </div>
    <p style="color: #8899ac;">REAL-TIME AIR QUALITY MONITORING & 3-DAY ML FORECAST</p>
    <hr style="margin-top: 0; margin-bottom: 25px;">
""", unsafe_allow_html=True)

if daily_summary_df is not None and not daily_summary_df.empty:
    
    # Calculate Metrics
    grand_avg = daily_summary_df['grand_avg_aqi'].iloc[-1] if 'grand_avg_aqi' in daily_summary_df.columns else daily_summary_df['daily_avg_aqi'].mean()
    
    # Filter for next 3 days
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    dates_tz_naive = daily_summary_df['date'].dt.tz_localize(None) if daily_summary_df['date'].dt.tz is not None else daily_summary_df['date']
    forecast_df = daily_summary_df[dates_tz_naive >= today].copy()
    
    if forecast_df.empty:
        forecast_df = daily_summary_df.copy()

    # Determine Top Model Name by matching RMSE values
    def get_model_name():
        w_rmse = model_info.get("winner_rmse")
        if w_rmse == "Pending" or w_rmse is None: return "Pending"
        
        # Check which individual model matches the winning RMSE
        # Matches your screenshot keys
        mapping = {
            "XGBoost": model_info.get("xgboost_rmse"),
            "Random Forest": model_info.get("randomforest_rmse"),
            "SVR": model_info.get("svr_rmse")
        }
        for name, val in mapping.items():
            try:
                if abs(float(val) - float(w_rmse)) < 0.0001:
                    return name
            except: continue
        return "Pending"

    top_model = get_model_name()

    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Latest AQI", f"{current_aqi:.1f}" if current_aqi else f"{forecast_df.iloc[0]['daily_avg_aqi']:.1f}")
    with k2:
        st.metric("3-Day Average", f"{grand_avg:.1f}")
    with k3:
        st.metric("Winner RMSE", model_info.get("winner_rmse", "Pending"))
    with k4:
        st.metric("Top Model", top_model)

    st.markdown("### 📊 DAILY FORECAST BREAKDOWN")
    
    # Forecast Cards
    cols = st.columns(min(len(forecast_df), 4))
    for idx, (_, row) in enumerate(forecast_df.iterrows()):
        if idx >= 4: break
        with cols[idx]:
            aqi_val = row['daily_avg_aqi']
            date_str = row['date'].strftime('%a, %b %d')
            
            # Determine color
            color = "#4caf50" if aqi_val <= 50 else "#ffc107" if aqi_val <= 100 else "#ff9800" if aqi_val <= 150 else "#f44336"
            
            st.markdown(f"""
                <div class="forecast-card">
                    <p style="color: #8899ac; font-size: 0.9rem; margin-bottom: 5px;">{date_str.upper()}</p>
                    <h2 style="margin: 0; font-size: 2.5rem; color: #ffffff;">{aqi_val:.1f}</h2>
                </div>
            """, unsafe_allow_html=True)

    # Trend Chart
    st.markdown("### 📈 3-DAY AQI FORECAST TREND")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['daily_avg_aqi'], mode='lines+markers', line=dict(color='#00d4ff', width=3)))
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Model Performance Table
    st.markdown("### 🤖 MODEL PERFORMANCE COMPARISON")
    perf_data = {
        "Model": ["Random Forest", "XGBoost", "SVR"],
        "RMSE": [
            model_info.get("randomforest_rmse", "Pending"),
            model_info.get("xgboost_rmse", "Pending"),
            model_info.get("svr_rmse", "Pending")
        ]
    }
    st.table(pd.DataFrame(perf_data))

else:
    st.error("⚠️ Forecast data not available. Please check GitHub Actions and Hopsworks.")
