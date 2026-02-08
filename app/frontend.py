import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
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
    daily_summary_df = None
    current_aqi = None
    model_info = {}
    
    # Option 1: Read from local CSV (updated by GitHub Actions)
    local_file = "data/forecast_data.csv"
    
    if os.path.exists(local_file):
        try:
            daily_summary_df = pd.read_csv(local_file)
            daily_summary_df['date'] = pd.to_datetime(daily_summary_df['date'])
            daily_summary_df = daily_summary_df.sort_values('date', ascending=True)
        except Exception as e:
            st.error(f"Error reading local CSV: {e}")
    
    # Option 2: Fetch directly from Hopsworks (fallback or for model info)
    try:
        api_key = st.secrets.get("MY_HOPSWORK_KEY") or os.getenv("MY_HOPSWORK_KEY")
        if api_key:
            project = hopsworks.login(api_key_value=api_key)
            fs = project.get_feature_store()
            mr = project.get_model_registry()
            
            # Fetch daily summary if local file doesn't exist
            if daily_summary_df is None or daily_summary_df.empty:
                try:
                    fg_summary = fs.get_feature_group(name="karachi_aqi_daily_summary", version=2)
                    daily_summary_df = fg_summary.read()
                    daily_summary_df['date'] = pd.to_datetime(daily_summary_df['date'])
                    daily_summary_df = daily_summary_df.sort_values('date', ascending=True)
                except Exception as e:
                    st.warning(f"Could not fetch daily summary: {e}")
            
            # Fetch current AQI from historical data
            try:
                fg_historical = fs.get_feature_group(name="karachi_aqi", version=4)
                historical_df = fg_historical.read().sort_values(['year', 'month', 'day', 'hour'])
                if not historical_df.empty:
                    current_aqi = float(historical_df.iloc[-1]['aqi'])
            except Exception as e:
                st.warning(f"Could not fetch current AQI: {e}")
            
            # Fetch model info from model registry
            try:
                models = mr.get_models("karachi_aqi_model")
                if models:
                    latest_model = models[0]
                    model_info = {
                        "name": latest_model.name,
                        "version": latest_model.version,
                        "test_rmse": latest_model.training_metrics.get("test_rmse", "N/A"),
                        "winner_rmse": latest_model.training_metrics.get("winner_rmse", "N/A"),
                        "description": latest_model.description
                    }
            except Exception as e:
                st.warning(f"Could not fetch model info: {e}")
            
            hopsworks.logout()
    except Exception as e:
        st.warning(f"Hopsworks connection not available: {e}")
    
    return daily_summary_df, current_aqi, model_info

daily_summary_df, current_aqi, model_info = load_all_data()

# --- UI LOGIC ---
st.title("🌬️ Karachi AQI Sentinel")
st.markdown("### Real-time Air Quality Monitoring & 3-Day Forecast")

if daily_summary_df is not None and not daily_summary_df.empty:
    # Extract metrics
    grand_avg = daily_summary_df['grand_avg_aqi'].iloc[0] if 'grand_avg_aqi' in daily_summary_df.columns else daily_summary_df['daily_avg_aqi'].mean()
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if current_aqi:
            st.metric("Current AQI", f"{round(current_aqi, 1)}", 
                     help="Latest measured AQI value")
        else:
            st.metric("Current AQI", "Loading...", help="Latest measured AQI value")
    
    with col2:
        st.metric("3-Day Average", f"{round(grand_avg, 1)}", 
                 help="Average predicted AQI for next 3 days")
    
    with col3:
        if model_info and 'test_rmse' in model_info:
            st.metric("Model RMSE", f"{round(float(model_info['test_rmse']), 2)}", 
                     help="Model prediction error metric")
        else:
            st.metric("Model Status", "Active", help="Model is running")
    
    with col4:
        if model_info and 'description' in model_info:
            st.metric("Top Model", model_info['description'].replace("Winner: ", ""), 
                     help="Best performing model")
        else:
            st.metric("Top Model", "Ensemble", help="Voting ensemble model")
    
    # Daily Breakdown
    st.subheader("📊 Daily Forecast Breakdown")
    
    # Create cards for each day
    cols = st.columns(len(daily_summary_df))
    for idx, (_, row) in enumerate(daily_summary_df.iterrows()):
        with cols[idx]:
            date_str = pd.to_datetime(row['date']).strftime('%b %d')
            aqi_val = round(row['daily_avg_aqi'], 1)
            
            # AQI category
            if aqi_val <= 50:
                category = "Good"
                color = "🟢"
            elif aqi_val <= 100:
                category = "Moderate"
                color = "🟡"
            elif aqi_val <= 150:
                category = "Unhealthy (SG)"
                color = "🟠"
            elif aqi_val <= 200:
                category = "Unhealthy"
                color = "🔴"
            else:
                category = "Hazardous"
                color = "🟣"
            
            st.markdown(f"""
            <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="margin: 0; color: #00d4ff;">{date_str}</h3>
                <h1 style="margin: 10px 0; color: white;">{aqi_val}</h1>
                <p style="margin: 0; color: #888;">{color} {category}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Trend Chart
    st.subheader("📈 3-Day AQI Trend")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_summary_df['date'],
        y=daily_summary_df['daily_avg_aqi'],
        mode='lines+markers',
        name='Daily Average AQI',
        line=dict(color='#00d4ff', width=3),
        marker=dict(size=10),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.2)'
    ))
    
    # Add grand average line
    fig.add_hline(y=grand_avg, line_dash="dash", 
                  line_color="yellow", 
                  annotation_text=f"3-Day Avg: {round(grand_avg, 1)}",
                  annotation_position="right")
    
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="AQI Value",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data Table
    with st.expander("📋 View Detailed Forecast Data"):
        display_df = daily_summary_df.copy()
        display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
        st.dataframe(display_df, use_container_width=True)
    
    # Model Information
    if model_info:
        st.subheader("🤖 Model Information")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"""
            **Model Name:** {model_info.get('name', 'N/A')}  
            **Version:** {model_info.get('version', 'N/A')}  
            **Type:** {model_info.get('description', 'Ensemble Model')}
            """)
        with col2:
            st.info(f"""
            **Test RMSE:** {model_info.get('test_rmse', 'N/A')}  
            **Winner RMSE:** {model_info.get('winner_rmse', 'N/A')}  
            **Status:** ✅ Active
            """)

else:
    st.warning("⚠️ Forecast data not available. Please wait for the next data sync or check your Hopsworks connection.")
    st.info("💡 GitHub Actions should sync data hourly. If this persists, check your workflow logs.")

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
**Karachi AQI Sentinel** monitors air quality in Karachi, Pakistan using machine learning.

- **Data Source:** Hopsworks Feature Store
- **Update Frequency:** Hourly (via GitHub Actions)
- **Forecast Horizon:** 3 days
- **Model:** Ensemble (RF + XGBoost + SVR)
""")

if daily_summary_df is not None:
    st.sidebar.write("### 📊 Data Info")
    st.sidebar.write(f"**Records:** {len(daily_summary_df)}")
    st.sidebar.write(f"**Forecast Period:** {daily_summary_df['date'].min().strftime('%Y-%m-%d')} to {daily_summary_df['date'].max().strftime('%Y-%m-%d')}")

st.sidebar.write("---")
st.sidebar.caption("🔄 Last updated: Check GitHub Actions for sync status")
