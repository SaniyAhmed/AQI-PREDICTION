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
            st.sidebar.success("✅ Data loaded from local cache")
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
                    st.sidebar.success("✅ Data loaded from Hopsworks")
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
        st.sidebar.warning(f"Hopsworks connection not available: {e}")
    
    return daily_summary_df, current_aqi, model_info

daily_summary_df, current_aqi, model_info = load_all_data()

# --- UI LOGIC ---
st.title("🌬️ Karachi AQI Sentinel")
st.markdown("### Real-time Air Quality Monitoring & 3-Day Forecast")

if daily_summary_df is not None and not daily_summary_df.empty:
    # Extract the correct grand average from the latest row
    grand_avg = daily_summary_df['grand_avg_aqi'].iloc[-1] if 'grand_avg_aqi' in daily_summary_df.columns else daily_summary_df['daily_avg_aqi'].mean()
    
    # Filter only future dates (forecast)
    today = pd.Timestamp.now().normalize()
    forecast_df = daily_summary_df[daily_summary_df['date'] >= today].copy()
    
    # If no future dates, show all data (for testing/demo)
    if forecast_df.empty:
        forecast_df = daily_summary_df.copy()
        st.info("ℹ️ Showing all available forecast data (including past dates for demo)")
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if current_aqi:
            st.metric("Current AQI", f"{round(current_aqi, 1)}", 
                     help="Latest measured AQI value")
        else:
            # Use first forecast value if current not available
            st.metric("Latest AQI", f"{round(forecast_df.iloc[0]['daily_avg_aqi'], 1)}", 
                     help="Most recent AQI value")
    
    with col2:
        st.metric("3-Day Average", f"{round(grand_avg, 1)}", 
                 help="Average predicted AQI for next 3 days")
    
    with col3:
        if model_info and 'test_rmse' in model_info and model_info['test_rmse'] != "N/A":
            st.metric("Model RMSE", f"{round(float(model_info['test_rmse']), 2)}", 
                     help="Ensemble model prediction error")
        else:
            st.metric("Forecast Days", f"{len(forecast_df)}", 
                     help="Number of forecast days available")
    
    with col4:
        if model_info and 'description' in model_info:
            winner_name = model_info['description'].replace("Winner: ", "")
            st.metric("Top Model", winner_name, 
                     help="Best performing individual model")
        else:
            st.metric("Model Type", "Ensemble", 
                     help="Voting ensemble model")
    
    # Daily Breakdown
    st.subheader("📊 Daily Forecast Breakdown")
    
    # Create cards for each forecast day
    cols = st.columns(min(len(forecast_df), 4))  # Max 4 columns
    for idx, (_, row) in enumerate(forecast_df.iterrows()):
        if idx >= 4:  # Limit to 4 cards
            break
        with cols[idx]:
            date_str = pd.to_datetime(row['date']).strftime('%b %d')
            aqi_val = round(row['daily_avg_aqi'], 1)
            
            # AQI category
            if aqi_val <= 50:
                category = "Good"
                color = "🟢"
                bg_color = "#2d5016"
            elif aqi_val <= 100:
                category = "Moderate"
                color = "🟡"
                bg_color = "#5c4a1a"
            elif aqi_val <= 150:
                category = "Unhealthy (SG)"
                color = "🟠"
                bg_color = "#663d1a"
            elif aqi_val <= 200:
                category = "Unhealthy"
                color = "🔴"
                bg_color = "#661a1a"
            else:
                category = "Hazardous"
                color = "🟣"
                bg_color = "#4a1a4a"
            
            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #00d4ff;">
                <h3 style="margin: 0; color: #00d4ff;">{date_str}</h3>
                <h1 style="margin: 10px 0; color: white; font-size: 2.5em;">{aqi_val}</h1>
                <p style="margin: 0; color: #ddd; font-size: 1.1em;">{color} {category}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Trend Chart
    st.subheader("📈 3-Day AQI Forecast Trend")
    
    fig = go.Figure()
    
    # Main forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['daily_avg_aqi'],
        mode='lines+markers',
        name='Daily Average AQI',
        line=dict(color='#00d4ff', width=3),
        marker=dict(size=12, color='#00d4ff', line=dict(color='white', width=2)),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.2)',
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>AQI: %{y:.1f}<extra></extra>'
    ))
    
    # Add grand average line
    fig.add_hline(
        y=grand_avg, 
        line_dash="dash", 
        line_color="yellow", 
        line_width=2,
        annotation_text=f"3-Day Avg: {round(grand_avg, 1)}",
        annotation_position="right"
    )
    
    # Add AQI category zones
    fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Good", annotation_position="left")
    fig.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.1, line_width=0, annotation_text="Moderate", annotation_position="left")
    fig.add_hrect(y0=100, y1=150, fillcolor="orange", opacity=0.1, line_width=0, annotation_text="Unhealthy SG", annotation_position="left")
    fig.add_hrect(y0=150, y1=200, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Unhealthy", annotation_position="left")
    
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="AQI Value",
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model Performance Table
    st.subheader("🤖 Model Performance Comparison")
    
    if model_info and 'test_rmse' in model_info and 'winner_rmse' in model_info:
        # Create model comparison table
        model_data = {
            "Model": ["Random Forest", "XGBoost", "SVR", "🏆 Ensemble (Voting)"],
            "RMSE": ["N/A", "N/A", "N/A", round(float(model_info['test_rmse']), 4) if model_info['test_rmse'] != "N/A" else "N/A"],
            "Status": ["Component", "Component", "Component", "✅ Active"]
        }
        
        # Try to extract winner info
        if 'description' in model_info and "Winner:" in model_info['description']:
            winner = model_info['description'].replace("Winner: ", "").strip()
            winner_rmse = round(float(model_info['winner_rmse']), 4) if model_info['winner_rmse'] != "N/A" else "N/A"
            
            # Update the winner's RMSE
            for i, name in enumerate(model_data["Model"]):
                if winner.lower() in name.lower():
                    model_data["RMSE"][i] = f"🏆 {winner_rmse}"
                    model_data["Status"][i] = "🏆 Winner"
        
        model_df = pd.DataFrame(model_data)
        
        st.dataframe(
            model_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Model": st.column_config.TextColumn("Model Name", width="medium"),
                "RMSE": st.column_config.TextColumn("Test RMSE", width="small"),
                "Status": st.column_config.TextColumn("Status", width="small")
            }
        )
    else:
        st.info("Model performance metrics will be displayed here once available from the model registry.")
    
    # Data Table
    with st.expander("📋 View Detailed Forecast Data"):
        display_df = forecast_df.copy()
        display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
        display_df = display_df.rename(columns={
            'date': 'Date',
            'daily_avg_aqi': 'Daily Avg AQI',
            'grand_avg_aqi': '3-Day Avg AQI',
            'forecast_type': 'Model Type'
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Model Information Details
    if model_info:
        with st.expander("ℹ️ Model Registry Information"):
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"""
**Model Name:** {model_info.get('name', 'N/A')}  
**Version:** {model_info.get('version', 'N/A')}  
**Type:** {model_info.get('description', 'Ensemble Model')}
                """)
            with col2:
                st.info(f"""
**Ensemble Test RMSE:** {model_info.get('test_rmse', 'N/A')}  
**Winner Model RMSE:** {model_info.get('winner_rmse', 'N/A')}  
**Status:** ✅ Active & Deployed
                """)

else:
    st.error("⚠️ Forecast data not available.")
    st.info("""
    **Troubleshooting:**
    - Check if `data/forecast_data.csv` exists in your repository
    - Verify GitHub Actions workflow has run successfully
    - Check Hopsworks connection if accessing directly
    - Review logs for any errors
    """)

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
**Karachi AQI Sentinel** monitors air quality in Karachi, Pakistan using machine learning.

- **Data Source:** Hopsworks Feature Store
- **Update Frequency:** Hourly (via GitHub Actions)
- **Forecast Horizon:** 3 days
- **Model:** Ensemble (RF + XGBoost + SVR)
- **Prediction Method:** Voting Regressor
""")

if daily_summary_df is not None:
    st.sidebar.write("### 📊 Data Info")
    st.sidebar.write(f"**Total Records:** {len(daily_summary_df)}")
    st.sidebar.write(f"**Forecast Records:** {len(forecast_df)}")
    st.sidebar.write(f"**Date Range:** {daily_summary_df['date'].min().strftime('%Y-%m-%d')} to {daily_summary_df['date'].max().strftime('%Y-%m-%d')}")
    st.sidebar.write(f"**Grand Avg AQI:** {round(grand_avg, 1)}")

st.sidebar.write("---")
st.sidebar.caption("🔄 Data synced via GitHub Actions")
st.sidebar.caption(f"📅 Current Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
