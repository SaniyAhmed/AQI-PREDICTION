import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
import warnings

# Suppress incompatibility warnings
warnings.filterwarnings('ignore', category=UserWarning, module='hopsworks')

# ===== CRITICAL: SET ENVIRONMENT VARIABLES BEFORE ANY HOPSWORKS IMPORT =====
os.environ["HSFS_DISABLE_FLIGHT_CLIENT"] = "True"
os.environ["HSFS_DISABLE_HIVE_CLIENT"] = "True"
os.environ["DISABLE_INSECURE_GRPC"] = "True"

import hopsworks
import joblib

# --- PAGE CONFIG ---
st.set_page_config(page_title="Karachi AQI Sentinel", layout="wide", page_icon="🌬️")

# --- CUSTOM CSS (STRICTLY PRESERVED) ---
st.markdown("""
    <style>
    .block-container {
        max-width: 98% !important;
        padding-top: 3rem !important; 
        padding-bottom: 0rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    .main-title { 
        font-size: 95px !important; 
        font-weight: 900; 
        color: #ffffff; 
        line-height: 1.1; 
        margin-top: 20px !important; 
        margin-bottom: 0px; 
    }
    .sub-title { font-size: 42px !important; color: #00d4ff; margin-bottom: 40px; }
    div[data-testid="stMetricValue"] { font-size: 90px !important; font-weight: bold; color: #00d4ff !important; }
    div[data-testid="stMetricLabel"] { font-size: 34px !important; color: #ffffff !important; font-weight: 600 !important; }
    .stMetric { background-color: #1E1E1E; padding: 40px !important; border-radius: 20px; border: 3px solid #444; }
    .big-header { font-size: 65px !important; font-weight: 800 !important; color: #00d4ff !important; margin-bottom: 30px !important; display: block; }
    .medium-header { font-size: 48px !important; font-weight: 700 !important; color: #ffffff !important; margin-bottom: 25px !important; display: block; }
    .logic-text { 
        font-size: 34px !important; 
        line-height: 1.6 !important; 
        color: #f0f0f0 !important; 
        display: block;
        background: rgba(0, 212, 255, 0.05);
        padding: 40px;
        border-radius: 20px;
        border-left: 15px solid #00d4ff;
    }
    .giant-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 32px !important; 
        color: white;
        margin-top: 20px;
        background-color: #1E1E1E;
        border-radius: 15px;
        overflow: hidden;
    }
    .giant-table th {
        background-color: #00d4ff;
        color: #000;
        padding: 25px;
        text-align: left;
        font-weight: 800;
    }
    .giant-table td {
        padding: 25px;
        border-bottom: 1px solid #444;
    }
    .champion-row {
        background-color: rgba(0, 255, 204, 0.15);
        color: #00ffcc;
        font-weight: bold;
    }
    .registry-path { 
        font-size: 32px !important; 
        font-family: 'Courier New', monospace; 
        color: #00ffcc !important; 
        background: #111; 
        padding: 25px; 
        border-radius: 10px; 
        border: 2px dashed #00ffcc;
        margin-top: 30px;
        text-align: center;
    }
    .stPlotlyChart { height: 700px !important; }
    .error-box {
        background: rgba(255, 100, 100, 0.1);
        border-left: 5px solid #ff6464;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .success-box {
        background: rgba(0, 255, 0, 0.1);
        border-left: 5px solid #00ff00;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-title">🌬️ Karachi AQI Sentinel</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced Environmental Monitoring & AI Forecasting</p>', unsafe_allow_html=True)

# --- ULTRA-DIRECT DATA FETCHING FUNCTION ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_hopsworks_data():
    """
    Direct Hopsworks access bypassing Query Service completely
    """
    with st.spinner("🔄 Connecting to Hopsworks Feature Store..."):
        try:
            api_key = st.secrets["MY_HOPSWORK_KEY"]
            project = hopsworks.login(api_key_value=api_key)
            fs = project.get_feature_store()
            mr = project.get_model_registry()
            
            st.success("✅ Successfully connected to Hopsworks")
            
        except Exception as e:
            st.error(f"❌ Failed to connect to Hopsworks: {str(e)}")
            return None, None, []

    # === STEP 1: FETCH MODEL LEADERBOARD ===
    with st.spinner("🏆 Loading model registry..."):
        model_types = ["karachi_aqi_randomforest", "karachi_aqi_xgboost", "karachi_aqi_svr"]
        leaderboard = []
        best_model_obj = None
        lowest_rmse = float('inf')

        for m_name in model_types:
            try:
                models = mr.get_models(m_name)
                if models and len(models) > 0:
                    latest = models[0]
                    
                    # Try different ways to access metrics
                    metrics = {}
                    if hasattr(latest, 'training_metrics') and latest.training_metrics:
                        metrics = latest.training_metrics
                    elif hasattr(latest, 'metrics') and latest.metrics:
                        metrics = latest.metrics
                    
                    curr_rmse = float(metrics.get('test_rmse', 999.0))
                    
                    leaderboard.append({
                        "Model": m_name.replace("karachi_aqi_", "").replace("_", " ").title(),
                        "RMSE": round(curr_rmse, 4),
                        "RawName": latest.name
                    })
                    
                    if curr_rmse < lowest_rmse:
                        lowest_rmse = curr_rmse
                        best_model_obj = latest
            except Exception as e:
                st.warning(f"⚠️ Could not load model {m_name}: {str(e)[:100]}")
                continue

        # Mark champion
        for item in leaderboard:
            item["Status"] = "Champion" if best_model_obj and item["RawName"] == best_model_obj.name else "Challenger"

    # === STEP 2: DATA LOADING (File + Hopsworks Fallback) ===
    with st.spinner("📊 Loading forecast data..."):
        df = None
        
        # === PRIORITY 1: Read from synced CSV file (updated hourly by GitHub Action) ===
        local_file = "data/forecast_data.csv"
        
        if os.path.exists(local_file):
            try:
                st.info(f"📂 Reading from synced data file...")
                df = pd.read_csv(local_file)
                
                if df is not None and not df.empty:
                    st.success(f"✅ Loaded {len(df)} records from auto-synced file!")
                    st.info("💡 Data auto-updates hourly via GitHub Action")
                else:
                    raise ValueError("Empty file")
                    
            except Exception as file_error:
                st.warning(f"⚠️ File read failed: {str(file_error)[:100]}")
                df = None
        
        # === PRIORITY 2: Try direct Hopsworks access (will likely fail) ===
        if df is None or df.empty:
            st.info("🔄 No local file found - attempting Hopsworks direct access...")
            
            try:
                # Try Feature View first
                try:
                    fv = fs.get_feature_view(name="karachi_aqi_view", version=1)
                    df = fv.get_batch_data()
                    st.success(f"✅ Loaded {len(df)} records from Feature View!")
                except:
                    # Fallback to Feature Group
                    fg = fs.get_feature_group(name="karachi_aqi_forecast", version=1)
                    df = fg.read()
                    st.success(f"✅ Loaded {len(df)} records from Feature Group!")
                    
            except Exception as hops_error:
                st.error(f"❌ Hopsworks Query Service blocking access: {str(hops_error)[:150]}")
                st.error("🚫 **Setup Required**: Enable GitHub Action for hourly data sync")
                
                with st.expander("📋 Setup Instructions"):
                    st.markdown("""
                    ### Enable Automated Data Sync:
                    
                    1. **Add GitHub Secret**: Go to repo Settings → Secrets → Add `MY_HOPSWORK_KEY`
                    2. **Enable Workflow**: The `.github/workflows/sync_hopsworks_data.yml` will auto-run hourly
                    3. **Manual Trigger**: Go to Actions tab → "Sync Hopsworks Data" → Run workflow
                    4. **Wait**: First run creates `data/forecast_data.csv`
                    5. **Refresh**: Dashboard will load data automatically
                    
                    This bypasses Query Service incompatibility completely!
                    """)
                
                return None, best_model_obj, leaderboard

        # Process the dataframe if we got it
        if df is not None and not df.empty:
            try:
                # Ensure timestamp column exists
                if 'prediction_timestamp' in df.columns:
                    df['prediction_timestamp'] = pd.to_datetime(df['prediction_timestamp'])
                    df = df.sort_values('prediction_timestamp').reset_index(drop=True)
                
                # Ensure AQI column exists
                if 'predicted_aqi' not in df.columns:
                    if 'aqi' in df.columns:
                        df['predicted_aqi'] = df['aqi']
                    elif 'pm25' in df.columns:
                        # If only PM2.5 exists, we can estimate AQI
                        df['predicted_aqi'] = df['pm25']  # Simplified
                    else:
                        st.error("❌ Cannot find AQI or PM2.5 columns in data")
                        return None, best_model_obj, leaderboard
                    
                st.success(f"✅ Successfully loaded {len(df)} forecast records")
                
            except Exception as e:
                st.error(f"❌ Error processing data: {str(e)}")
                return None, best_model_obj, leaderboard
        else:
            st.error("❌ No data retrieved from any method")
            return None, best_model_obj, leaderboard

        return df, best_model_obj, leaderboard

# --- EXECUTE DATA FETCH ---
df, best_model_obj, leaderboard = fetch_hopsworks_data()

# === DISPLAY DASHBOARD ===
if df is not None and not df.empty and best_model_obj is not None:
    try:
        winner = best_model_obj.name.replace("karachi_aqi_", "").replace("_", " ").title()
        version = best_model_obj.version
        
        # Get metrics safely
        m_metrics = {}
        if hasattr(best_model_obj, 'training_metrics') and best_model_obj.training_metrics:
            m_metrics = best_model_obj.training_metrics
        elif hasattr(best_model_obj, 'metrics') and best_model_obj.metrics:
            m_metrics = best_model_obj.metrics
            
        rmse = float(m_metrics.get('test_rmse', 0.0))
        
        current_aqi = round(df["predicted_aqi"].iloc[0], 1)
        avg_72h_aqi = round(df["predicted_aqi"].mean(), 1)

        # --- TOP KPI ROW ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current AQI", f"{current_aqi}")
        with col2:
            st.metric("72h Average", f"{avg_72h_aqi}")
        with col3:
            st.metric("Champion", f"{winner}")
        with col4:
            st.metric("Model RMSE", f"{rmse:.4f}")

        st.divider()

        # --- MAP & CHART ROW ---
        left_col, right_col = st.columns([1, 2], gap="large")
        with left_col:
            st.markdown('<span class="medium-header">📍 Monitoring Station</span>', unsafe_allow_html=True)
            map_data = pd.DataFrame({'lat': [24.8607], 'lon': [67.0011]})
            st.map(map_data, zoom=10, use_container_width=True)

        with right_col:
            st.markdown('<span class="medium-header">📈 72-Hour Forecast Trend</span>', unsafe_allow_html=True)
            fig = px.area(df, x="prediction_timestamp", y="predicted_aqi", markers=True)
            fig.update_layout(
                template="plotly_dark", height=700,
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(tickfont=dict(size=24), title_font=dict(size=26)),
                yaxis=dict(tickfont=dict(size=24), title_font=dict(size=26)),
                yaxis_title="AQI Value"
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- BOTTOM PART ---
        st.divider()
        logic_col, table_col = st.columns([1.2, 1], gap="large")

        with logic_col:
            st.markdown('<span class="big-header">💡 AI Architecture Logic</span>', unsafe_allow_html=True)
            clean_name = str(winner).lower().replace(" ", "").replace("_", "")
            
            if "forest" in clean_name or "random" in clean_name:
                st.markdown('<span class="medium-header">🌳 Random Forest Ensemble Logic</span>', unsafe_allow_html=True)
                st.markdown(f'<span class="logic-text">The <b>{winner}</b> architecture was selected for its superior ability to handle multi-variate weather data. By ensemble-averaging 100+ decision trees, it effectively filters out sensor noise caused by Karachi\'s dense urban traffic spikes.</span>', unsafe_allow_html=True)
            elif "xgboost" in clean_name or "xgb" in clean_name:
                st.markdown('<span class="medium-header">🚀 XGBoost Gradient Boosting</span>', unsafe_allow_html=True)
                st.markdown(f'<span class="logic-text">The <b>{winner}</b> model excels at identifying non-linear patterns. Its gradient boosting framework captures sudden industrial pollution surges better than standard regressors.</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="medium-header">⚙️ Optimized Champion Selection</span>', unsafe_allow_html=True)
                st.markdown(f'<span class="logic-text">The <b>{winner}</b> model is currently the most accurate available, having outperformed challengers during the automated tournament.</span>', unsafe_allow_html=True)

        with table_col:
            st.markdown('<span class="big-header">🏆 Tournament Result</span>', unsafe_allow_html=True)
            if leaderboard:
                table_html = '<table class="giant-table"><thead><tr><th>Model</th><th>RMSE</th><th>Status</th></tr></thead><tbody>'
                for row in sorted(leaderboard, key=lambda x: x['RMSE']):
                    status_class = 'class="champion-row"' if row['Status'] == 'Champion' else ''
                    table_html += f'<tr {status_class}><td>{row["Model"]}</td><td>{row["RMSE"]}</td><td>{row["Status"]}</td></tr>'
                table_html += '</tbody></table>'
                st.markdown(table_html, unsafe_allow_html=True)
                st.markdown(f'<div class="registry-path">Registry Path: v{version}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error displaying dashboard: {str(e)}")
        st.info("💡 Try refreshing the page or clicking 'Force Registry Resync' in the sidebar")

elif df is not None and not df.empty:
    # We have data but no model
    st.warning("⚠️ Forecast data loaded but models unavailable")
    st.info("Partial dashboard mode - displaying forecast data only")
    
    current_aqi = round(df["predicted_aqi"].iloc[0], 1)
    avg_72h_aqi = round(df["predicted_aqi"].mean(), 1)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current AQI", f"{current_aqi}")
    with col2:
        st.metric("72h Average", f"{avg_72h_aqi}")
    
    fig = px.area(df, x="prediction_timestamp", y="predicted_aqi", markers=True)
    fig.update_layout(template="plotly_dark", height=600)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("❌ Unable to load any data from Hopsworks")
    st.markdown("""
    ### Troubleshooting Steps:
    1. Check if your Hopsworks API key is correctly set in Streamlit secrets
    2. Verify that the feature group `karachi_aqi_forecast` exists
    3. Ensure your Hopsworks project is accessible
    4. Try clicking the 'Force Registry Resync' button in the sidebar
    """)

# === SIDEBAR ===
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1684/1684375.png", width=150)
st.sidebar.title("Sentinel Controls")

if st.sidebar.button("♻️ Force Registry Resync"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown("### System Status")
st.sidebar.info(f"🐍 Python {sys.version_info.major}.{sys.version_info.minor}")
try:
    import hopsworks
    st.sidebar.success(f"✅ Hopsworks {hopsworks.__version__}")
except:
    st.sidebar.warning("⚠️ Hopsworks version unknown")

st.sidebar.markdown("### Debug Info")
if st.sidebar.checkbox("Show Environment Variables"):
    st.sidebar.code(f"""
HSFS_DISABLE_FLIGHT_CLIENT: {os.environ.get('HSFS_DISABLE_FLIGHT_CLIENT', 'Not Set')}
HSFS_DISABLE_HIVE_CLIENT: {os.environ.get('HSFS_DISABLE_HIVE_CLIENT', 'Not Set')}
    """)
