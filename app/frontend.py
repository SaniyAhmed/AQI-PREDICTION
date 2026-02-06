import streamlit as st
import pandas as pd
import plotly.express as px
import time
import hopsworks
import joblib
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Karachi AQI Sentinel", layout="wide", page_icon="🌬️")

# --- CUSTOM CSS ---
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
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-title">🌬️ Karachi AQI Sentinel</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced Environmental Monitoring & AI Forecasting</p>', unsafe_allow_html=True)

# --- UPDATED STABLE DATA FETCHING ---
@st.cache_data(ttl=3600)
def fetch_hopsworks_data():
    try:
        # Pull key from Streamlit Secrets
        api_key = st.secrets["MY_HOPSWORK_KEY"]
        project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store()
        mr = project.get_model_registry()

        # 1. Get Leaderboard & Champion info
        model_types = ["karachi_aqi_randomforest", "karachi_aqi_xgboost", "karachi_aqi_svr"]
        leaderboard = []
        best_model_obj = None
        lowest_rmse = float('inf')

        for m_name in model_types:
            try:
                versions = mr.get_models(m_name)
                if versions:
                    versions.sort(key=lambda x: x.version, reverse=True)
                    latest = versions[0]
                    # Try to find metrics in different places
                    metrics = getattr(latest, "training_metrics", getattr(latest, "metrics", {}))
                    curr_rmse = float(metrics.get('test_rmse', 999.0))
                    
                    leaderboard.append({
                        "Model": m_name.replace("karachi_aqi_", "").title(),
                        "RMSE": round(curr_rmse, 4),
                        "RawName": latest.name
                    })
                    
                    if curr_rmse < lowest_rmse:
                        lowest_rmse = curr_rmse
                        best_model_obj = latest
            except:
                continue

        # Mark Champion
        for item in leaderboard:
            item["Status"] = "Champion" if best_model_obj and item["RawName"] == best_model_obj.name else "Challenger"

        # 2. Get Forecast Data (Using the most stable "read" method)
        fg = fs.get_feature_group("karachi_aqi_forecast", version=1)
        
        # We try to read using the REST API which is safer for Cloud
        try:
            df = fg.read(read_options={"use_hive": False})
        except:
            # Fallback for severe connection issues
            st.warning("🔄 High-speed query failed. Switching to direct fetch...")
            df = fg.select_all().read(read_options={"use_hive": False})

        df['prediction_timestamp'] = pd.to_datetime(df['prediction_timestamp'])
        df = df.sort_values('prediction_timestamp')

        return df, best_model_obj, leaderboard

    except Exception as e:
        st.error(f"❌ Connection Error: {e}")
        return None, None, []

# --- EXECUTE DATA FETCH ---
df, best_model_obj, leaderboard = fetch_hopsworks_data()

if df is not None and best_model_obj is not None:
    # Prepare metrics
    winner = best_model_obj.name.replace("karachi_aqi_", "").replace("_", " ").title()
    version = best_model_obj.version
    m_metrics = getattr(best_model_obj, "training_metrics", getattr(best_model_obj, "metrics", {}))
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

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1684/1684375.png", width=150)
st.sidebar.title("Sentinel Controls")
if st.sidebar.button("♻️ Force Registry Resync"):
    st.cache_data.clear()
    st.rerun()
