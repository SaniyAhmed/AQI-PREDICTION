import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='hopsworks')
os.environ["HSFS_DISABLE_FLIGHT_CLIENT"] = "True"
os.environ["HSFS_DISABLE_HIVE_CLIENT"] = "True"

import hopsworks

st.set_page_config(page_title="Karachi AQI Sentinel", layout="wide", page_icon="🌬️")

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Rajdhani', sans-serif !important; letter-spacing: 0.05em; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1b2a 0%, #112240 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 4px 20px rgba(0,212,255,0.08);
}
div[data-testid="metric-container"] label {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase;
    color: #7ca9d4 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    color: #00d4ff !important;
}
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e3a5f, transparent);
    margin: 2rem 0;
}
.insight-box {
    background: linear-gradient(135deg, #0a1628 0%, #0d2040 100%);
    border: 1px solid #1e3a5f;
    border-left: 4px solid #00d4ff;
    border-radius: 10px;
    padding: 20px 24px;
    margin-top: 8px;
    font-family: 'Inter', sans-serif;
    line-height: 1.75;
    color: #c8dff0;
    font-size: 0.95rem;
}
.insight-box strong { color: #00d4ff; }
.insight-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #00d4ff;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_float(val):
    """Convert a registry metric value to float, or return None."""
    try:
        f = float(val)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def get_winner_from_rmse(model_info: dict):
    """
    Compare individual model RMSEs and return the winner based on the winner_rmse value.
    Matches the name of the individual model that has the same RMSE as 'winner_rmse'.
    """
    # Keys matching your Hopsworks Screenshot exactly
    candidates = {
        "RandomForest": safe_float(model_info.get("randomforest_rmse")),
        "XGBoost":      safe_float(model_info.get("xgboost_rmse")),
        "SVR":          safe_float(model_info.get("svr_rmse")),
    }
    
    winner_val = safe_float(model_info.get("winner_rmse"))
    valid = {k: v for k, v in candidates.items() if v is not None}

    best_name = "Pending"
    if winner_val is not None:
        # Find which model matches the winner_rmse
        for name, rmse in valid.items():
            if abs(rmse - winner_val) < 1e-5:
                best_name = name
                break
    
    return best_name, (round(winner_val, 4) if winner_val else None), {k: round(v, 4) for k, v in valid.items()}


def aqi_category(val):
    for threshold, cat, icon, bg, accent in [
        (50,  "Good",            "🟢", "#1a3a1a", "#4caf50"),
        (100, "Moderate",        "🟡", "#3a3010", "#ffc107"),
        (150, "Unhealthy (SG)", "🟠", "#3a2010", "#ff9800"),
        (200, "Unhealthy",       "🔴", "#3a1010", "#f44336"),
        (999, "Hazardous",       "🟣", "#2a0a2a", "#9c27b0"),
    ]:
        if val <= threshold:
            return cat, icon, bg, accent
    return "Hazardous", "🟣", "#2a0a2a", "#9c27b0"


# ── Data Fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_all_data():
    daily_summary_df = None
    current_aqi      = None
    model_info       = {}

    local_file = "data/forecast_data.csv"
    if os.path.exists(local_file):
        try:
            daily_summary_df = pd.read_csv(local_file)
            daily_summary_df['date'] = pd.to_datetime(daily_summary_df['date'])
            daily_summary_df = daily_summary_df.sort_values('date', ascending=True)
        except Exception as e:
            st.error(f"Error reading local CSV: {e}")

    try:
        api_key = st.secrets.get("MY_HOPSWORK_KEY") or os.getenv("MY_HOPSWORK_KEY")
        if api_key:
            project = hopsworks.login(api_key_value=api_key)
            fs = project.get_feature_store()
            mr = project.get_model_registry()

            if daily_summary_df is None or daily_summary_df.empty:
                try:
                    fg_summary = fs.get_feature_group(name="karachi_aqi_daily_summary", version=2)
                    daily_summary_df = fg_summary.read()
                    daily_summary_df['date'] = pd.to_datetime(daily_summary_df['date'])
                    daily_summary_df = daily_summary_df.sort_values('date', ascending=True)
                except Exception as e:
                    st.warning(f"Could not fetch daily summary: {e}")

            try:
                fg_historical = fs.get_feature_group(name="karachi_aqi", version=4)
                historical_df = fg_historical.read().sort_values(['year', 'month', 'day', 'hour'])
                if not historical_df.empty:
                    current_aqi = float(historical_df.iloc[-1]['aqi'])
            except Exception:
                pass

            try:
                models = mr.get_models("karachi_aqi_model")
                if models:
                    latest = models[0]
                    m = latest.training_metrics
                    # THESE KEYS MATCH YOUR SCREENSHOT EXACTLY
                    model_info = {
                        "name":           latest.name,
                        "version":        latest.version,
                        "ensemble_rmse":  m.get("test_rmse", "N/A"),
                        "winner_rmse":    m.get("winner_rmse", "N/A"),
                        "randomforest_rmse": m.get("randomforest_rmse", "N/A"),
                        "xgboost_rmse":      m.get("xgboost_rmse", "N/A"),
                        "svr_rmse":          m.get("svr_rmse", "N/A"),
                        "description":    latest.description,
                    }
            except Exception:
                pass

            hopsworks.logout()
    except Exception as e:
        st.sidebar.warning(f"Hopsworks connection not available: {e}")

    return daily_summary_df, current_aqi, model_info


daily_summary_df, current_aqi, model_info = load_all_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:4px;">
  <span style="font-size:2.6rem;">🌬️</span>
  <div>
    <h1 style="margin:0;font-size:2.4rem;font-family:'Rajdhani',sans-serif;
               color:#00d4ff;letter-spacing:0.06em;">KARACHI AQI SENTINEL</h1>
    <p style="margin:0;color:#7ca9d4;font-size:0.9rem;letter-spacing:0.08em;text-transform:uppercase;">
      Real-time Air Quality Monitoring &amp; 3-Day ML Forecast
    </p>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
if daily_summary_df is not None and not daily_summary_df.empty:

    grand_avg = (
        daily_summary_df['grand_avg_aqi'].iloc[-1]
        if 'grand_avg_aqi' in daily_summary_df.columns
        else daily_summary_df['daily_avg_aqi'].mean()
    )

    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    dates_tz_naive = (
        daily_summary_df['date'].dt.tz_localize(None)
        if daily_summary_df['date'].dt.tz is not None
        else daily_summary_df['date']
    )
    forecast_df = daily_summary_df[dates_tz_naive >= today].copy()
    if forecast_df.empty:
        forecast_df = daily_summary_df.copy()

    winner_name, winner_rmse, all_rmse = get_winner_from_rmse(model_info)

    # ── KPI Row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        val = round(current_aqi, 1) if current_aqi else round(forecast_df.iloc[0]['daily_avg_aqi'], 1)
        st.metric("Latest AQI", str(val))

    with k2:
        st.metric("3-Day Average", str(round(grand_avg, 1)))

    with k3:
        st.metric("Winner RMSE", str(winner_rmse) if winner_rmse else "Pending")

    with k4:
        st.metric("Top Model", winner_name)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Daily Forecast Cards ──────────────────────────────────────────────────
    st.markdown('<h2 style="font-family:Rajdhani,sans-serif;color:#e2f0ff;">📊 DAILY FORECAST BREAKDOWN</h2>', unsafe_allow_html=True)
    cols = st.columns(min(len(forecast_df), 4))
    for idx, (_, row) in enumerate(forecast_df.iterrows()):
        if idx >= 4: break
        aqi_val  = round(row['daily_avg_aqi'], 1)
        date_str = pd.to_datetime(row['date']).strftime('%a, %b %d')
        cat, icon, bg, accent = aqi_category(aqi_val)
        with cols[idx]:
            st.markdown(f"""
            <div style="background:linear-gradient(160deg,{bg} 0%,#0d1b2a 100%);
                        border:1px solid {accent}55;border-top:3px solid {accent};
                        border-radius:14px;padding:22px 16px;text-align:center;">
              <p style="margin:0;font-size:0.75rem;color:{accent};font-weight:600;">{date_str}</p>
              <p style="margin:8px 0;font-size:3rem;font-weight:700;color:#ffffff;">{aqi_val}</p>
              <p style="margin:0;font-size:0.85rem;color:#b0c8e0;">{icon} {cat}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Trend Chart ───────────────────────────────────────────────────────────
    st.markdown('<h2 style="font-family:Rajdhani,sans-serif;color:#e2f0ff;">📈 3-DAY AQI FORECAST TREND</h2>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['daily_avg_aqi'], mode='lines+markers', line=dict(color='#00d4ff', width=3)))
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
    st.plotly_chart(fig, use_container_width=True)

    # ── Model Comparison Table ────────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<h2 style="font-family:Rajdhani,sans-serif;color:#e2f0ff;">🤖 MODEL PERFORMANCE COMPARISON</h2>', unsafe_allow_html=True)

    MODELS_META = [
        ("RandomForest", "🌲", "randomforest_rmse",  "#4caf50"),
        ("XGBoost",      "⚡", "xgboost_rmse", "#00d4ff"),
        ("SVR",          "📐", "svr_rmse",  "#ff9800"),
    ]

    table_rows = []
    for mname, icon, rmse_key, accent in MODELS_META:
        raw_val = model_info.get(rmse_key, "N/A")
        rmse_f = safe_float(raw_val)
        rmse_str = f"{rmse_f:.4f}" if rmse_f else "Pending"
        is_winner = (mname == winner_name)
        status = '🏆 WINNER' if is_winner else '—'
        bg = "rgba(0,212,255,0.07)" if is_winner else "transparent"
        
        table_rows.append(f"""
        <tr style="background:{bg};border-bottom:1px solid #1e3a5f;">
          <td style="padding:15px;color:{accent};">{icon} {mname}</td>
          <td style="padding:15px;text-align:center;">{rmse_str}</td>
          <td style="padding:15px;text-align:center;">{status}</td>
        </tr>""")

    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;border:1px solid #1e3a5f;">
      <tr style="background:#112240;color:#7ca9d4;text-transform:uppercase;font-size:0.8rem;">
        <th style="padding:10px;text-align:left;">Model</th>
        <th style="padding:10px;">Test RMSE</th>
        <th style="padding:10px;">Status</th>
      </tr>
      {"".join(table_rows)}
    </table>
    """, unsafe_allow_html=True)

else:
    st.error("⚠️ Forecast data not available.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("ℹ️ About")
st.sidebar.info("Karachi AQI Sentinel monitors air quality using machine learning.")
