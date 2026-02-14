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
    Compare the three individual model RMSEs stored in model_info and return
    (winner_name, winner_rmse_float, {name: rmse_float}).
    """
    candidates = {
        "RandomForest": safe_float(model_info.get("rf_rmse")),
        "XGBoost":      safe_float(model_info.get("xgb_rmse")),
        "SVR":          safe_float(model_info.get("svr_rmse")),
    }
    valid = {k: v for k, v in candidates.items() if v is not None}
    
    # 1. Try to find logic based on 'winner_rmse' match
    winner_rmse_val = safe_float(model_info.get("winner_rmse"))
    if winner_rmse_val is not None and valid:
        # Find which model has matching RMSE (with tolerance)
        for name, rmse in valid.items():
            if abs(rmse - winner_rmse_val) < 0.0001:
                return name, winner_rmse_val, {k: round(v, 4) for k, v in valid.items()}

    # 2. Fallback: Computed minimum from available individual RMSEs
    if valid:
        best = min(valid, key=valid.get)
        return best, round(valid[best], 4), {k: round(v, 4) for k, v in valid.items()}

    # 3. Last resort: registry "winner" string
    fallback_name = model_info.get("winner")
    if not fallback_name or fallback_name == "N/A":
        # Try parsing from description if available
        desc = model_info.get("description", "")
        if "Winner: " in desc:
            fallback_name = desc.split("Winner: ")[1].strip()
        else:
            fallback_name = "N/A"

    return fallback_name, winner_rmse_val, {}


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

# CRITICAL FIX: Cache key includes current hour to auto-refresh every hour
def get_cache_key():
    """Generate cache key that changes every hour to force refresh"""
    from datetime import datetime
    return datetime.now().strftime('%Y-%m-%d-%H')

@st.cache_data(ttl=60, show_spinner=False)
def load_all_data(_cache_buster):
    """_cache_buster parameter forces cache invalidation when changed"""
    daily_summary_df = None
    current_aqi      = None
    model_info       = {}

    # Try loading from local CSV first (GitHub Actions workflow saves this)
    local_file = "data/forecast_data.csv"
    if os.path.exists(local_file):
        try:
            daily_summary_df = pd.read_csv(local_file)
            daily_summary_df['date'] = pd.to_datetime(daily_summary_df['date'])
            daily_summary_df = daily_summary_df.sort_values('date', ascending=True)
            print(f"✅ Loaded {len(daily_summary_df)} records from local CSV")
        except Exception as e:
            st.error(f"Error reading local CSV: {e}")

    try:
        api_key = st.secrets.get("MY_HOPSWORK_KEY") or os.getenv("MY_HOPSWORK_KEY")
        if api_key:
            project = hopsworks.login(api_key_value=api_key)
            fs = project.get_feature_store()
            mr = project.get_model_registry()

            # Fetch daily summary
            try:
                fg_summary = fs.get_feature_group(name="karachi_aqi_daily_summary", version=2)
                daily_summary_df = fg_summary.read()
                daily_summary_df['date'] = pd.to_datetime(daily_summary_df['date'])
                daily_summary_df = daily_summary_df.sort_values('date', ascending=True)
                print(f"✅ Loaded {len(daily_summary_df)} forecast records")
            except Exception as e:
                st.warning(f"Could not fetch daily summary: {e}")

            # Fetch current AQI from historical data
            try:
                fg_historical = fs.get_feature_group(name="karachi_aqi", version=5)
                historical_df = fg_historical.read().sort_values(['year', 'month', 'day', 'hour'])
                if not historical_df.empty:
                    current_aqi = float(historical_df.iloc[-1]['aqi'])
                    print(f"✅ Current AQI: {current_aqi}")
            except Exception as e:
                print(f"⚠️ Could not fetch current AQI: {e}")

            # FIXED: Fetch model registry with better version handling
            try:
                models = mr.get_models("karachi_aqi_model")
                if models:
                    # Sort by version (highest first) to get the LATEST model
                    models_sorted = sorted(models, key=lambda x: int(x.version), reverse=True)
                    latest = models_sorted[0]
                    
                    print(f"📊 Loading model version {latest.version}")
                    m = latest.training_metrics
                    
                    # FIXED: Handle new metric keys from improved pipeline
                    model_info = {
                        "name":           latest.name,
                        "version":        latest.version,
                        "ensemble_rmse":  m.get("rmse", "N/A"),  # This is the ensemble RMSE
                        "winner_rmse":    m.get("winner_rmse", "N/A"),
                        "winner":         m.get("winner", "N/A"),
                        "rf_rmse":        m.get("randomforest_rmse", "N/A"),
                        "xgb_rmse":       m.get("xgboost_rmse", "N/A"),
                        "svr_rmse":       m.get("svr_rmse", "N/A"),
                        "description":    latest.description,
                    }
                    
                    print(f"📊 Model metrics: {model_info}")
                    
            except Exception as e:
                print(f"⚠️ Could not fetch model info: {e}")

            hopsworks.logout()
    except Exception as e:
        st.sidebar.warning(f"Hopsworks connection not available: {e}")

    return daily_summary_df, current_aqi, model_info


daily_summary_df, current_aqi, model_info = load_all_data(get_cache_key())

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

# CRITICAL: Show data load timestamp so user can verify freshness
if daily_summary_df is not None and not daily_summary_df.empty:
    latest_date = daily_summary_df['date'].max()
    st.caption(f"📊 Data loaded: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | Latest forecast: {latest_date.strftime('%Y-%m-%d')}")

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
        st.info("ℹ️ Showing all available data (past dates — for demo)")

    winner_name, winner_rmse, all_rmse = get_winner_from_rmse(model_info)

    # ── KPI Row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        val = round(current_aqi, 1) if current_aqi else round(forecast_df.iloc[0]['daily_avg_aqi'], 1)
        st.metric("Current AQI" if current_aqi else "Latest AQI", str(val),
                  help="Latest measured AQI value")

    with k2:
        st.metric("3-Day Average", str(round(grand_avg, 1)),
                  help="Average predicted AQI over next 3 days")

    with k3:
        # Show winner_rmse if available, otherwise ensemble rmse, otherwise "Pending"
        rmse_display = (
            str(winner_rmse)
            if winner_rmse is not None
            else (
                str(round(float(model_info.get("ensemble_rmse")), 4))
                if safe_float(model_info.get("ensemble_rmse")) is not None
                else "Pending"
            )
        )
        st.metric("Winner RMSE", rmse_display,
                  help="Lowest test RMSE among RandomForest / XGBoost / SVR")

    with k4:
        # Always the best individual model name — never "Ensemble"
        top_model_display = winner_name if winner_name not in ("N/A", None, "") else "Pending"
        st.metric("Top Model", top_model_display,
                  help="Individual model with lowest test RMSE")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Daily Forecast Cards ──────────────────────────────────────────────────
    st.markdown(
        '<h2 style="font-family:Rajdhani,sans-serif;color:#e2f0ff;letter-spacing:0.07em;">'
        '📊 DAILY FORECAST BREAKDOWN</h2>',
        unsafe_allow_html=True
    )
    cols = st.columns(min(len(forecast_df), 4))
    for idx, (_, row) in enumerate(forecast_df.iterrows()):
        if idx >= 4:
            break
        aqi_val  = round(row['daily_avg_aqi'], 1)
        date_str = pd.to_datetime(row['date']).strftime('%a, %b %d')
        cat, icon, bg, accent = aqi_category(aqi_val)
        with cols[idx]:
            st.markdown(f"""
            <div style="background:linear-gradient(160deg,{bg} 0%,#0d1b2a 100%);
                        border:1px solid {accent}55;border-top:3px solid {accent};
                        border-radius:14px;padding:22px 16px;text-align:center;
                        box-shadow:0 6px 24px {accent}22;">
              <p style="margin:0 0 4px;font-family:'Rajdhani',sans-serif;font-size:0.75rem;
                        letter-spacing:0.14em;text-transform:uppercase;color:{accent};font-weight:600;">
                {date_str}
              </p>
              <p style="margin:8px 0 4px;font-family:'Rajdhani',sans-serif;font-size:3rem;
                        font-weight:700;color:#ffffff;line-height:1;">{aqi_val}</p>
              <p style="margin:0;font-size:0.85rem;color:#b0c8e0;">{icon}&nbsp;&nbsp;{cat}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Trend Chart ───────────────────────────────────────────────────────────
    st.markdown(
        '<h2 style="font-family:Rajdhani,sans-serif;color:#e2f0ff;letter-spacing:0.07em;">'
        '📈 3-DAY AQI FORECAST TREND</h2>',
        unsafe_allow_html=True
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_df['date'], y=forecast_df['daily_avg_aqi'],
        mode='lines+markers', name='Daily Avg AQI',
        line=dict(color='#00d4ff', width=3),
        marker=dict(size=13, color='#00d4ff', line=dict(color='#0d1b2a', width=2.5)),
        fill='tozeroy', fillcolor='rgba(0,212,255,0.12)',
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>AQI: %{y:.1f}<extra></extra>'
    ))
    fig.add_hline(y=grand_avg, line_dash="dot", line_color="#ffd166", line_width=2,
                  annotation_text=f"3-Day Avg: {round(grand_avg, 1)}",
                  annotation_position="top right",
                  annotation_font_color="#ffd166")
    fig.add_hrect(y0=0,   y1=50,  fillcolor="#4caf50", opacity=0.07, line_width=0)
    fig.add_hrect(y0=50,  y1=100, fillcolor="#ffc107", opacity=0.07, line_width=0)
    fig.add_hrect(y0=100, y1=150, fillcolor="#ff9800", opacity=0.07, line_width=0)
    fig.add_hrect(y0=150, y1=200, fillcolor="#f44336", opacity=0.07, line_width=0)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Date", yaxis_title="AQI",
        hovermode='x unified', height=420,
        font=dict(family='Rajdhani, Inter, sans-serif'),
        showlegend=False,
        xaxis=dict(gridcolor='#1e3a5f', linecolor='#1e3a5f'),
        yaxis=dict(gridcolor='#1e3a5f', linecolor='#1e3a5f'),
        margin=dict(l=10, r=10, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


    # ── Model Comparison Table ────────────────────────────────────────────────
    st.markdown(
        '<h2 style="font-family:Rajdhani,sans-serif;color:#e2f0ff;letter-spacing:0.07em;">'
        '🤖 MODEL PERFORMANCE COMPARISON</h2>',
        unsafe_allow_html=True
    )

    MODELS_META = [
        ("RandomForest", "🌲", "rf_rmse",  "#4caf50"),
        ("XGBoost",      "⚡", "xgb_rmse", "#00d4ff"),
        ("SVR",          "📐", "svr_rmse",  "#ff9800"),
    ]

    # Build rows as plain Python — no f-string HTML soup
    table_rows = []
    
    # 1. First ensure we have the definitive winner RMSE from registry
    definitive_winner_rmse = safe_float(model_info.get("winner_rmse"))
    
    for mname, icon, rmse_key, accent in MODELS_META:
        raw_val  = model_info.get(rmse_key, "N/A") if model_info else "N/A"
        rmse_f   = safe_float(raw_val)
        rmse_str = f"{rmse_f:.4f}" if rmse_f is not None else "Pending"

        is_winner = False
        
        # Priority 1: Match the specific registry winner_rmse value
        if definitive_winner_rmse is not None and rmse_f is not None:
             if abs(rmse_f - definitive_winner_rmse) < 0.0001:
                 is_winner = True
        
        # Priority 2: Fallback to name match if RMSE approach failed/ambiguous
        if not is_winner and winner_name == mname:
             is_winner = True

        row_bg      = "rgba(0,212,255,0.07)" if is_winner else "rgba(255,255,255,0.02)"
        status_cell = (
            '<span style="background:#00d4ff22;color:#00d4ff;border:1px solid #00d4ff55;'
            'border-radius:6px;padding:2px 10px;font-size:0.72rem;letter-spacing:0.1em;'
            'font-weight:700;">🏆 WINNER</span>'
            if is_winner else
            '<span style="color:#4a6a8a;font-size:0.85rem;">—</span>'
        )

        row_html = (
            f'<tr style="background:{row_bg};border-bottom:1px solid #1e3a5f44;">'
            f'<td style="padding:14px 18px;font-family:Rajdhani,sans-serif;font-size:1.05rem;color:{accent};font-weight:600;letter-spacing:0.04em;">{icon}&nbsp; {mname}</td>'
            f'<td style="padding:14px 18px;font-family:Rajdhani,sans-serif;font-size:1.05rem;color:#c8dff0;text-align:center;">{rmse_str}</td>'
            f'<td style="padding:14px 18px;text-align:center;">{status_cell}</td>'
            f'</tr>'
        )
        table_rows.append(row_html)

    rows_joined = "".join(table_rows)

    st.markdown(f"""
<div style="border-radius:14px;overflow:hidden;border:1px solid #1e3a5f;
            box-shadow:0 4px 24px rgba(0,0,0,0.4);margin-bottom:8px;">
  <table style="width:100%;border-collapse:collapse;">
    <thead>
      <tr style="background:linear-gradient(90deg,#0d2040,#112240);
                 border-bottom:2px solid #1e3a5f;">
        <th style="padding:12px 18px;text-align:left;font-family:Rajdhani,sans-serif;
                   letter-spacing:0.12em;text-transform:uppercase;color:#7ca9d4;font-size:0.78rem;">
          Model
        </th>
        <th style="padding:12px 18px;text-align:center;font-family:Rajdhani,sans-serif;
                   letter-spacing:0.12em;text-transform:uppercase;color:#7ca9d4;font-size:0.78rem;">
          Test RMSE ↓
        </th>
        <th style="padding:12px 18px;text-align:center;font-family:Rajdhani,sans-serif;
                   letter-spacing:0.12em;text-transform:uppercase;color:#7ca9d4;font-size:0.78rem;">
          Status
        </th>
      </tr>
    </thead>
    <tbody>
      {rows_joined}
    </tbody>
  </table>
</div>
<p style="font-size:0.78rem;color:#4a6a8a;margin-top:6px;">
  ↓ Lower RMSE = better accuracy. Winner feeds into the Voting Ensemble at 2× weight.
</p>
""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Map + Insight ─────────────────────────────────────────────────────────
    col_map, col_insight = st.columns([1.05, 1])

    with col_map:
        st.markdown(
            '<h2 style="font-family:Rajdhani,sans-serif;color:#e2f0ff;letter-spacing:0.07em;">'
            '📍 MONITORING STATION</h2>',
            unsafe_allow_html=True
        )
        STATION_LAT, STATION_LON = 24.8607, 67.0011
        map_fig = go.Figure(go.Scattermapbox(
            lat=[STATION_LAT], lon=[STATION_LON],
            mode="markers+text",
            marker=dict(size=18, color="#00d4ff", opacity=0.95),
            text=["📡 Karachi Central"],
            textposition="top right",
            textfont=dict(color="#ffffff", size=13, family="Rajdhani, sans-serif"),
            hovertemplate=(
                "<b>Karachi AQI Station</b><br>"
                "Lat: 24.8607°N &nbsp; Lon: 67.0011°E<extra></extra>"
            )
        ))
        map_fig.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                center=dict(lat=STATION_LAT, lon=STATION_LON),
                zoom=10.5,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=330,
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(map_fig, use_container_width=True)

    with col_insight:
        st.markdown(
            '<h2 style="font-family:Rajdhani,sans-serif;color:#e2f0ff;letter-spacing:0.07em;">'
            '🔬 AQI ANALYSIS INSIGHT</h2>',
            unsafe_allow_html=True
        )
        avg_aqi = round(grand_avg, 1)

        if avg_aqi > 150:
            driver_line = (
                "<strong>PM2.5 is the dominant pollutant</strong> — fine particulate matter from "
                "vehicle exhaust and industrial emissions is well above safe thresholds, posing a "
                "direct health risk especially for sensitive groups."
            )
        elif avg_aqi > 100:
            driver_line = (
                "<strong>PM2.5 and PM10 are the primary contributors.</strong> Coarse dust particles "
                "(PM10) from construction and road traffic are compounding the fine-particle (PM2.5) "
                "load typical of Karachi's dense urban core."
            )
        elif avg_aqi > 50:
            driver_line = (
                "<strong>PM10 (coarse dust) is the leading driver</strong> at this level. "
                "Ground-level dust from dry roads, construction sites, and coastal wind patterns "
                "is keeping air quality in the moderate range."
            )
        else:
            driver_line = (
                "<strong>PM2.5 and PM10 are within safe limits.</strong> "
                "Clean sea breezes from the Arabian Sea are helping disperse pollutants, "
                "resulting in good air quality across the city."
            )

        wind_line = (
            "<strong>Wind speed is a key dispersal factor</strong> — speeds above 15 km/h "
            "flush pollutants away from the surface, while calm nights allow particulates "
            "to accumulate close to ground level."
        )

        if avg_aqi > 150:
            health_line = "⚠️ <strong>Health advisory:</strong> Sensitive groups (children, elderly, respiratory patients) should minimise outdoor activity."
        elif avg_aqi > 100:
            health_line = "🟡 <strong>Moderate caution:</strong> Unusually sensitive individuals may experience minor discomfort outdoors."
        else:
            health_line = "✅ <strong>Air quality is acceptable</strong> for most residents. Continue monitoring for changes."

        st.markdown(f"""
        <div class="insight-box">
          <div class="insight-title">📊 Forecast Summary · {avg_aqi} AQI (3-day avg)</div>
          <p>{driver_line}</p>
          <p>{wind_line}</p>
          <p>{health_line}</p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.error("⚠️ Forecast data not available.")
    st.info("""
    **Troubleshooting:**
    - Verify GitHub Actions workflow has run successfully
    - Check Hopsworks connection
    - Review logs for errors
    - Try clearing Streamlit cache (press 'C' in the app)
    """)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
**Karachi AQI Sentinel** monitors air quality in Karachi, Pakistan using machine learning.

- **Data Source:** Hopsworks Feature Store
- **Update Frequency:** Hourly (via GitHub Actions)
- **Forecast Horizon:** 3 days
- **Model:** Voting Ensemble (RF + XGBoost + SVR)
""")

if daily_summary_df is not None:
    st.sidebar.write("### 📊 Data Info")
    st.sidebar.write(f"**Total Records:** {len(daily_summary_df)}")
    try:
        st.sidebar.write(f"**Forecast Records:** {len(forecast_df)}")
        st.sidebar.write(
            f"**Date Range:** {daily_summary_df['date'].min().strftime('%Y-%m-%d')} "
            f"to {daily_summary_df['date'].max().strftime('%Y-%m-%d')}"
        )
        st.sidebar.write(f"**Grand Avg AQI:** {round(grand_avg, 1)}")
        # FIXED: Add model version info
        if model_info:
            st.sidebar.write(f"**Model Version:** {model_info.get('version', 'N/A')}")
    except NameError:
        pass

st.sidebar.write("---")

# FIXED: Add cache clear button
if st.sidebar.button("🔄 Clear Cache & Reload"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption("🔄 Data synced via GitHub Actions")
st.sidebar.caption(f"📅 {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
