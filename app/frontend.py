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

    # === STEP 2: REAL-TIME DATA FROM HOPSWORKS (REST API ONLY) ===
    with st.spinner("📊 Fetching real-time data from Hopsworks..."):
        df = None
        
        # Get feature group
        try:
            fg = fs.get_feature_group("karachi_aqi_forecast", version=1)
            st.info(f"✅ Feature Group found: {fg.name} v{fg.version}")
        except Exception as e:
            st.error(f"❌ Cannot access feature group: {str(e)}")
            return None, best_model_obj, leaderboard

        # === BYPASS ALL SDK METHODS - USE PURE REST API ===
        st.info("🔍 Bypassing SDK completely - using direct REST API calls...")
        
        try:
            import requests
            import json
            
            # Get connection details - compatible with Hopsworks 4.2.2
            # Try multiple ways to get the host URL
            try:
                # Method 1: From feature store object
                host = fs._feature_store_api._client._base
            except:
                try:
                    # Method 2: Build from project info
                    host = f"https://c.app.hopsworks.ai"
                except:
                    # Method 3: Get from any available API object
                    host = "https://c.app.hopsworks.ai:443"
            
            project_name = project.name
            project_id = project.id
            
            st.info(f"📡 Connecting to: {host}")
            st.info(f"📦 Project: {project_name} (ID: {project_id})")
            st.info(f"🗂️ Feature Group: {fg.name} v{fg.version}")
            
            # Construct headers with API key
            headers = {
                "Authorization": f"ApiKey {api_key}",
                "Content-Type": "application/json"
            }
            
            # === METHOD 1: Get Feature Group Data via Storage Connector API ===
            try:
                st.info("🔍 Method 1: Fetching via storage connector API...")
                
                # Get the feature group's storage connector info
                fg_id = fg.id
                fs_id = fs.id
                
                # Endpoint to get feature group details including storage location
                fg_details_url = f"{host}/hopsworks-api/api/project/{project_id}/featurestores/{fs_id}/featuregroups/{fg_id}"
                
                response = requests.get(fg_details_url, headers=headers)
                
                if response.status_code == 200:
                    fg_data = response.json()
                    st.success(f"✅ Retrieved feature group metadata")
                    
                    # Try to get the location
                    location = fg_data.get('location', '')
                    st.info(f"📂 Storage location: {location[:100]}...")
                    
                    # Now try to read the actual data using the query endpoint
                    # This is a different endpoint that might work
                    query_url = f"{host}/hopsworks-api/api/project/{project_id}/featurestores/{fs_id}/query"
                    
                    # Create a simple query payload
                    query_payload = {
                        "query": f"SELECT * FROM {fg.name}_{fg.version}",
                        "featurestore": fs.name
                    }
                    
                    query_response = requests.post(query_url, headers=headers, json=query_payload)
                    
                    if query_response.status_code == 200:
                        result = query_response.json()
                        # Try to convert to dataframe
                        if isinstance(result, dict) and 'data' in result:
                            df = pd.DataFrame(result['data'])
                        else:
                            df = pd.DataFrame(result)
                        
                        if not df.empty:
                            st.success(f"✅ Method 1: Retrieved {len(df)} records!")
                        else:
                            raise ValueError("Empty result")
                    else:
                        raise ValueError(f"Query failed: {query_response.status_code}")
                else:
                    raise ValueError(f"Cannot get FG details: {response.status_code}")
                    
            except Exception as e1:
                st.warning(f"⚠️ Method 1 failed: {str(e1)[:200]}")
                df = None
            
            # === METHOD 2: Direct SQL Execution Endpoint ===
            if df is None or df.empty:
                try:
                    st.info("🔍 Method 2: Direct SQL execution...")
                    
                    # Try the storage/query endpoint
                    sql_url = f"{host}/hopsworks-api/api/project/{project_id}/featurestores/{fs.id}/storageconnectors/HOPSFS_CONNECTOR/query"
                    
                    sql_payload = {
                        "query": f"SELECT * FROM `{project_name}_featurestore`.`{fg.name}_{fg.version}` LIMIT 1000"
                    }
                    
                    sql_response = requests.post(sql_url, headers=headers, json=sql_payload)
                    
                    if sql_response.status_code == 200:
                        data = sql_response.json()
                        df = pd.DataFrame(data)
                        
                        if not df.empty:
                            st.success(f"✅ Method 2: Retrieved {len(df)} records!")
                        else:
                            raise ValueError("Empty result")
                    else:
                        raise ValueError(f"SQL query failed: {sql_response.status_code}")
                        
                except Exception as e2:
                    st.warning(f"⚠️ Method 2 failed: {str(e2)[:200]}")
            
            # === METHOD 3: Feature Store Statistics Endpoint ===
            if df is None or df.empty:
                try:
                    st.info("🔍 Method 3: Getting data from statistics/preview endpoint...")
                    
                    # Many Hopsworks versions have a preview/sample endpoint
                    preview_url = f"{host}/hopsworks-api/api/project/{project_id}/featurestores/{fs.id}/featuregroups/{fg.id}/preview"
                    
                    preview_response = requests.get(
                        preview_url, 
                        headers=headers,
                        params={"limit": 100}  # Hopsworks max limit is 100
                    )
                    
                    if preview_response.status_code == 200:
                        data = preview_response.json()
                        
                        # DEBUG: Show what we received
                        st.info(f"📥 Response type: {type(data)}")
                        if isinstance(data, dict):
                            st.info(f"📋 Response keys: {list(data.keys())[:10]}")
                        
                        # Try multiple ways to extract the data
                        df = None
                        
                        # Format 1: Paginated response with 'items'
                        if isinstance(data, dict) and 'items' in data:
                            st.info(f"✓ Found 'items' key with {len(data['items'])} records")
                            df = pd.DataFrame(data['items'])
                        
                        # Format 2: Direct data array
                        elif isinstance(data, dict) and 'data' in data:
                            st.info(f"✓ Found 'data' key")
                            df = pd.DataFrame(data['data'])
                        
                        # Format 3: Rows array
                        elif isinstance(data, dict) and 'rows' in data:
                            st.info(f"✓ Found 'rows' key")
                            df = pd.DataFrame(data['rows'])
                        
                        # Format 4: Count response (means we need different endpoint)
                        elif isinstance(data, dict) and 'count' in data:
                            count = data.get('count', 0)
                            st.info(f"📊 Feature group has {count} total records")
                            # Try to get actual data if count > 0
                            if count > 0:
                                # Try without limit to get storage path
                                raise ValueError(f"Got count response ({count} records), need to use different endpoint")
                        
                        # Format 5: List response
                        elif isinstance(data, list) and len(data) > 0:
                            st.info(f"✓ Got list with {len(data)} items")
                            df = pd.DataFrame(data)
                        
                        # Format 6: Single dict record
                        elif isinstance(data, dict) and len(data) > 0:
                            st.info(f"✓ Got single record dict, converting to DataFrame")
                            df = pd.DataFrame([data])
                        
                        else:
                            st.warning(f"⚠️ Unknown response format: {str(data)[:200]}")
                            raise ValueError(f"Unrecognized response format")
                        
                        if df is not None and not df.empty:
                            st.success(f"✅ Method 3: Retrieved {len(df)} records!")
                        else:
                            raise ValueError("DataFrame is empty after parsing")
                    else:
                        raise ValueError(f"Preview failed: {preview_response.status_code} - {preview_response.text[:200]}")
                        
                except Exception as e3:
                    st.warning(f"⚠️ Method 3 failed: {str(e3)[:200]}")
            
            # === METHOD 4: Commit Data Endpoint ===
            if df is None or df.empty:
                try:
                    st.info("🔍 Method 4: Trying commit data endpoint...")
                    
                    # Try to get the latest commit
                    commits_url = f"{host}/hopsworks-api/api/project/{project_id}/featurestores/{fs.id}/featuregroups/{fg.id}/commits"
                    
                    commits_response = requests.get(commits_url, headers=headers)
                    
                    if commits_response.status_code == 200:
                        commits = commits_response.json()
                        st.info(f"📊 Found commits data")
                        
                        # If we have commit info, try to read the data
                        read_url = f"{host}/hopsworks-api/api/project/{project_id}/featurestores/{fs.id}/featuregroups/{fg.id}/read"
                        
                        read_response = requests.get(
                            read_url,
                            headers=headers,
                            params={"limit": 100}  # Hopsworks max limit is 100
                        )
                        
                        if read_response.status_code == 200:
                            data = read_response.json()
                            df = pd.DataFrame(data)
                            
                            if not df.empty:
                                st.success(f"✅ Method 4: Retrieved {len(df)} records!")
                            else:
                                raise ValueError("Empty read result")
                        else:
                            raise ValueError(f"Read failed: {read_response.status_code}")
                    else:
                        raise ValueError(f"Commits endpoint failed: {commits_response.status_code}")
                        
                except Exception as e4:
                    st.warning(f"⚠️ Method 4 failed: {str(e4)[:200]}")
            
            # === Final check ===
            if df is None or df.empty:
                st.error("❌ All REST API methods failed!")
                st.error("💡 The Hopsworks backend may not have REST endpoints available for data access.")
                st.info("🔧 **Alternative Solution**: Download data manually from Hopsworks UI and upload to Streamlit")
                
                with st.expander("📋 Manual Download Instructions"):
                    st.markdown("""
                    ### Manual Workaround:
                    
                    1. **Go to Hopsworks UI** → Feature Store → `karachi_aqi_forecast`
                    2. **Click "Preview"** or "Download"
                    3. **Export as CSV/Parquet**
                    4. **Upload to your GitHub repo** as `data/forecast_data.csv`
                    5. **Update the code** to read from local file instead
                    
                    This bypasses all API/version issues completely.
                    """)
                
                return None, best_model_obj, leaderboard
                
        except Exception as api_error:
            st.error(f"❌ REST API connection failed: {str(api_error)}")
            return None, best_model_obj, leaderboard

        # Process the dataframe
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
