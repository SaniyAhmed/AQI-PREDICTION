import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
import os

def advanced_processing(file_path):
    print(f"🛠️ Starting Hybrid Preprocessing: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    
    # 1. Cleaning
    df = df.drop_duplicates()
    df = df.ffill().bfill() 

    # 2. FEATURE ENGINEERING (Lags)
    df['aqi_lag_1'] = df['aqi'].shift(1)
    df['aqi_lag_2'] = df['aqi'].shift(2)
    df['pm25_lag_1'] = df['pm25'].shift(1)
    df = df.dropna()

    # 3. DOMAIN EXPERT SELECTION (The "Must-Haves")
    mandatory_features = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']
    # Filter to ensure they exist in raw data
    mandatory_features = [f for f in mandatory_features if f in df.columns]

    # 4. PREPARING FOR RFE (The "Supporting Cast")
    # We let RFE choose from everything ELSE (Weather, Lags, etc.)
    target = 'aqi'
    metadata = ['timestamp_unix', 'year', 'month', 'day', 'hour']
    
    # X_rfe contains only the candidates for AI selection
    X_rfe = df.drop(columns=[target] + metadata + mandatory_features, errors='ignore')
    y = df[target]

    # 5. RFE SELECTION
    print(f"🎯 AI is choosing the best 5 supporting features from: {X_rfe.columns.tolist()}")
    estimator = XGBRegressor(n_estimators=100, random_state=42)
    
    # We ask RFE to pick the top 5 supporting features
    selector = RFE(estimator, n_features_to_select=5, step=1)
    selector = selector.fit(X_rfe, y)
    
    ai_selected_features = X_rfe.columns[selector.support_].tolist()
    
    # 6. MERGE BOTH LISTS
    final_feature_list = mandatory_features + ai_selected_features
    print(f"✅ FINAL HYBRID LIST: {final_feature_list}")

    # 7. STANDARDIZATION
    scaler = StandardScaler()
    df_scaled = df.copy()
    # We scale only the features, not the target or keys
    df_scaled[final_feature_list] = scaler.fit_transform(df[final_feature_list])

    # 8. FINAL DATASET CONSTRUCTION
    primary_keys = ['year', 'month', 'day', 'hour']
    final_columns = list(dict.fromkeys(final_feature_list + [target] + primary_keys))
    processed_df = df_scaled[final_columns].copy()
    
    for col in primary_keys:
        processed_df[col] = processed_df[col].astype('int64')

    # 9. SAVE
    output_dir = os.path.join('data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'processed_karachi_data.csv')
    processed_df.to_csv(output_path, index=False)
    
    print(f"💾 Saved {len(processed_df)} rows to: {output_path}")
    return final_feature_list

if __name__ == "__main__":
    FILE_PATH = os.path.join('data', 'raw', 'karachi_schema_data.csv')
    try:
        selected = advanced_processing(FILE_PATH)
    except Exception as e:
        print(f"❌ Error: {e}")
