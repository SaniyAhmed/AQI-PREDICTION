import requests
import json
import time

# Use the exact URL your frontend uses
BACKEND_URL = "http://localhost:5000/model-metrics"

def check_backend_truth():
    print("="*50)
    print("🔍 API DEBUGGER: CHECKING THE SOURCE OF TRUTH")
    print("="*50)
    
    try:
        # We add a timestamp to ensure we aren't getting a cached response from your OS
        t_stamp = time.time()
        print(f"📡 Sending request to: {BACKEND_URL}?t={t_stamp}")
        
        response = requests.get(f"{BACKEND_URL}?t={t_stamp}")
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n✅ SUCCESS: Backend responded.")
            print("-" * 30)
            print("📦 RAW JSON DATA RECEIVED:")
            print(json.dumps(data, indent=4))
            print("-" * 30)
            
            # Extract the critical field
            winner = data.get('winner_name', 'MISSING_FIELD')
            print(f"🏆 'winner_name' value: [{winner}]")
            
            # Technical check on string contents
            print(f"🧪 String Length: {len(str(winner))}")
            print(f"🧪 Contains 'random': {'random' in str(winner).lower()}")
            print(f"🧪 Contains 'forest': {'forest' in str(winner).lower()}")
            print("-" * 30)
            
            if "random" in str(winner).lower() or "forest" in str(winner).lower():
                print("🎯 THE FRONTEND SHOULD SHOW: [Random Forest Logic]")
            else:
                print("⚠️ THE FRONTEND WILL FALLBACK TO: [SVR Logic]")
                
        else:
            print(f"❌ ERROR: Backend returned Status Code {response.status_code}")
            
    except Exception as e:
        print(f"❌ CONNECTION FAILED: Is your backend_flask.py running?")
        print(f"Error details: {e}")

    print("="*50)

if __name__ == "__main__":
    check_backend_truth()