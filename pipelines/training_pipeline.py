import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1️⃣ Login and get feature store
project = hopsworks.login()
fs = project.get_feature_store()

# 2️⃣ Get the safe FeatureView
fv = fs.get_feature_view(name="karachi_aqi_view", version=3)

# 3️⃣ Fetch training data
#    Use feature view's get_training_data to fetch a pandas DataFrame safely
td = fv.get_training_data()
df = td.to_pandas()  # ensure we have a pandas DataFrame

# 4️⃣ Separate features and label
label_column = "aqi"
X = df.drop(columns=[label_column])
y = df[label_column]

# 5️⃣ Train-test split (keep your logic intact)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Model training (example: RandomForest, same as your pipeline)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7️⃣ Save the trained model
joblib.dump(model, "aqi_model.pkl")

print("✅ Training pipeline executed successfully")
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
