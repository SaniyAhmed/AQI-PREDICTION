# 🌍 Karachi AQI Forecasting System (Serverless MLOps)

A production-ready, end-to-end MLOps pipeline designed to forecast the Air Quality Index (AQI) for Karachi, Pakistan, over a 72-hour horizon. This system leverages a serverless architecture to provide real-time health insights and meteorological transparency.

---

## 🔗 Live Dashboard

👉 **Streamlit App:** https://karachi-aqi-sentinel.streamlit.app/

---

## 🚀 System Architecture

The project follows a modular MLOps lifecycle managed via **GitHub Actions** and **Hopsworks**:

1. **Feature Pipeline:** Hourly ingestion of weather and pollutant data from the Open-Meteo API.
2. **Feature Store:** Uses Hopsworks as a Single Source of Truth to eliminate **Training-Serving Skew**.
3. **Training Pipeline:** A daily "Champion-Challenger" cycle evaluating **XGBoost, Random Forest, and SVR**.
4. **Inference Dashboard:** A Streamlit UI providing live forecasts and **SHAP-based explainability**.

---

## 🛠️ Tech Stack

| Layer              | Technology                               |
| :----------------- | :--------------------------------------- |
| **Language**       | Python 3.x                               |
| **Feature Store**  | Hopsworks                                |
| **ML Models**      | Scikit-learn, XGBoost, SVR, RandomForest |
| **Orchestration**  | GitHub Actions (CI/CD)                   |
| **Dashboard**      | Streamlit & Flask                        |
| **Explainability** | SHAP                                     |

---

## 📊 Key Insights & Analytics

* **Atmospheric Momentum:** The system utilizes "AQI Change Rate" to capture sudden pollution spikes.
* **SHAP Transparency:** Identifies $PM_{2.5}$ and Wind Speed as the primary drivers of Karachi's air quality.
* **Robustness:** Mitigates overfitting through Time-Series Nested Cross-Validation.

---

## 📂 Project Structure

* `app/`: Streamlit frontend and Flask backend logic
* `data/`: Contains raw and processed data
* `notebooks/`: Exploratory Data Analysis (EDA) and model prototyping
* `pipelines/`: Automation scripts for data ingestion and training
* `docs/`: Full technical project report

---

**Developed by:** Saniya Ahmed | Data Science Intern @ 10Pearls
