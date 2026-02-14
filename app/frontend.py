🔐 Logging into Hopsworks...
2026-02-14 08:28:44,689 INFO: Initializing external client
2026-02-14 08:28:44,689 INFO: Base URL: https://c.app.hopsworks.ai:443
2026-02-14 08:28:47,666 INFO: Python Engine initialized.

Logged in to project, explore it here https://c.app.hopsworks.ai:443/p/1335454
📥 Fetching karachi_aqi_daily_summary v2...
Finished: Reading data from Hopsworks, using Hopsworks Feature Query Service (0.88s) 
✅ Successfully fetched 9 rows from karachi_aqi_daily_summary v2

📊 Data Summary:
   Columns: ['date', 'daily_avg_aqi', 'grand_avg_aqi', 'forecast_type']
   Date Range: 2026-02-08 to 2026-02-16
   Grand Average AQI: 104.74

✅ Successfully saved to data/forecast_data.csv

📋 Forecast Data:
      date  daily_avg_aqi  grand_avg_aqi  forecast_type
2026-02-08         102.56         104.74 3-day-ensemble
2026-02-09          90.07          94.02 3-day-ensemble
2026-02-10         105.06         107.43 3-day-ensemble
2026-02-11         101.27         103.44 3-day-ensemble
2026-02-12         101.32         103.43 3-day-ensemble
2026-02-13         131.28         131.93 3-day-ensemble
2026-02-14         125.02         125.05 3-day-ensemble
2026-02-15         125.09         125.05 3-day-ensemble
2026-02-16         125.03         125.05 3-day-ensemble


🤖 Fetching Latest Model Info...

📊 Latest Model (Version 79):
   Description: Winner: SVR | NO SMOOTHING
   Metrics:
      randomforest_rmse: 2.1161527995478964
      svr_rmse: 0.6000716259093256
      rmse: 0.7560341935210918
      winner_rmse: 0.6000716259093256
      xgboost_rmse: 0.931509688724162
2026-02-14 08:29:06,159 INFO: Closing external client and cleaning up certificates.
Connection closed.

🔓 Logged out from Hopsworks
