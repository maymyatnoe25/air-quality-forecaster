import requests
import pandas as pd

# 1. The web address (API) that gives us air quality data
url = "https://air-quality-api.open-meteo.com/v1/air-quality"

# 2. What we're asking for: location, which measurement, how many past days
params = {
    "latitude": 3.139,        # Kuala Lumpur
    "longitude": 101.687,
    "hourly": "pm2_5",        # the dust particle measurement
    "past_days": 3,           # last 3 days
    "forecast_days": 1,
    "timezone":  "Asia/Kuala_Lumpur",
}

# 3. Send the request and get the answer
response = requests.get(url, params=params, timeout=60)
response.raise_for_status()   # stops with an error if something went wrong

# 4. Turn the answer into a table
data = response.json()
df = pd.DataFrame(data["hourly"])

# 5. Look at it
print(df.head(10))     # first 10 rows
print(df.shape)        # (rows, columns)
print(df.describe())   # summary statistics
df["time"] = pd.to_datetime(df["time"])
print(df.groupby(df["time"].dt.hour)["pm2_5"].mean().round(1))