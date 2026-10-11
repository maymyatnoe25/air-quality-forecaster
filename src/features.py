import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 6, 12, 24, 48, 72]
WINDOWS = [6, 24, 72]


def build_features(df):
    """Every feature in row t uses only information available at time t."""
    out = df.copy()

    out["hour_sin"] = np.sin(2 * np.pi * out.index.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out.index.hour / 24)
    out["month_sin"] = np.sin(2 * np.pi * out.index.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * out.index.month / 12)
    out["dow"] = out.index.dayofweek

    rad = np.deg2rad(out["wind_direction_10m"])
    out["wind_sin"], out["wind_cos"] = np.sin(rad), np.cos(rad)
    out = out.drop(columns=["wind_direction_10m"])

    for lag in LAGS:
        out[f"pm25_lag_{lag}"] = out["pm2_5"].shift(lag)
    for w in WINDOWS:
        out[f"pm25_mean_{w}"] = out["pm2_5"].rolling(w).mean()
        out[f"pm25_std_{w}"] = out["pm2_5"].rolling(w).std()
    out["pm25_diff_1"] = out["pm2_5"].diff(1)
    out["pm25_diff_24"] = out["pm2_5"].diff(24)
    out["rain_sum_24"] = out["precipitation"].rolling(24).sum()
    return out