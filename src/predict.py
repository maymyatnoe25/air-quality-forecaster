import json

import joblib
import pandas as pd

from src import config
from src.features import build_features
from src.live import fetch_recent

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(config.MODELS / "bundle.joblib")
    return _bundle


def category(pm25):
    """US EPA PM2.5 categories."""
    bounds = [(9.0, "Good"), (35.4, "Moderate"),
              (55.4, "Unhealthy for sensitive groups"),
              (125.4, "Unhealthy"), (225.4, "Very unhealthy")]
    for upper, label in bounds:
        if pm25 <= upper:
            return label
    return "Hazardous"


def alert_level(pm25, trigger):
    if pm25 > config.ALERT_SEVERE:
        return "severe"
    if pm25 > trigger:
        return "early_warning"
    return "none"


def get_forecast():
    b = _load()
    feats = build_features(fetch_recent())
    row = feats.dropna(subset=b["features"]).iloc[[-1]]   # latest complete hour
    X = row[b["features"]]
    issued = row.index[0]

    forecasts = []
    for h in sorted(b["models"]):
        m = b["models"][h]
        p = max(float(m["point"].predict(X)[0]), 0.0)
        lo = max(float(m["q10"].predict(X)[0]), 0.0)
        hi = max(float(m["q90"].predict(X)[0]), 0.0)
        lo, hi = min(lo, p), max(hi, p)
        forecasts.append({
            "horizon_h": int(h),
            "target_time": (issued + pd.Timedelta(hours=h)).isoformat(),
            "pm25": round(p, 1),
            "low": round(lo, 1),
            "high": round(hi, 1),
            "category": category(p),
            "alert": alert_level(p, b["triggers"][h]),
        })

    current = float(row["pm2_5"].iloc[0])
    return {
        "city": config.DEFAULT_CITY,
        "issued_at": issued.isoformat(),
        "current_pm25": round(current, 1),
        "current_category": category(current),
        "forecasts": forecasts,
    }


if __name__ == "__main__":
    print(json.dumps(get_forecast(), indent=2))