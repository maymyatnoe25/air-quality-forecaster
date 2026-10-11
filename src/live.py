import pandas as pd
import requests

from src import config

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://api.open-meteo.com/v1/forecast"


def _get(url, lat, lon, variables, days):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(variables),
        "past_days": days,
        "forecast_days": 1,
        "timezone": config.TZ,
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(config.TZ)
    return df.set_index("time")


def fetch_recent(city=config.DEFAULT_CITY, days=7):
    """Latest hourly air quality + weather, up to the current hour (KL time)."""
    lat, lon = config.CITIES[city]
    aq = _get(AQ_URL, lat, lon, config.AQ_VARS, days)
    wx = _get(WX_URL, lat, lon, config.WX_VARS, days)
    df = aq.join(wx, how="inner")
    now = pd.Timestamp.now(tz=config.TZ).floor("h")
    df = df[df.index <= now]          # drop the future hours the API also returns
    return df.interpolate(limit=6, limit_area="inside")