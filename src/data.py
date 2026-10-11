import argparse
import time
from datetime import date, timedelta

import pandas as pd
import requests

from src import config

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_chunk(url, lat, lon, variables, start, end):
    """Ask the API for one date range and return it as a table."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(variables),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": config.TZ,
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    df = pd.DataFrame(response.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def date_chunks(start, end, days=90):
    """Split a long period into pieces of 90 days."""
    s = start
    while s <= end:
        e = min(s + timedelta(days=days - 1), end)
        yield s, e
        s = e + timedelta(days=1)


def download(url, lat, lon, variables, start, end):
    """Download all the chunks and glue them into one table."""
    pieces = []
    for s, e in date_chunks(start, end):
        print(f"  {s} to {e}")
        pieces.append(fetch_chunk(url, lat, lon, variables, s, e))
        time.sleep(0.5)  # short pause so we are polite to the free API
    df = pd.concat(pieces)
    return df[~df.index.duplicated()]


def download_city(city):
    lat, lon = config.CITIES[city]
    start = date.fromisoformat(config.START_DATE)
    end = date.today() - timedelta(days=6)  # the weather archive lags a few days

    print(f"Downloading air quality for {city}...")
    aq = download(AQ_URL, lat, lon, config.AQ_VARS, start, end)
    print(f"Downloading weather for {city}...")
    wx = download(WX_URL, lat, lon, config.WX_VARS, start, end)

    df = aq.join(wx, how="inner")
    df.index = df.index.tz_localize(config.TZ)  # mark times as Kuala Lumpur time
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default=config.DEFAULT_CITY)
    args = parser.parse_args()

    if args.city not in config.CITIES:
        raise SystemExit(f"Unknown city. Choose from: {list(config.CITIES)}")

    df = download_city(args.city)

    config.RAW.mkdir(parents=True, exist_ok=True)
    filename = args.city.lower().replace(" ", "_") + ".parquet"
    df.to_parquet(config.RAW / filename)

    print("\nSaved:", config.RAW / filename)
    print("Shape (rows, columns):", df.shape)
    print("From", df.index.min(), "to", df.index.max())
    print("\nMissing values per column:")
    print(df.isna().sum())