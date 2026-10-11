from pathlib import Path

# ---- Cities: name -> (latitude, longitude) ----
CITIES = {
    "Kuala Lumpur":   (3.139, 101.687),
    "Putrajaya":      (2.926, 101.696),
    "Shah Alam":      (3.074, 101.518),
    "George Town":    (5.414, 100.329),
    "Ipoh":           (4.598, 101.090),
    "Johor Bahru":    (1.493, 103.741),
    "Melaka":         (2.190, 102.250),
    "Seremban":       (2.726, 101.938),
    "Kuantan":        (3.808, 103.326),
    "Kuala Terengganu": (5.330, 103.137),
    "Kota Bharu":     (6.125, 102.238),
    "Alor Setar":     (6.125, 100.368),
    "Kuching":        (1.554, 110.359),
    "Miri":           (4.400, 113.991),
    "Kota Kinabalu":  (5.980, 116.074),
}
DEFAULT_CITY = "Kuala Lumpur"

# ---- Time ----
TZ = "Asia/Kuala_Lumpur"        # the same for every city in Malaysia
START_DATE = "2022-08-01"       # how far back we download

# ---- What we download ----
AQ_VARS = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone",
           "sulphur_dioxide", "carbon_monoxide"]
WX_VARS = ["temperature_2m", "relative_humidity_2m", "precipitation",
           "surface_pressure", "wind_speed_10m", "wind_direction_10m"]

# ---- Folders ----
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"