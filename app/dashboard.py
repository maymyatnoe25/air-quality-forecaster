import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.live import fetch_recent
from src.predict import get_forecast

st.set_page_config(page_title="Malaysia Air Quality Forecaster", page_icon="🌫️", layout="wide")


@st.cache_data(ttl=1800)
def load_forecast():
    return get_forecast()


@st.cache_data(ttl=1800)
def load_history():
    return fetch_recent()


st.title(f"🌫️ PM2.5 Forecast: {config.DEFAULT_CITY}")
tab1, tab2, tab3 = st.tabs(["Forecast", "Model performance", "About & limitations"])

with tab1:
    try:
        f, hist = load_forecast(), load_history()
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()

    c1, c2 = st.columns(2)
    c1.metric("Current PM2.5 (µg/m³)", f["current_pm25"])
    c2.metric("Current category", f["current_category"])

    fc = f["forecasts"]
    severe = [x for x in fc if x["alert"] == "severe"]
    early = [x for x in fc if x["alert"] == "early_warning"]
    if severe:
        x = min(severe, key=lambda z: z["horizon_h"])
        st.error(f"🔴 Severe: PM2.5 forecast above {config.ALERT_SEVERE} µg/m³ "
                 f"within {x['horizon_h']}h (≈{x['pm25']}).")
    elif early:
        x = min(early, key=lambda z: z["horizon_h"])
        st.warning(f"🟠 Early warning: elevated PM2.5 possible within {x['horizon_h']}h "
                   f"(≈{x['pm25']}). Sensitive groups may want to limit outdoor exercise.")
    else:
        st.success("🟢 No alert triggered for the next 48 hours.")

    fdf = pd.DataFrame(fc)
    fdf["time"] = pd.to_datetime(fdf["target_time"])
    h72 = hist.tail(72)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h72.index, y=h72["pm2_5"], name="Observed (last 72h)"))
    fig.add_trace(go.Scatter(
        x=fdf["time"], y=fdf["pm25"], mode="markers+lines", name="Forecast",
        error_y=dict(type="data", symmetric=False,
                     array=fdf["high"] - fdf["pm25"],
                     arrayminus=fdf["pm25"] - fdf["low"])))
    fig.add_hline(y=config.ALERT_SENSITIVE, line_dash="dash",
                  annotation_text="35.4 sensitive-group level")
    fig.add_hline(y=config.ALERT_SEVERE, line_dash="dot",
                  annotation_text="55.4 severe level")
    fig.update_layout(yaxis_title="PM2.5 (µg/m³)", height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(fdf[["horizon_h", "pm25", "low", "high", "category", "alert"]],
                 hide_index=True)
    st.caption(f"Issued at {f['issued_at']} (Malaysia time). "
               "The bars show an 80% range that is slightly overconfident (see the About tab).")

with tab2:
    m = json.loads((config.MODELS / "metrics.json").read_text())
    rows = []
    for h in config.HORIZONS:
        r = m[str(h)]
        base = min(r["persistence"]["mae"], r["seasonal_naive"]["mae"])
        sev = r["alert_severe_at_trigger"]["lightgbm"]
        rows.append({
            "Horizon (h)": h,
            "Best baseline MAE": base,
            "LightGBM MAE": r["lightgbm"]["mae"],
            "Improvement %": (1 - r["lightgbm"]["mae"] / base) * 100,
            "Early-warning trigger": r["early_trigger"],
            "Severe recall": sev["recall"],
            "Severe precision": sev["precision"],
            "80% interval coverage %": r["interval_coverage_80"] * 100,
        })
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True)
    st.caption("Held-out chronological test set (Feb to Oct 2026). MAE in µg/m³. "
               "The baseline is the better of 'persistence' and 'same clock time yesterday'.")

with tab3:
    st.markdown("""
**Data:** Open-Meteo. Air quality comes from the CAMS atmospheric *model*, not ground sensors.

**Method:** LightGBM per horizon (6, 12, 24, 48h) on lag, rolling, calendar and weather features.
Quantile models give an 80% range. Chronological split with a gap and an automated leakage test.

**Limitations:**
- Severe episodes (above about 100 µg/m³) are underpredicted.
- The 80% range really covers about 72-75% of outcomes.
- The early warning trades precision for recall: many alerts are false alarms.
- Models are trained on Kuala Lumpur only.
- Training used archived weather, live use uses weather forecasts, an effect I have not measured.
- Informational only, not medical advice.
""")