import json
import math

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (fbeta_score, mean_absolute_error,
                             mean_squared_error, precision_score,
                             recall_score)

from src import config
from src.features import build_features

NOT_FEATURES = ["y", "seasonal_naive", "is_glitch", "pm2_5_filled"]


def make_dataset(feat, h):
    """Add the target (PM2.5 h hours ahead) and a 'same hour yesterday' baseline."""
    d = feat.copy()
    d["y"] = d["pm2_5"].shift(-h)
    # value at the same clock time as the target, one or more days earlier
    d["seasonal_naive"] = d["pm2_5"].shift(24 * math.ceil(h / 24) - h)
    return d.dropna()


def chrono_split(d, h):
    """70/15/15 in time order, with an h-hour gap so targets never overlap."""
    n = len(d)
    i1, i2 = int(n * 0.70), int(n * 0.85)
    return d.iloc[: i1 - h], d.iloc[i1: i2 - h], d.iloc[i2:]


def fit_lgb(train, valid, cols, **kw):
    m = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=31,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=42, verbose=-1, **kw)
    m.fit(train[cols], train["y"],
          eval_set=[(valid[cols], valid["y"])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def reg_metrics(y, pred):
    return {"mae": float(mean_absolute_error(y, pred)),
            "rmse": float(mean_squared_error(y, pred) ** 0.5)}


def alert_metrics(y_true, y_pred):
    return {"precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))}


def tune_trigger(pred_valid, y_valid):
    """Pick the early-warning trigger on VALIDATION data (best F2 for severe hours)."""
    severe = y_valid > config.ALERT_SEVERE
    best_t, best_f2 = None, -1.0
    for t in range(20, 62, 2):
        f2 = fbeta_score(severe, pred_valid > t, beta=2, zero_division=0)
        if f2 > best_f2:
            best_t, best_f2 = t, f2
    return best_t


def main():
    df = pd.read_parquet(config.PROCESSED / "kuala_lumpur_clean.parquet")
    feat = build_features(df)

    bundle = {"models": {}, "triggers": {}, "horizons": config.HORIZONS}
    report = {}

    for h in config.HORIZONS:
        print(f"\n=== Horizon {h}h ===")
        d = make_dataset(feat, h)
        cols = [c for c in d.columns if c not in NOT_FEATURES]
        train, valid, test = chrono_split(d, h)
        print("rows  train/valid/test:", len(train), len(valid), len(test))

        point = fit_lgb(train, valid, cols)
        q10 = fit_lgb(train, valid, cols, objective="quantile", alpha=0.1)
        q90 = fit_lgb(train, valid, cols, objective="quantile", alpha=0.9)

        pred = np.clip(point.predict(test[cols]), 0, None)
        lo = np.minimum(np.clip(q10.predict(test[cols]), 0, None), pred)
        hi = np.maximum(np.clip(q90.predict(test[cols]), 0, None), pred)

        # early-warning trigger: chosen on validation only
        pred_valid = np.clip(point.predict(valid[cols]), 0, None)
        trigger = tune_trigger(pred_valid, valid["y"])

        severe = test["y"] > config.ALERT_SEVERE
        sens = test["y"] > config.ALERT_SENSITIVE
        inside = (test["y"] >= lo) & (test["y"] <= hi)

        report[str(h)] = {
            "n_test": int(len(test)),
            "test_from": str(test.index.min()),
            "test_to": str(test.index.max()),
            "persistence": reg_metrics(test["y"], test["pm2_5"]),
            "seasonal_naive": reg_metrics(test["y"], test["seasonal_naive"]),
            "lightgbm": reg_metrics(test["y"], pred),
            "early_trigger": int(trigger),
            "severe_events_share": float(severe.mean()),
            "alert_severe_at_trigger": {
                "lightgbm": alert_metrics(severe, pred > trigger),
                "persistence": alert_metrics(severe, test["pm2_5"] > trigger)},
            "alert_sensitive_at_35.4": {
                "lightgbm": alert_metrics(sens, pred > config.ALERT_SENSITIVE),
                "persistence": alert_metrics(sens, test["pm2_5"] > config.ALERT_SENSITIVE)},
            "interval_coverage_80": float(inside.mean()),
            "interval_coverage_80_when_severe": float(inside[severe].mean()),
            "avg_interval_width": float((hi - lo).mean()),
        }
        bundle["models"][h] = {"point": point, "q10": q10, "q90": q90}
        bundle["triggers"][h] = int(trigger)
        bundle["features"] = cols

        r = report[str(h)]
        gain = 1 - r["lightgbm"]["mae"] / r["persistence"]["mae"]
        print(f"MAE  persistence {r['persistence']['mae']:.2f} | "
              f"seasonal {r['seasonal_naive']['mae']:.2f} | "
              f"LightGBM {r['lightgbm']['mae']:.2f}  ({gain:.1%} vs persistence)")
        print(f"Trigger {trigger} | severe recall "
              f"{r['alert_severe_at_trigger']['lightgbm']['recall']:.2f} | "
              f"interval coverage {r['interval_coverage_80']:.1%}")

    config.MODELS.mkdir(exist_ok=True)
    joblib.dump(bundle, config.MODELS / "bundle.joblib")
    (config.MODELS / "metrics.json").write_text(json.dumps(report, indent=2))
    print("\nSaved models/bundle.joblib and models/metrics.json")


if __name__ == "__main__":
    main()