import numpy as np
import pandas as pd

from src import config
from src.features import build_features


def test_no_future_leakage():
    cols = config.AQ_VARS + config.WX_VARS
    idx = pd.date_range("2024-01-01", periods=400, freq="h", tz=config.TZ)
    df = pd.DataFrame(np.random.default_rng(0).random((400, len(cols))),
                      index=idx, columns=cols)

    a = build_features(df)

    changed = df.copy()
    changed.iloc[300:] = 999
    b = build_features(changed)

    pd.testing.assert_frame_equal(a.iloc[:300], b.iloc[:300])