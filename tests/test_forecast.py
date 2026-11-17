import numpy as np
import pandas as pd

from src.forecast import (GRID, MIN_REPORTING, TRAIN_END, VAL_END, feature_cols, make_table,
                          regression_metrics, run_forecast, split_table)


def synthetic(seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2016-01-01", "2017-12-31 23:00", freq="h")
    n = len(ts)
    temp = 12 + 13 * np.sin((ts.dayofyear.values - 110) / 365 * 2 * np.pi) + rng.normal(0, 1, n)
    load = 1000 + 400 * np.exp(-((ts.hour.values - 14) ** 2) / 30) + 8 * np.clip(temp - 18, 0, None) + rng.normal(0, 15, n)
    total = pd.DataFrame({"timestamp": ts, "kwh": load, "n_reporting": 131})
    total.loc[[10, 500], "kwh"] = np.nan
    total.loc[3000:3004, "n_reporting"] = 90          # hours that must never be used as targets
    weather = pd.DataFrame({"timestamp": ts, "airTemperature": temp, "cloudCoverage": 2.0,
                            "dewTemperature": temp - 5, "precipDepth1HR": 0.0, "precipDepth6HR": 0.0,
                            "seaLvlPressure": 1015.0, "windDirection": 90.0, "windSpeed": 3.0})
    return total, weather


def test_split_is_ordered_disjoint_and_skips_low_reporting_targets():
    total, weather = synthetic()
    t = make_table(total, weather, 6)
    parts = split_table(t)
    assert parts["train"]["target_timestamp"].max() <= TRAIN_END
    assert parts["val"]["target_timestamp"].min() > TRAIN_END
    assert parts["val"]["target_timestamp"].max() <= VAL_END
    assert parts["test"]["target_timestamp"].min() > VAL_END
    used = pd.concat(parts.values())
    assert (used["target_n_reporting"] >= MIN_REPORTING).all()
    assert used["target"].notna().all()


def test_feature_columns_exclude_targets_and_weather_switch_works():
    total, weather = synthetic()
    t = make_table(total, weather, 1)
    cols = feature_cols(t, True)
    for bad in ("target", "target_timestamp", "naive_24", "naive_168", "n_reporting", "persist"):
        assert bad not in cols
    assert "airTemperature" in cols
    assert not any(c.startswith("airTemperature") or c.endswith("_gap_flag") for c in feature_cols(t, False))
    assert "kwh" in cols and "hour_sin" in cols


def test_regression_metrics_known_values():
    m = regression_metrics([100, 200], [110, 180])
    assert m["mae"] == 15.0 and m["rmse"] == round(((100 + 400) / 2) ** 0.5, 1)


def test_model_beats_weekly_naive_baseline_on_synthetic_data():
    total, weather = synthetic()
    res = run_forecast(total, weather, horizons=(1,), grid=GRID[:1])
    m = res["metrics"][1]
    assert m["hgb_with_weather"]["mae"] < m["naive_168"]["mae"]
    assert m["hgb_with_weather"]["mae"] < m["persistence"]["mae"]
    assert len(res["preds"][1]) == m["rows"]["test"]
