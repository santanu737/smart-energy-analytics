import numpy as np
import pandas as pd

from src.anomaly import (ML_FEATURES, fit_isolation_forest, flat_run_mask, inject_faults,
                         make_ml_features, rule_flags, score_features)


def make_long(n_hours=1400, n_buildings=3, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2016-01-01", periods=n_hours, freq="h")
    frames = []
    for i in range(n_buildings):
        base = 50 * (i + 1) * (1 + 0.5 * np.sin(2 * np.pi * ts.hour / 24))
        frames.append(pd.DataFrame({"timestamp": ts, "building_id": f"b{i}",
                                    "kwh": (base + rng.normal(0, 2, n_hours)).astype("float32")}))
    df = pd.concat(frames, ignore_index=True)
    df["building_id"] = df["building_id"].astype("category")   # real data uses a categorical id
    return df


def test_flat_run_mask_marks_only_long_runs_and_nan_breaks_runs():
    x = np.array([1, 2, 2, 2, 2, 3, np.nan, 3, 3, 3.0])
    m = flat_run_mask(x, 4)
    assert m.tolist() == [False, True, True, True, True, False, False, False, False, False]


def test_rule_flags_zero_and_flat_are_separate():
    idx = pd.date_range("2016-01-01", periods=60, freq="h")
    a = np.arange(60, dtype="float32") + 1
    a[10] = 0
    a[20:50] = 7
    raw = pd.DataFrame({"a": a}, index=idx)
    zero, flat = rule_flags(raw, min_flat_hours=24)
    assert zero["a"].sum() == 1 and zero["a"].iloc[10]
    assert flat["a"].sum() == 30 and flat["a"].iloc[20:50].all()


def test_inject_faults_only_after_start_and_labels_match_changes():
    df = make_long(1000, 3)
    start = pd.Timestamp("2016-01-01") + pd.Timedelta(hours=400)
    inj = inject_faults(df, start=start, spike_rate=0.01, drop_rate=0.01, stretch_rate=0.003)
    orig = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    assert (inj.loc[inj["timestamp"] <= start, "fault"] == 0).all()
    changed = ~np.isclose(inj["kwh"].to_numpy(), orig["kwh"].to_numpy())
    assert (changed == (inj["fault"].to_numpy() > 0)).all()
    ratio = inj["kwh"].to_numpy() / orig["kwh"].to_numpy()
    assert (ratio[inj["fault"] == 1] > 2.9).all()
    assert (ratio[inj["fault"] == 2] < 0.11).all()
    assert set(inj["fault"].unique()) == {0, 1, 2, 3}


def test_features_do_not_leak_future_and_iforest_scores_spikes_higher():
    df = make_long(1400, 3)
    train_end = pd.Timestamp("2016-01-01") + pd.Timedelta(hours=800)
    feats = make_ml_features(df, train_end)
    for c in ML_FEATURES:
        assert c in feats.columns
    assert not np.isinf(feats[ML_FEATURES].to_numpy(dtype="float64")).any()

    # changing values after row t must not change features at row t
    df2 = df.copy()
    late = df2["timestamp"] > pd.Timestamp("2016-01-01") + pd.Timedelta(hours=1000)
    df2.loc[late, "kwh"] = 9999.0
    f2 = make_ml_features(df2, train_end)
    early = feats["timestamp"] <= pd.Timestamp("2016-01-01") + pd.Timedelta(hours=1000)
    a = feats.loc[early, ML_FEATURES].reset_index(drop=True)
    b = f2.loc[f2["timestamp"] <= pd.Timestamp("2016-01-01") + pd.Timedelta(hours=1000), ML_FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)

    bundle = fit_isolation_forest(feats, train_end)
    inj = inject_faults(df, start=train_end, spike_rate=0.01, drop_rate=0.0, stretch_rate=0.0)
    fi = make_ml_features(inj, train_end, extra=("fault",))
    s = score_features(bundle, fi)
    ok = ~np.isnan(s) & (fi["timestamp"] > train_end).to_numpy()
    spike = fi["fault"].to_numpy() == 1
    assert spike[ok].sum() > 0
    assert np.nanmean(s[ok & spike]) > np.nanmean(s[ok & ~spike])
