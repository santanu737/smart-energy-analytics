"""Phase 4: anomaly detection for campus buildings.

Three layers:
  1. Rules on the RAW electricity file: zero readings and flat runs (>= 24 h
     with the exact same value, which usually means a stuck meter).
  2. Isolation Forest on the CLEANED data, using building-relative features,
     for spikes, drops and odd patterns that rules cannot describe.
  3. Validation: real data has no anomaly labels, so synthetic faults are
     injected into the test period and we measure precision and recall.

Run:  python -m src.anomaly
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from src.features import build_anomaly_features

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "bdg2"
MODELS = ROOT / "models"

TRAIN_END = pd.Timestamp("2017-03-31 23:00:00")
CONTAMINATION = 0.005
MIN_FLAT_HOURS = 24
RANDOM_STATE = 42
ML_FEATURES = ["ratio_to_building_mean", "dev_rel", "rolling_z",
               "hour_sin", "hour_cos", "is_weekend"]
FAULT_NAMES = {1: "spike", 2: "drop", 3: "stretch"}


# ---------------------------------------------------------------- rules
def load_raw_wide(path: Path, buildings) -> pd.DataFrame:
    """Read the wide raw/cleaned electricity CSV for the given buildings only."""
    wanted = set(map(str, buildings))
    first = pd.read_csv(path, nrows=0).columns[0]
    df = pd.read_csv(path, usecols=lambda c: c == first or c in wanted,
                     index_col=0, parse_dates=True)
    return df.astype("float32")


def flat_run_mask(x: np.ndarray, min_len: int) -> np.ndarray:
    """True where x sits inside a run of >= min_len identical values.

    NaN breaks a run (NaN != NaN), and NaN positions are never flagged.
    """
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=bool)
    same = np.zeros(n, dtype=bool)
    same[1:] = x[1:] == x[:-1]
    run_id = np.cumsum(~same)
    lengths = np.bincount(run_id)
    return (lengths[run_id] >= min_len) & ~np.isnan(x)


def rule_flags(raw: pd.DataFrame, min_flat_hours: int = MIN_FLAT_HOURS):
    """Return (zero_mask, flat_mask) as boolean DataFrames shaped like raw.

    A zero run is reported as 'zero', not as 'flat'.
    """
    zero = raw.eq(0) & raw.notna()
    flat = pd.DataFrame(
        {c: flat_run_mask(raw[c].to_numpy(dtype="float64"), min_flat_hours) for c in raw.columns},
        index=raw.index)
    return zero, flat & ~zero


def mask_to_events(mask: pd.DataFrame, values: pd.DataFrame, source: str) -> pd.DataFrame:
    """Turn a boolean wide mask into a long table of events."""
    ti, bi = np.nonzero(mask.to_numpy())
    return pd.DataFrame({
        "timestamp": mask.index[ti],
        "building_id": mask.columns[bi].astype(str),
        "kwh": values.to_numpy()[ti, bi],
        "source": source,
        "score": np.nan,
    })


# ------------------------------------------------------------------- ML
def make_ml_features(long_df: pd.DataFrame, train_end=TRAIN_END, extra=()) -> pd.DataFrame:
    """Building-relative features. Only past values / training statistics are used."""
    slim = long_df[["timestamp", "building_id", "kwh", *extra]].copy()
    feats = build_anomaly_features(slim, train_end)

    # building mean over the training period (scale for the deviation feature)
    train = slim[slim["timestamp"] <= pd.Timestamp(train_end)]
    bmean = train.groupby("building_id", observed=True)["kwh"].mean().replace(0, np.nan)
    bid = feats["building_id"].astype("category")
    mean_per_cat = bmean.reindex(bid.cat.categories).to_numpy(dtype="float64")
    row_mean = mean_per_cat[bid.cat.codes.to_numpy()]
    feats["dev_rel"] = feats["dev_from_how_median"].to_numpy(dtype="float64") / row_mean

    hour = feats["timestamp"].dt.hour.to_numpy()
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24.0).astype("float32")
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24.0).astype("float32")
    feats["is_weekend"] = (feats["timestamp"].dt.day_of_week >= 5).astype("int8")

    feats["rolling_z"] = feats["rolling_z"].clip(-10, 10)
    feats["dev_rel"] = feats["dev_rel"].clip(-5, 5)
    feats["ratio_to_building_mean"] = feats["ratio_to_building_mean"].clip(0, 10)
    return feats


def fit_isolation_forest(feats: pd.DataFrame, train_end=TRAIN_END,
                         contamination: float = CONTAMINATION, max_rows: int = 300_000) -> dict:
    """Fit on the training period only. The threshold is the training score quantile."""
    train = feats[feats["timestamp"] <= pd.Timestamp(train_end)].dropna(subset=ML_FEATURES)
    if len(train) > max_rows:
        train = train.sample(max_rows, random_state=RANDOM_STATE)
    X = train[ML_FEATURES].to_numpy(dtype="float32")
    model = IsolationForest(n_estimators=200, max_samples=512, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X)
    train_scores = -model.score_samples(X)
    return {"model": model, "features": list(ML_FEATURES),
            "threshold": float(np.quantile(train_scores, 1 - contamination)),
            "contamination": contamination, "train_end": str(pd.Timestamp(train_end)),
            "n_train_rows": int(len(train))}


def score_features(bundle: dict, feats: pd.DataFrame) -> np.ndarray:
    """Anomaly score (higher = more unusual). NaN where features are missing."""
    X = feats[bundle["features"]].to_numpy(dtype="float32")
    ok = ~np.isnan(X).any(axis=1)
    score = np.full(len(feats), np.nan, dtype="float32")
    if ok.any():
        score[ok] = -bundle["model"].score_samples(X[ok])
    return score


# ----------------------------------------------------------- validation
def inject_faults(long_df: pd.DataFrame, start=TRAIN_END, seed: int = RANDOM_STATE,
                  spike_rate: float = 0.0015, drop_rate: float = 0.001,
                  stretch_rate: float = 0.0002) -> pd.DataFrame:
    """Copy of the data with synthetic faults after `start` and a `fault` column.

    fault: 0 none, 1 spike (x3 to x6), 2 drop (x0.02 to x0.10), 3 stretch (6 to 12 h at x2.5).
    """
    rng = np.random.default_rng(seed)
    df = (long_df[["timestamp", "building_id", "kwh"]]
          .sort_values(["building_id", "timestamp"]).reset_index(drop=True))
    kwh = df["kwh"].to_numpy(dtype="float64").copy()
    fault = np.zeros(len(df), dtype="int8")
    bid = pd.factorize(df["building_id"])[0]

    eligible = np.flatnonzero((df["timestamp"] > pd.Timestamp(start)).to_numpy() & ~np.isnan(kwh))
    n = len(eligible)
    pick = rng.permutation(eligible)
    ns, nd, nt = int(n * spike_rate), int(n * drop_rate), int(n * stretch_rate)
    spikes, drops = pick[:ns], pick[ns:ns + nd]
    starts = pick[ns + nd:ns + nd + nt]

    kwh[spikes] *= rng.uniform(3, 6, ns)
    fault[spikes] = 1
    kwh[drops] *= rng.uniform(0.02, 0.10, nd)
    fault[drops] = 2
    for s in starts:
        length = int(rng.integers(6, 13))
        seg = np.arange(s, min(s + length, len(df)))
        seg = seg[(bid[seg] == bid[s]) & ~np.isnan(kwh[seg]) & (fault[seg] == 0)]
        kwh[seg] *= 2.5
        fault[seg] = 3

    out = df.copy()
    out["kwh"] = kwh.astype("float32")
    out["fault"] = fault
    return out


def _prf(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int((y & pred).sum())
    fp = int((~y & pred).sum())
    fn = int((y & ~pred).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "flagged": int(pred.sum()), "true_faults": int(y.sum())}


def evaluate_detector(feats: pd.DataFrame, score: np.ndarray, threshold: float,
                      start=TRAIN_END, z_threshold: float = 4.0) -> dict:
    """Compare Isolation Forest with a rolling z-score rule on injected faults."""
    test = (feats["timestamp"] > pd.Timestamp(start)).to_numpy() & ~np.isnan(score)
    fault = feats["fault"].to_numpy()[test]
    y = fault > 0
    s = score[test]
    z = np.abs(np.nan_to_num(feats["rolling_z"].to_numpy(dtype="float64")[test], nan=0.0))
    pred_if, pred_z = s > threshold, z > z_threshold
    res = {
        "eval_rows": int(test.sum()),
        "isolation_forest": {**_prf(y, pred_if), "average_precision": round(float(average_precision_score(y, s)), 4)},
        "zscore_rule": {**_prf(y, pred_z), "average_precision": round(float(average_precision_score(y, z)), 4)},
        "recall_by_fault_type": {},
    }
    for code, name in FAULT_NAMES.items():
        m = fault == code
        if m.any():
            res["recall_by_fault_type"][name] = {
                "isolation_forest": round(float(pred_if[m].mean()), 4),
                "zscore_rule": round(float(pred_z[m].mean()), 4), "count": int(m.sum())}
    return res


# ------------------------------------------------------------------ run
def run_phase4(save: bool = True) -> dict:
    long_df = pd.read_parquet(PROC / "campus_long.parquet", columns=["timestamp", "building_id", "kwh"])
    long_df["timestamp"] = pd.to_datetime(long_df["timestamp"])
    buildings = [str(b) for b in long_df["building_id"].unique()]
    print(f"buildings: {len(buildings)} | rows: {len(long_df):,}")

    print("1/4 rules on the raw file (zeros, flat runs)")
    raw = load_raw_wide(RAW / "electricity.csv", buildings)
    zero, flat = rule_flags(raw)
    ev_zero = mask_to_events(zero, raw, "zero")
    ev_flat = mask_to_events(flat, raw, "flat_run")
    print(f"   zero readings: {len(ev_zero):,} in {ev_zero['building_id'].nunique()} buildings")
    print(f"   flat-run hours (>= {MIN_FLAT_HOURS} h): {len(ev_flat):,} in {ev_flat['building_id'].nunique()} buildings")

    print("2/4 features + Isolation Forest on the cleaned data")
    feats = make_ml_features(long_df)
    bundle = fit_isolation_forest(feats)
    score = score_features(bundle, feats)
    flagged = score > bundle["threshold"]
    valid = ~np.isnan(score)
    print(f"   trained on {bundle['n_train_rows']:,} rows | threshold {bundle['threshold']:.4f} | "
          f"flagged {int(flagged.sum()):,} of {int(valid.sum()):,} rows ({flagged.sum() / valid.sum() * 100:.2f}%)")

    ev_ml = feats.loc[flagged, ["timestamp", "building_id", "kwh"]].copy()
    ev_ml["building_id"] = ev_ml["building_id"].astype(str)
    ev_ml["source"] = "isolation_forest"
    ev_ml["score"] = score[flagged]

    print("3/4 validation with injected faults (test period)")
    inj = inject_faults(long_df)
    feats_i = make_ml_features(inj, extra=("fault",))
    score_i = score_features(bundle, feats_i)
    validation = evaluate_detector(feats_i, score_i, bundle["threshold"])
    for name in ("isolation_forest", "zscore_rule"):
        r = validation[name]
        print(f"   {name:17s} precision {r['precision']:.3f} recall {r['recall']:.3f} "
              f"f1 {r['f1']:.3f} AP {r['average_precision']:.3f}")
    for name, r in validation["recall_by_fault_type"].items():
        print(f"   recall {name:8s} IF {r['isolation_forest']:.3f} | z-rule {r['zscore_rule']:.3f} (n={r['count']})")

    print("4/4 summaries")
    per_b = (pd.DataFrame({"building_id": feats["building_id"].astype(str).to_numpy(),
                           "valid": valid, "flag": flagged})
             .groupby("building_id")[["valid", "flag"]].sum())
    per_b["ml_anomaly_rate_%"] = (per_b["flag"] / per_b["valid"] * 100).round(2)
    per_b["zero_hours"] = ev_zero.groupby("building_id").size().reindex(per_b.index).fillna(0).astype(int)
    per_b["flat_hours"] = ev_flat.groupby("building_id").size().reindex(per_b.index).fillna(0).astype(int)
    top = per_b.sort_values("ml_anomaly_rate_%", ascending=False).head(10)
    print(top[["flag", "ml_anomaly_rate_%", "zero_hours", "flat_hours"]].to_string())

    events = pd.concat([ev_ml, ev_zero, ev_flat], ignore_index=True)
    events["kwh"] = events["kwh"].astype("float32")
    events["score"] = events["score"].astype("float32")
    events = events.sort_values(["timestamp", "building_id"]).reset_index(drop=True)

    metrics = {"train_end": bundle["train_end"], "contamination": CONTAMINATION,
               "threshold": bundle["threshold"], "validation": validation,
               "rules": {"zero_readings": int(len(ev_zero)), "flat_run_hours": int(len(ev_flat)),
                         "min_flat_hours": MIN_FLAT_HOURS},
               "ml_flagged_rows": int(flagged.sum()), "ml_valid_rows": int(valid.sum()),
               "events_total": int(len(events))}
    if save:
        MODELS.mkdir(exist_ok=True)
        PROC.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODELS / "isolation_forest.joblib")
        (MODELS / "anomaly_metrics.json").write_text(json.dumps(metrics, indent=2))
        events.to_parquet(PROC / "anomalies.parquet")
        per_b.reset_index().to_csv(PROC / "anomaly_summary_by_building.csv", index=False)
        print("saved: models/isolation_forest.joblib, models/anomaly_metrics.json, "
              "data/processed/anomalies.parquet, data/processed/anomaly_summary_by_building.csv")
    return metrics


if __name__ == "__main__":
    run_phase4()
