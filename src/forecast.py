"""Phase 5: campus load forecasting (1, 6 and 24 hours ahead).

Rows are forecast origins t. The target is campus kWh at t + horizon.
Models: persistence, naive_24, naive_168 (baselines) and HistGradientBoosting
with and without weather. Time-based split on the TARGET timestamp, no shuffling:
  train       : up to 2017-03-31 23:00
  validation  : 2017-04-01 to 2017-06-30  (picks the hyper-parameters)
  test        : 2017-07-01 onwards        (final numbers, model refit on train+val)
Hours where fewer than MIN_REPORTING buildings reported are not used as targets.

Run:  python -m src.forecast
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import build_forecast_table

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
IMAGES = ROOT / "docs" / "images"

TRAIN_END = pd.Timestamp("2017-03-31 23:00:00")
VAL_END = pd.Timestamp("2017-06-30 23:00:00")
MIN_REPORTING = 125
HORIZONS = (1, 6, 24)
RANDOM_STATE = 42

WEATHER = ["airTemperature", "cloudCoverage", "dewTemperature", "precipDepth1HR",
           "precipDepth6HR", "seaLvlPressure", "windDirection", "windSpeed"]
NOT_FEATURES = {"timestamp", "target_timestamp", "target", "naive_24", "naive_168",
                "n_reporting", "target_n_reporting", "persist"}
GRID = [
    {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15},
    {"learning_rate": 0.05, "max_iter": 600, "max_leaf_nodes": 15},
    {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 31},
    {"learning_rate": 0.05, "max_iter": 600, "max_leaf_nodes": 31},
]


def is_weather_col(c: str) -> bool:
    return c in WEATHER or c.endswith("_gap_flag") or c.startswith("airTemperature_")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """campus_total plus one weather row per hour (from the processed long file)."""
    import pyarrow.parquet as pq
    total = pd.read_parquet(PROC / "campus_total.parquet")
    total["timestamp"] = pd.to_datetime(total["timestamp"])
    names = set(pq.read_schema(PROC / "campus_long.parquet").names)
    cols = ["timestamp"] + [c for c in WEATHER + [f"{w}_gap_flag" for w in WEATHER] if c in names]
    wx = pd.read_parquet(PROC / "campus_long.parquet", columns=cols)
    wx["timestamp"] = pd.to_datetime(wx["timestamp"])
    return total, wx.drop_duplicates("timestamp")


def make_table(total: pd.DataFrame, weather: pd.DataFrame, horizon: int) -> pd.DataFrame:
    t = build_forecast_table(total[["timestamp", "kwh", "n_reporting"]], weather, horizon)
    rep = total.set_index("timestamp")["n_reporting"]
    t["target_n_reporting"] = t["target_timestamp"].map(rep)
    t["persist"] = t["kwh"]
    return t


def split_table(t: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Usable rows split by target timestamp. Returns train / val / test."""
    ok = (t["target"].notna() & t["kwh"].notna() & t["naive_24"].notna() & t["naive_168"].notna()
          & (t["target_n_reporting"] >= MIN_REPORTING))
    u = t[ok]
    ts = u["target_timestamp"]
    return {"train": u[ts <= TRAIN_END],
            "val": u[(ts > TRAIN_END) & (ts <= VAL_END)],
            "test": u[ts > VAL_END]}


def feature_cols(t: pd.DataFrame, use_weather: bool = True) -> list[str]:
    cols = [c for c in t.columns if c not in NOT_FEATURES]
    return cols if use_weather else [c for c in cols if not is_weather_col(c)]


def regression_metrics(y, p) -> dict:
    y, p = np.asarray(y, dtype="float64"), np.asarray(p, dtype="float64")
    return {"mae": round(float(mean_absolute_error(y, p)), 1),
            "rmse": round(float(mean_squared_error(y, p) ** 0.5), 1),
            "r2": round(float(r2_score(y, p)), 4),
            "mape_%": round(float(np.mean(np.abs((y - p) / y)) * 100), 2)}


def _fit(params: dict, X, y) -> HistGradientBoostingRegressor:
    m = HistGradientBoostingRegressor(random_state=RANDOM_STATE, l2_regularization=1.0, **params)
    return m.fit(X, y)


def fit_best(parts: dict, cols: list[str], grid: list[dict]):
    """Pick the grid entry with the lowest validation MAE, refit on train+val."""
    tr, va = parts["train"], parts["val"]
    scores = []
    for params in grid:
        m = _fit(params, tr[cols], tr["target"])
        scores.append(mean_absolute_error(va["target"], m.predict(va[cols])))
    best = grid[int(np.argmin(scores))]
    both = pd.concat([tr, va])
    return _fit(best, both[cols], both["target"]), best, float(min(scores))


def run_forecast(total: pd.DataFrame, weather: pd.DataFrame,
                 horizons=HORIZONS, grid=GRID) -> dict:
    """Train and evaluate every horizon. Returns models, metrics and test predictions."""
    out = {"models": {}, "features": {}, "metrics": {}, "preds": {}, "params": {}, "tables": {}}
    for h in horizons:
        t = make_table(total, weather, h)
        parts = split_table(t)
        te = parts["test"]
        res = {"rows": {k: int(len(v)) for k, v in parts.items()},
               "persistence": regression_metrics(te["target"], te["persist"]),
               "naive_24": regression_metrics(te["target"], te["naive_24"]),
               "naive_168": regression_metrics(te["target"], te["naive_168"])}
        cols_w = feature_cols(t, True)
        cols_n = feature_cols(t, False)
        m_w, p_w, v_w = fit_best(parts, cols_w, grid)
        m_n, p_n, v_n = fit_best(parts, cols_n, grid)
        pred_w, pred_n = m_w.predict(te[cols_w]), m_n.predict(te[cols_n])
        res["hgb_with_weather"] = regression_metrics(te["target"], pred_w)
        res["hgb_no_weather"] = regression_metrics(te["target"], pred_n)
        res["best_params"] = p_w
        res["val_mae_with_weather"] = round(v_w, 1)
        res["val_mae_no_weather"] = round(v_n, 1)
        out["models"][h], out["features"][h] = m_w, cols_w
        out["metrics"][h] = res
        out["params"][h] = p_w
        out["preds"][h] = pd.DataFrame({"target_timestamp": te["target_timestamp"].to_numpy(),
                                        "actual": te["target"].to_numpy(), "hgb": pred_w,
                                        "hgb_no_weather": pred_n, "naive_24": te["naive_24"].to_numpy(),
                                        "persistence": te["persist"].to_numpy()})
        out["tables"][h] = te
    return out


def make_plots(result: dict, out_dir: Path = IMAGES) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    hs = [h for h in (1, 24) if h in result["preds"]] or list(result["preds"])[:1]

    fig, axes = plt.subplots(len(hs), 1, figsize=(11, 3.6 * len(hs)), squeeze=False)
    for ax, h in zip(axes[:, 0], hs):
        p = result["preds"][h].set_index("target_timestamp")
        week = p.iloc[:168] if len(p) >= 168 else p
        ax.plot(week.index, week["actual"], label="actual", lw=1.6)
        ax.plot(week.index, week["hgb"], "--", label="forecast (HGB)", lw=1.4)
        ax.plot(week.index, week["naive_24"], ":", label="naive 24h", lw=1.1)
        ax.set_title(f"Campus load: {h}h-ahead forecast vs actual (first test week)")
        ax.set_ylabel("kWh per hour"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); f = out_dir / "forecast_vs_actual.png"; fig.savefig(f, dpi=130); plt.close(fig)
    saved.append(str(f))

    h = hs[0]
    p = result["preds"][h]
    hour = pd.to_datetime(p["target_timestamp"]).dt.hour
    err = pd.DataFrame({"hgb": (p["actual"] - p["hgb"]).abs(),
                        "naive_24": (p["actual"] - p["naive_24"]).abs(), "hour": hour}).groupby("hour").mean()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    err.plot(ax=ax, marker="o"); ax.set_title(f"Mean absolute error by hour of day ({h}h ahead)")
    ax.set_ylabel("MAE (kWh)"); ax.grid(alpha=0.3)
    fig.tight_layout(); f = out_dir / "forecast_error_by_hour.png"; fig.savefig(f, dpi=130); plt.close(fig)
    saved.append(str(f))

    te, cols, model = result["tables"][h], result["features"][h], result["models"][h]
    sample = te.sample(min(1500, len(te)), random_state=RANDOM_STATE)
    imp = permutation_importance(model, sample[cols], sample["target"], n_repeats=3,
                                 random_state=RANDOM_STATE, scoring="neg_mean_absolute_error")
    s = pd.Series(imp.importances_mean, index=cols).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    s.plot.barh(ax=ax); ax.set_title(f"Permutation importance ({h}h model, top 15)")
    ax.set_xlabel("increase in MAE when shuffled")
    fig.tight_layout(); f = out_dir / "forecast_feature_importance.png"; fig.savefig(f, dpi=130); plt.close(fig)
    saved.append(str(f))
    return saved


def run_phase5(save: bool = True) -> dict:
    total, weather = load_inputs()
    print(f"campus_total rows: {len(total):,} | weather columns: {len(weather.columns) - 1}")
    result = run_forecast(total, weather)
    rows = []
    for h, m in result["metrics"].items():
        for name in ("persistence", "naive_24", "naive_168", "hgb_no_weather", "hgb_with_weather"):
            rows.append({"horizon_h": h, "model": name, **m[name]})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    for h, m in result["metrics"].items():
        print(f"h={h}: rows {m['rows']} | best params {m['best_params']}")
    if save:
        MODELS.mkdir(exist_ok=True)
        joblib.dump({"models": result["models"], "features": result["features"],
                     "train_end": str(TRAIN_END), "val_end": str(VAL_END),
                     "min_reporting": MIN_REPORTING}, MODELS / "forecast_model.joblib")
        (MODELS / "forecast_metrics.json").write_text(json.dumps(result["metrics"], indent=2))
        files = make_plots(result)
        print("saved: models/forecast_model.joblib, models/forecast_metrics.json,", ", ".join(Path(f).name for f in files))
    return result["metrics"]


if __name__ == "__main__":
    run_phase5()
