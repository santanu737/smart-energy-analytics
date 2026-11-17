"""Data and model access for the API. Everything is loaded once at startup.

Optional parts (anomalies, forecast model, metrics) may be missing if the
Phase 4/5 scripts have not been run yet. The service reports what is loaded
in `self.loaded`, and the API answers 503 with a clear message for them.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src import config

ANOMALIES_PARQUET = config.PROCESSED_DIR / "anomalies.parquet"
FORECAST_MODEL = config.MODELS_DIR / "forecast_model.joblib"
FORECAST_METRICS = config.MODELS_DIR / "forecast_metrics.json"
ANOMALY_METRICS = config.MODELS_DIR / "anomaly_metrics.json"
HORIZONS = (1, 6, 24)
WEATHER_COLS = ["airTemperature", "cloudCoverage", "dewTemperature", "precipDepth1HR",
                "precipDepth6HR", "seaLvlPressure", "windDirection", "windSpeed"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


class NotAvailable(Exception):
    """A part of the system (model, anomalies) has not been built yet."""


def fmt_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.strftime("%Y-%m-%d %H:%M:%S")


def to_records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe records: NaN and inf become None, timestamps become strings."""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = fmt_ts(d[c])
    d = d.astype(object)
    d = d.where(pd.notna(d), None)
    return d.to_dict(orient="records")


def clean_float(x):
    return None if x is None or not np.isfinite(x) else float(x)


class DataService:
    def __init__(self):
        self.loaded: dict[str, bool] = {}
        self.replay_index = 0
        self.load()

    # ------------------------------------------------------------ loading
    def load(self):
        print("Loading data for the API ...")
        for p in (config.CAMPUS_TOTAL_PARQUET, config.CAMPUS_LONG_PARQUET, config.BUILDINGS_META_CSV):
            if not p.exists():
                raise FileNotFoundError(f"{p} not found. Run: python -m src.data_loader --build")

        self.campus_total = pd.read_parquet(config.CAMPUS_TOTAL_PARQUET)
        self.campus_total["timestamp"] = pd.to_datetime(self.campus_total["timestamp"])
        long = pd.read_parquet(config.CAMPUS_LONG_PARQUET, columns=["timestamp", "building_id", "kwh"])
        long["timestamp"] = pd.to_datetime(long["timestamp"])
        self.long_df = long
        self.by_building = {str(k): g.reset_index(drop=True)
                            for k, g in long.groupby("building_id", observed=True)}
        self.meta = pd.read_csv(config.BUILDINGS_META_CSV)
        self.meta["building_id"] = self.meta["building_id"].astype(str)
        self.loaded.update(campus_total=True, campus_long=True, buildings_meta=True)

        self.stats = self._building_stats()
        self.loaded["building_stats"] = True
        self.weather = self._load_weather()
        self.loaded["weather"] = self.weather is not None

        self.anomalies = None
        if ANOMALIES_PARQUET.exists():
            a = pd.read_parquet(ANOMALIES_PARQUET)
            a["timestamp"] = pd.to_datetime(a["timestamp"])
            self.anomalies = a
        self.loaded["anomalies"] = self.anomalies is not None

        self.forecast_metrics = json.loads(FORECAST_METRICS.read_text()) if FORECAST_METRICS.exists() else None
        self.anomaly_metrics = json.loads(ANOMALY_METRICS.read_text()) if ANOMALY_METRICS.exists() else None
        self.bundle, self.fc_tables = None, {}
        if FORECAST_MODEL.exists():
            self._load_forecast()
        self.loaded["forecast_model"] = self.bundle is not None
        self.loaded["forecast_metrics"] = self.forecast_metrics is not None
        self.loaded["anomaly_metrics"] = self.anomaly_metrics is not None
        print("Loaded:", {k: v for k, v in self.loaded.items()})

    def _load_weather(self):
        weather_path = (config.API_WEATHER_CSV
                        if config.API_WEATHER_CSV.exists() else config.WEATHER_CSV)
        if not weather_path.exists():
            return None
        keep = set(WEATHER_COLS) | {"timestamp", "site_id"}
        w = pd.read_csv(weather_path, usecols=lambda c: c in keep)
        w = w[w["site_id"] == config.CAMPUS_SITE_ID].drop(columns="site_id")
        w["timestamp"] = pd.to_datetime(w["timestamp"])
        return w.sort_values("timestamp").reset_index(drop=True)

    def _load_forecast(self):
        from src import forecast as fc
        self.bundle = joblib.load(FORECAST_MODEL)
        total, weather = fc.load_inputs()
        self.fc_tables = {h: fc.make_table(total, weather, h) for h in self.bundle["models"]}

    def _building_stats(self) -> pd.DataFrame:
        l = self.long_df
        k = l["kwh"].astype("float64")
        hour = l["timestamp"].dt.hour.to_numpy()
        weekend = (l["timestamp"].dt.day_of_week >= 5).to_numpy()
        bid = l["building_id"]
        g = k.groupby(bid, observed=True)
        st = pd.DataFrame({"total_kwh": g.sum(), "mean_kwh_per_h": g.mean(), "hours": g.count()})
        off = k.where((hour <= 5) | weekend).groupby(bid, observed=True).mean()
        day = k.where((~weekend) & (hour >= 8) & (hour <= 17)).groupby(bid, observed=True).mean()
        st["baseload_ratio"] = off / day.replace(0, np.nan)   # night + weekend load / weekday daytime load
        st.index = st.index.astype(str)
        m = self.meta.set_index("building_id")
        for c in ("primaryspaceusage", "sqm", "yearbuilt", "numberoffloors"):
            st[c] = m[c] if c in m.columns else np.nan
        st["eui_annual"] = st["mean_kwh_per_h"] * 8760 / st["sqm"].replace(0, np.nan)
        st.index.name = "building_id"
        return st.reset_index()

    # ----------------------------------------------------------- helpers
    @staticmethod
    def _slice(df: pd.DataFrame, start, end) -> pd.DataFrame:
        if start is not None:
            df = df[df["timestamp"] >= start]
        if end is not None:
            df = df[df["timestamp"] <= end]
        return df

    def has_building(self, building_id: str) -> bool:
        return building_id in self.by_building

    # --------------------------------------------------------- endpoints
    def get_summary(self, start=None, end=None) -> dict:
        df = self._slice(self.campus_total, start, end)
        ok = df.dropna(subset=["kwh"])
        if ok.empty:
            return {}
        peak = ok.loc[ok["kwh"].idxmax()]
        n_anom = None
        if self.anomalies is not None:
            n_anom = int(len(self._slice(self.anomalies, start, end)))
        return {"total_kwh": float(ok["kwh"].sum()), "avg_load_kw": float(ok["kwh"].mean()),
                "peak_load_kw": float(peak["kwh"]), "peak_time": str(peak["timestamp"]),
                "hours_with_data": int(len(ok)), "anomaly_count": n_anom,
                "anomalies_available": self.anomalies is not None,
                "building_count": int(len(self.by_building)),
                "start_date": str(df["timestamp"].min()), "end_date": str(df["timestamp"].max())}

    def get_campus_history(self, start=None, end=None, resolution="hourly") -> list[dict]:
        df = self._slice(self.campus_total, start, end)[["timestamp", "kwh", "n_reporting"]]
        if resolution == "daily":
            d = df.set_index("timestamp").resample("D").agg({"kwh": "mean", "n_reporting": "mean"})
            d["kwh"] = d["kwh"] * 24      # average hourly load x 24, so missing hours do not shrink the day
            df = d.reset_index()
        return to_records(df)

    def get_building_history(self, building_id, start=None, end=None, resolution="hourly") -> list[dict]:
        df = self._slice(self.by_building[building_id], start, end)
        if resolution == "daily":
            d = df.set_index("timestamp")["kwh"].resample("D").mean() * 24
            df = d.rename("kwh").reset_index()
        return to_records(df[["timestamp", "kwh"]])

    def get_building_profile(self, building_id) -> dict:
        df = self.by_building[building_id]
        p = (df.assign(dow=df["timestamp"].dt.day_of_week, hour=df["timestamp"].dt.hour)
             .pivot_table(index="dow", columns="hour", values="kwh", aggfunc="mean")
             .reindex(index=range(7), columns=range(24)))
        return {"days": DAYS, "hours": list(range(24)),
                "values": [[clean_float(v) for v in row] for row in p.to_numpy(dtype="float64")]}

    def get_buildings(self) -> list[dict]:
        cols = ["building_id", "primaryspaceusage", "sqm", "yearbuilt", "numberoffloors",
                "total_kwh", "eui_annual", "baseload_ratio"]
        return to_records(self.stats[cols])

    def get_ranking(self, metric: str, limit: int) -> list[dict]:
        col = {"total": "total_kwh", "eui": "eui_annual", "baseload": "baseload_ratio"}[metric]
        d = self.stats.dropna(subset=[col]).sort_values(col, ascending=False).head(limit)
        return to_records(d[["building_id", "primaryspaceusage", "sqm", "total_kwh",
                             "eui_annual", "baseload_ratio"]])

    def get_usage_types(self) -> list[dict]:
        g = self.stats.groupby("primaryspaceusage", dropna=False).agg(
            buildings=("building_id", "size"), total_kwh=("total_kwh", "sum"))
        g["share_%"] = (g["total_kwh"] / g["total_kwh"].sum() * 100).round(1)
        return to_records(g.sort_values("total_kwh", ascending=False).reset_index())

    def get_anomalies(self, building_id=None, start=None, end=None, source=None, limit=50) -> list[dict]:
        if self.anomalies is None:
            raise NotAvailable("Anomaly results not found. Run: python -m src.anomaly")
        a = self._slice(self.anomalies, start, end)
        if building_id:
            a = a[a["building_id"] == building_id]
        if source:
            a = a[a["source"] == source]
        return to_records(a.sort_values("timestamp", ascending=False).head(limit))

    def get_weather(self, start=None, end=None, resolution="hourly") -> list[dict]:
        if self.weather is None:
            raise NotAvailable("weather.csv not found")
        w = self._slice(self.weather, start, end)
        if resolution == "daily":
            w = w.set_index("timestamp").resample("D").mean().reset_index()
        return to_records(w)

    def get_forecast(self, horizon: int, as_of: pd.Timestamp | None, points: int) -> dict:
        if self.bundle is None or horizon not in self.fc_tables:
            raise NotAvailable("Forecast model not found. Run: python -m src.forecast")
        t = self.fc_tables[horizon]
        if as_of is None:
            as_of = pd.Timestamp(self.bundle["val_end"]) + pd.Timedelta(hours=1)   # first test origin
        sel = t[t["timestamp"] >= as_of].head(points)
        if sel.empty:
            return {}
        cols = self.bundle["features"][horizon]
        pred = self.bundle["models"][horizon].predict(sel[cols])
        out = pd.DataFrame({"origin": sel["timestamp"].to_numpy(),
                            "target_timestamp": sel["target_timestamp"].to_numpy(),
                            "forecast": pred, "actual": sel["target"].to_numpy(),
                            "naive_24": sel["naive_24"].to_numpy()})
        return {"horizon": horizon, "model": "HistGradientBoosting with weather",
                "metrics_test": (self.forecast_metrics or {}).get(str(horizon)),
                "points": to_records(out)}

    def get_model_info(self) -> dict:
        return {"forecast": self.forecast_metrics, "anomaly": self.anomaly_metrics}

    def get_replay_next(self) -> dict:
        ct = self.campus_total
        if self.replay_index >= len(ct):
            self.replay_index = 0
        row = ct.iloc[[self.replay_index]]
        self.replay_index += 1
        return to_records(row[["timestamp", "kwh", "n_reporting"]])[0]

    def reset_replay(self) -> dict:
        self.replay_index = 0
        return {"status": "reset"}
