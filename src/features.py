"""Feature engineering utilities for forecasting and anomaly detection."""

import numpy as np
import pandas as pd

WEATHER_COLUMNS = [
    "airTemperature", "cloudCoverage", "dewTemperature", "precipDepth1HR",
    "precipDepth6HR", "seaLvlPressure", "windDirection", "windSpeed",
]


def add_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Add calendar features and cyclical encodings from a timestamp column."""
    out = df.copy()
    dt = out[ts_col].dt
    dow = dt.day_of_week
    out["hour"] = dt.hour
    out["dayofweek"] = dow
    out["month"] = dt.month
    out["is_weekend"] = dow.isin([5, 6]).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return out


def add_weather_features(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Add temperature lags and a rolling mean using only past values."""
    out = weather_df.copy()
    if "timestamp" in out.columns:
        out = out.sort_values("timestamp")

    temp = out["airTemperature"]
    out["airTemperature_lag_1"] = temp.shift(1)
    out["airTemperature_lag_3"] = temp.shift(3)
    out["airTemperature_lag_24"] = temp.shift(24)
    out["airTemperature_rolling_mean_24"] = temp.shift(1).rolling(24).mean()
    return out


def add_target_lags(series_df: pd.DataFrame, target: str = "kwh") -> pd.DataFrame:
    """Add lags and rolling statistics of the target using past values only.

    Convention: at origin hour t, `{target}_lag_k` is the value at t - k hours.
    Rolling statistics cover t-1 and earlier. The origin-hour value itself is
    the plain `{target}` column.
    """
    out = series_df.copy()
    if "timestamp" in out.columns:
        out = out.sort_values("timestamp")

    s = out[target]
    for k in (1, 2, 3, 24, 48, 168):
        out[f"{target}_lag_{k}"] = s.shift(k)

    s_shift = s.shift(1)
    out[f"{target}_rolling_mean_3"] = s_shift.rolling(3).mean()
    out[f"{target}_rolling_mean_24"] = s_shift.rolling(24).mean()
    out[f"{target}_rolling_std_24"] = s_shift.rolling(24).std()
    return out


def build_forecast_table(campus_total: pd.DataFrame, weather: pd.DataFrame,
                         horizon: int = 1) -> pd.DataFrame:
    """Build the supervised table for campus-load forecasting.

    Each row is a forecast origin t. The target is campus kwh at t + horizon
    (`target`, stamped `target_timestamp`). Calendar features come from
    target_timestamp. Weather and kwh features use only values at or before t.
    naive_24 / naive_168 are the values 24 h / 168 h before the TARGET hour.

    Rows: the first 168 origins (warm-up for the longest lag) are dropped. The
    last `horizon` origins are kept with target = NaN (useful for live
    forecasting); training code must drop rows where target is NaN. Rows with
    NaN kwh or NaN target in the middle are NOT dropped.
    """
    if not (1 <= horizon <= 24):
        raise ValueError("horizon must be between 1 and 24")

    # weather: keep only timestamp + weather columns + gap flags
    gap_flags = [c for c in weather.columns if c.endswith("_gap_flag")]
    keep = ["timestamp"] + [c for c in WEATHER_COLUMNS if c in weather.columns] + gap_flags
    w = weather[keep].drop_duplicates().sort_values("timestamp")
    if w["timestamp"].duplicated().any():
        raise ValueError("weather has several different rows for the same timestamp")

    ct = campus_total.sort_values("timestamp").reset_index(drop=True)
    if ct["timestamp"].duplicated().any():
        raise ValueError("campus_total has duplicated timestamps")
    if not ct["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("campus_total must be a regular hourly series")
    if len(ct) <= 168 + horizon:
        raise ValueError("series too short for the 168 h warm-up plus the horizon")

    df = ct.merge(w, on="timestamp", how="left").reset_index(drop=True)

    df["target_timestamp"] = df["timestamp"] + pd.to_timedelta(horizon, unit="h")
    time_feats = add_time_features(
        pd.DataFrame({"timestamp": df["target_timestamp"]}), "timestamp"
    ).drop(columns=["timestamp"])
    df = pd.concat([df, time_feats], axis=1)

    df = add_weather_features(df)
    df = add_target_lags(df, target="kwh")

    df["target"] = df["kwh"].shift(-horizon)
    df["naive_24"] = df["kwh"].shift(24 - horizon)
    df["naive_168"] = df["kwh"].shift(168 - horizon)

    return df.iloc[168:].reset_index(drop=True)


def build_anomaly_features(long_df: pd.DataFrame,
                           train_end: "str | pd.Timestamp") -> pd.DataFrame:
    """Building-level features for anomaly detection, using training stats only.

    Adds dev_from_how_median (vs the building's median for the same hour of
    week, training period only), ratio_to_building_mean (NaN, never inf) and
    rolling_z (168 h window of past values, min_periods=120).
    """
    out = long_df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

    train_end_dt = pd.to_datetime(train_end)
    train_mask = out["timestamp"] <= train_end_dt

    out["_how"] = out["timestamp"].dt.day_of_week * 24 + out["timestamp"].dt.hour

    train_df = out[train_mask]
    how_medians = (train_df.groupby(["building_id", "_how"], observed=True)["kwh"]
                   .median().rename("_how_median").reset_index())
    out = out.merge(how_medians, on=["building_id", "_how"], how="left")
    out["dev_from_how_median"] = out["kwh"] - out["_how_median"]

    b_means = (train_df.groupby("building_id", observed=True)["kwh"].mean()
               .replace(0, np.nan).rename("_b_mean").reset_index())
    out = out.merge(b_means, on="building_id", how="left")
    out["ratio_to_building_mean"] = (out["kwh"] / out["_b_mean"]).replace(
        [np.inf, -np.inf], np.nan)

    # rolling z-score per building, past values only (shift(1) inside each building)
    grp = out.groupby("building_id", observed=True)["kwh"]
    past = grp.shift(1)
    past_grp = past.groupby(out["building_id"], observed=True)
    rmean = past_grp.transform(lambda s: s.rolling(168, min_periods=120).mean())
    rstd = past_grp.transform(lambda s: s.rolling(168, min_periods=120).std())
    out["rolling_z"] = (out["kwh"] - rmean) / rstd.replace(0, np.nan)

    out = out.drop(columns=["_how", "_how_median", "_b_mean"])

    if "primaryspaceusage" in out.columns:
        if not isinstance(out["primaryspaceusage"].dtype, pd.CategoricalDtype):
            out["primaryspaceusage"] = out["primaryspaceusage"].astype("category")
        out["primaryspaceusage"] = out["primaryspaceusage"].cat.codes

    return out
