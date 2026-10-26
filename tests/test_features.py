"""Tests for feature engineering functions in src/features.py."""

import pytest
import pandas as pd
import numpy as np
from src.features import (
    add_time_features,
    add_weather_features,
    add_target_lags,
    build_forecast_table,
    build_anomaly_features
)

def test_add_time_features():
    dates = pd.date_range(start='2016-01-01', periods=48, freq='h')
    df = pd.DataFrame({'timestamp': dates, 'val': range(48)})
    res = add_time_features(df, 'timestamp')
    
    assert 'hour' in res.columns
    assert 'dayofweek' in res.columns
    assert 'month' in res.columns
    assert 'is_weekend' in res.columns
    assert 'hour_sin' in res.columns
    assert 'hour_cos' in res.columns
    assert 'dow_sin' in res.columns
    assert 'dow_cos' in res.columns
    
    assert res['hour'].between(0, 23).all()
    assert res['dayofweek'].between(0, 6).all()
    assert res['month'].between(1, 12).all()
    assert res['hour_sin'].between(-1, 1).all()
    assert res['hour_cos'].between(-1, 1).all()

def test_lag_24_and_no_future_leakage():
    dates = pd.date_range(start='2016-01-01', periods=50, freq='h')
    df = pd.DataFrame({'timestamp': dates, 'kwh': range(50)})
    res = add_target_lags(df, target='kwh')
    
    assert res.loc[30, 'kwh_lag_24'] == res.loc[6, 'kwh']
    
    df1 = pd.DataFrame({'timestamp': dates, 'kwh': range(50)})
    df2 = df1.copy()
    df2.loc[26:, 'kwh'] = 999
    
    res1 = add_target_lags(df1, target='kwh')
    res2 = add_target_lags(df2, target='kwh')
    
    cols_to_check = [c for c in res1.columns if c != 'timestamp']
    pd.testing.assert_series_equal(res1.loc[:20, cols_to_check].stack(), res2.loc[:20, cols_to_check].stack())

def test_dev_from_how_median_training_window():
    dates = pd.date_range(start='2016-01-01', periods=200, freq='h')
    data = []
    for b in ['b1', 'b2']:
        for i, d in enumerate(dates):
            data.append({
                'timestamp': d, 
                'building_id': b, 
                'kwh': float(i), 
                'primaryspaceusage': 'Office', 
                'sqm': 100.0, 
                'yearbuilt': 2000, 
                'numberoffloors': 2
            })
            
    df1 = pd.DataFrame(data)
    df2 = df1.copy()
    
    train_end = dates[100]
    df2.loc[df2['timestamp'] > train_end, 'kwh'] = 9999.0
    
    feat1 = build_anomaly_features(df1, train_end)
    feat2 = build_anomaly_features(df2, train_end)
    
    train_subset1 = feat1[feat1['timestamp'] <= train_end]
    train_subset2 = feat2[feat2['timestamp'] <= train_end]
    
    pd.testing.assert_series_equal(train_subset1['dev_from_how_median'], train_subset2['dev_from_how_median'])
    pd.testing.assert_series_equal(train_subset1['ratio_to_building_mean'], train_subset2['ratio_to_building_mean'])

def test_build_forecast_table_no_collisions_and_horizons():
    dates = pd.date_range(start='2016-01-01', periods=400, freq='h')
    campus_total = pd.DataFrame({
        'timestamp': dates,
        'kwh': np.linspace(100.0, 500.0, 400),
        'n_reporting': 10
    })
    
    weather = pd.DataFrame({
        'timestamp': dates,
        'kwh': 999.9,
        'building_id': 'extra',
        'airTemperature': 20.0,
        'cloudCoverage': 0.0,
        'dewTemperature': 10.0,
        'precipDepth1HR': 0.0,
        'precipDepth6HR': 0.0,
        'seaLvlPressure': 1013.0,
        'windDirection': 180.0,
        'windSpeed': 5.0,
        'extra_col': 123
    })
    
    for h in [1, 6]:
        ft = build_forecast_table(campus_total, weather, horizon=h)
        assert 'kwh_x' not in ft.columns
        assert 'kwh_y' not in ft.columns
        assert len(ft) == 400 - 168
        
        orig_idx = 200
        t_val = ft.loc[ft.index == orig_idx, 'target'].values[0]
        expected_val = campus_total.loc[orig_idx + 168 + h, 'kwh']
        assert t_val == expected_val
        
        naive_val = ft.loc[ft.index == orig_idx, 'naive_24'].values[0]
        expected_naive = campus_total.loc[orig_idx + 168 + h - 24, 'kwh']
        assert naive_val == expected_naive

def test_forecast_table_leakage():
    dates = pd.date_range(start='2016-01-01', periods=400, freq='h')
    campus_total1 = pd.DataFrame({
        'timestamp': dates,
        'kwh': np.linspace(100.0, 500.0, 400),
        'n_reporting': 10
    })
    campus_total2 = campus_total1.copy()
    campus_total2.loc[250:, 'kwh'] = 9999.0
    
    weather1 = pd.DataFrame({
        'timestamp': dates,
        'airTemperature': 20.0,
        'cloudCoverage': 0.0,
        'dewTemperature': 10.0,
        'precipDepth1HR': 0.0,
        'precipDepth6HR': 0.0,
        'seaLvlPressure': 1013.0,
        'windDirection': 180.0,
        'windSpeed': 5.0
    })
    weather2 = weather1.copy()
    weather2.loc[250:, 'airTemperature'] = 50.0
    
    ft1 = build_forecast_table(campus_total1, weather1, horizon=1)
    ft2 = build_forecast_table(campus_total2, weather2, horizon=1)
    
    row_idx = 200 - 168
    cols_to_check = [c for c in ft1.columns if c not in ['target', 'naive_24', 'naive_168', 'timestamp', 'target_timestamp']]
    
    pd.testing.assert_series_equal(
        ft1.loc[row_idx, cols_to_check],
        ft2.loc[row_idx, cols_to_check]
    )

def test_forecast_table_nan_middle():
    dates = pd.date_range(start='2016-01-01', periods=400, freq='h')
    campus_total = pd.DataFrame({
        'timestamp': dates,
        'kwh': np.linspace(100.0, 500.0, 400),
        'n_reporting': 10
    })
    campus_total.loc[200, 'kwh'] = np.nan
    
    weather = pd.DataFrame({
        'timestamp': dates,
        'airTemperature': 20.0,
        'cloudCoverage': 0.0,
        'dewTemperature': 10.0,
        'precipDepth1HR': 0.0,
        'precipDepth6HR': 0.0,
        'seaLvlPressure': 1013.0,
        'windDirection': 180.0,
        'windSpeed': 5.0
    })
    
    ft = build_forecast_table(campus_total, weather, horizon=1)
    assert len(ft) == 400 - 168

def test_rolling_z_no_future_leakage():
    dates = pd.date_range(start='2016-01-01', periods=200, freq='h')
    df1 = pd.DataFrame({
        'timestamp': dates,
        'building_id': 'b1',
        'kwh': np.linspace(10.0, 100.0, 200),
        'primaryspaceusage': 'Office',
        'sqm': 100.0,
        'yearbuilt': 2000,
        'numberoffloors': 1
    })
    df2 = df1.copy()
    df2.loc[150:, 'kwh'] = 9999.0
    
    feat1 = build_anomaly_features(df1, train_end=dates[50])
    feat2 = build_anomaly_features(df2, train_end=dates[50])
    
    assert not np.isnan(feat1.loc[140, 'rolling_z'])
    assert feat1.loc[140, 'rolling_z'] == feat2.loc[140, 'rolling_z']

def test_ratio_to_building_mean_zero():
    dates = pd.date_range(start='2016-01-01', periods=50, freq='h')
    df = pd.DataFrame({
        'timestamp': dates,
        'building_id': 'b_zero',
        'kwh': 0.0,
        'primaryspaceusage': 'Office',
        'sqm': 100.0,
        'yearbuilt': 2000,
        'numberoffloors': 1
    })
    feat = build_anomaly_features(df, train_end=dates[25])
    assert feat['ratio_to_building_mean'].isna().all()
    assert not np.isinf(feat['ratio_to_building_mean']).any()

