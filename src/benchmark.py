"""Building benchmarking and energy waste indicators module."""

import pandas as pd
import numpy as np
from src import config

def compute_benchmarks() -> pd.DataFrame:
    """
    Compute per-building metrics: total kWh, EUI (kWh/sqm/year), peak load, 
    baseload ratio, cost, carbon, and rankings.
    """
    long_path = config.CAMPUS_LONG_PARQUET
    meta_path = config.BUILDINGS_META_CSV
    
    if not long_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Processed campus dataset not found. Run data loader first.")
        
    long_df = pd.read_parquet(long_path)
    meta = pd.read_csv(meta_path)
    if 'building_id' in meta.columns:
        meta.set_index('building_id', inplace=True)
        
    long_df['timestamp'] = pd.to_datetime(long_df['timestamp'])
    
    # Total kWh and peak load per building
    agg = long_df.groupby('building_id')['kwh'].agg(
        total_kwh='sum',
        mean_kwh='mean',
        peak_kwh='max',
        valid_hours='count'
    ).reset_index()
    
    agg = agg.merge(meta.reset_index(), on='building_id', how='left')
    
    # EUI (kWh per square metre per year). Assuming 2 years of data (2016-2017), so divide total by (2 * sqm) or normalise.
    # BDG2 covers 2016-2017 (2 full years). EUI per year = total_kwh / (2 * sqm)
    agg['sqm'] = agg['sqm'].replace(0, np.nan)
    agg['eui_annual'] = agg['total_kwh'] / (2.0 * agg['sqm'])
    
    # Baseload ratio calculation:
    # Night (00:00 to 05:00) and weekend load divided by average weekday daytime load (08:00 to 17:00 on weekdays).
    long_df['hour'] = long_df['timestamp'].dt.hour
    long_df['dayofweek'] = long_df['timestamp'].dt.dayofweek
    long_df['is_weekend'] = long_df['dayofweek'].isin([5, 6])
    
    # Night or weekend
    night_weekend = (long_df['hour'] >= 0) & (long_df['hour'] <= 5) | long_df['is_weekend']
    # Weekday daytime
    weekday_day = (~long_df['is_weekend']) & (long_df['hour'] >= 8) & (long_df['hour'] <= 17)
    
    night_weekend_mean = long_df[night_weekend].groupby('building_id')['kwh'].mean()
    weekday_day_mean = long_df[weekday_day].groupby('building_id')['kwh'].mean()
    
    baseload_ratio = (night_weekend_mean / weekday_day_mean).rename('baseload_ratio')
    agg = agg.merge(baseload_ratio, on='building_id', how='left')
    
    # Cost and carbon estimates
    agg['estimated_cost'] = agg['total_kwh'] * config.TARIFF_PER_KWH
    agg['estimated_co2_kg'] = agg['total_kwh'] * config.CO2_KG_PER_KWH
    
    # Savings candidates flag: high EUI (above median) and high baseload ratio (above median)
    median_eui = agg['eui_annual'].median()
    median_baseload = agg['baseload_ratio'].median()
    
    agg['savings_candidate'] = (agg['eui_annual'] > median_eui) & (agg['baseload_ratio'] > median_baseload)
    
    return agg

if __name__ == "__main__":
    df = compute_benchmarks()
    print(f"Computed benchmarks for {len(df)} buildings. Savings candidates: {df['savings_candidate'].sum()}")
