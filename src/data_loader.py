"""Data loading and campus selection utilities for Phase 1."""

import pandas as pd
import numpy as np
import sys
from typing import Optional
from src import config

def load_metadata() -> pd.DataFrame:
    """Load the building metadata CSV, indexed by building_id."""
    df = pd.read_csv(config.METADATA_CSV)
    df.set_index('building_id', inplace=True)
    return df

def load_weather() -> pd.DataFrame:
    """Load the weather CSV with parsed timestamps."""
    df = pd.read_csv(config.WEATHER_CSV, parse_dates=['timestamp'])
    return df

def load_electricity(cleaned: bool = True, site_id: Optional[str] = None) -> pd.DataFrame:
    """
    Load wide electricity data with a datetime index named 'timestamp'.
    
    Args:
        cleaned: If True, load electricity_cleaned.csv, otherwise electricity.csv.
        site_id: If provided, only load building columns belonging to this site.
        
    Returns:
        pd.DataFrame: Wide format, float32, indexed by 'timestamp'.
    """
    path = config.ELECTRICITY_CLEANED_CSV if cleaned else config.ELECTRICITY_CSV
    
    usecols = None
    if site_id:
        meta = load_metadata()
        site_buildings = set(meta[meta['site_id'] == site_id].index)
        # Issue 1: read CSV header once
        csv_cols = pd.read_csv(path, nrows=0).columns
        usecols = ['timestamp'] + [b for b in csv_cols if b in site_buildings]
        
    df = pd.read_csv(path, usecols=usecols, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df.astype(np.float32)

def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide meter DataFrame to long format.
    
    Columns: timestamp, building_id (categorical), kwh (float32).
    Keeps NaN values.
    """
    long_df = df.reset_index().melt(
        id_vars='timestamp', 
        var_name='building_id', 
        value_name='kwh'
    )
    long_df['building_id'] = long_df['building_id'].astype('category')
    long_df['kwh'] = long_df['kwh'].astype(np.float32)
    return long_df

def site_completeness() -> pd.DataFrame:
    """
    Compute data completeness metrics per site.
    
    Returns a DataFrame with one row per site:
    - n_buildings_elec: Number of buildings with an electricity meter.
    - mean_completeness: Mean % of non-missing hours in cleaned data.
    - pct_90_plus: % of buildings with >= 90% non-missing hours.
    - campus_usage_share: Share of buildings with 'Education' or 'Office' usage.
    - weather_rows: Number of weather rows for the site.
    """
    meta = load_metadata()
    weather = load_weather()
    elec = load_electricity(cleaned=True)
    
    non_missing = elec.notnull().mean()
    
    sites = meta['site_id'].unique()
    report = []
    
    for site in sites:
        site_meta = meta[meta['site_id'] == site]
        site_bldgs = site_meta[site_meta['electricity'] == 'Yes'].index
        site_elec_completeness = non_missing.reindex(site_bldgs)
        
        n_elec = len(site_bldgs)
        mean_comp = site_elec_completeness.mean()
        p90 = (site_elec_completeness >= 0.9).mean()
        
        campus_types = ['Education', 'Office']
        usage_share = site_meta['primaryspaceusage'].isin(campus_types).mean()
        
        w_rows = len(weather[weather['site_id'] == site])
        
        report.append({
            'site_id': site,
            'n_buildings_elec': n_elec,
            'mean_completeness': mean_comp,
            'pct_90_plus': p90,
            'campus_usage_share': usage_share,
            'weather_rows': w_rows
        })
        
    return pd.DataFrame(report).set_index('site_id')

def select_campus(site_id: str, min_completeness: float = 0.9):
    """
    Keep only the site's buildings with >= min_completeness non-missing hours.
    
    Returns:
        tuple: (kept_metadata DataFrame, report dict)
    """
    meta = load_metadata()
    site_meta = meta[meta['site_id'] == site_id].copy()
    
    # Issue 6: detailed dropped buildings report
    path = config.ELECTRICITY_CLEANED_CSV
    csv_cols = set(pd.read_csv(path, nrows=0).columns)
    
    site_bldgs = site_meta.index
    has_meter = site_bldgs.isin(csv_cols)
    no_meter_list = site_bldgs[~has_meter].tolist()
    
    elec = load_electricity(cleaned=True, site_id=site_id)
    completeness = elec.notnull().mean()
    
    below_comp_dict = completeness[completeness < min_completeness].to_dict()
    kept_bldgs = completeness[completeness >= min_completeness].index
    
    # Issue 5: Filter columns and set categories
    meta_cols = ['site_id', 'primaryspaceusage', 'sub_primaryspaceusage', 'sqm', 'yearbuilt', 'numberoffloors']
    kept_meta = site_meta.loc[kept_bldgs, meta_cols].copy()
    for col in ['site_id', 'primaryspaceusage', 'sub_primaryspaceusage']:
        if col in kept_meta.columns:
            kept_meta[col] = kept_meta[col].astype('category')
    
    report = {
        'site_id': site_id,
        'initial_buildings': len(site_meta),
        'kept_buildings': len(kept_meta),
        'dropped': {
            'no_electricity_meter': no_meter_list,
            'below_completeness': below_comp_dict
        }
    }
    
    return kept_meta, report

def build_campus_dataset():
    """
    Build the final campus dataset using CAMPUS_SITE_ID from config.
    
    Steps:
    1. Filter buildings for the selected site (completeness >= 90%).
    2. Convert electricity to long format.
    3. Join metadata and weather (with interpolation).
    4. Compute campus total series and n_reporting.
    5. Save results to PROCESSED_DIR.
    
    The campus total is computed only for hours where at least 80% of 
    the selected buildings have a valid reading.
    Note: hours with 80% to 99% reporting slightly understate the total campus load.
    """
    site_id = config.CAMPUS_SITE_ID
    if site_id is None:
        raise ValueError("CAMPUS_SITE_ID is not set in src/config.py")
        
    # 1. Select buildings
    kept_meta, report = select_campus(site_id, min_completeness=0.9)
    print(f"Campus Selection Report for {site_id}:")
    print(f"  Kept: {report['kept_buildings']}")
    print(f"  Dropped (no meter): {len(report['dropped']['no_electricity_meter'])}")
    print(f"  Dropped (low completeness): {len(report['dropped']['below_completeness'])}")
    
    # 2. Load and transform electricity
    elec_wide = load_electricity(cleaned=True, site_id=site_id)
    elec_wide = elec_wide[kept_meta.index]
    long_df = to_long(elec_wide)
    
    # 3. Join metadata (Issue 5: Ensure building_id remains categorical)
    long_df = long_df.merge(kept_meta.reset_index(), on='building_id', how='left')
    long_df['building_id'] = long_df['building_id'].astype('category')
    
    # 4. Weather processing
    weather = load_weather()
    # Issue 3: Drop site_id to avoid merge collisions
    site_weather = weather[weather['site_id'] == site_id].set_index('timestamp').drop(columns=['site_id']).copy()
    
    # Reindex weather to electricity timestamps
    site_weather = site_weather.reindex(elec_wide.index)
    site_weather.index.name = 'timestamp'
    
    # Issue 4: Weather gap rule
    for col in site_weather.columns:
        mask = site_weather[col].isnull()
        groups = (mask != mask.shift()).cumsum()[mask]
        gap_lengths = groups.map(groups.value_counts())
        
        # Interpolate
        site_weather[col] = site_weather[col].interpolate(method='linear', limit=3, limit_area='inside')
        
        # Re-mask gaps longer than 3
        site_weather.loc[mask & (gap_lengths > 3), col] = np.nan
        
        # Flag 1 where still NaN (long gaps and missing hours)
        site_weather[f'{col}_gap_flag'] = site_weather[col].isnull().astype(np.int8)
        
        # Convert to float32
        site_weather[col] = site_weather[col].astype(np.float32)
        
    long_df = long_df.merge(site_weather.reset_index(), on='timestamp', how='left')
    
    # 5. Campus total (Issue 7: add n_reporting)
    n_bldgs = len(kept_meta)
    threshold = 0.8 * n_bldgs
    
    reporting_count = elec_wide.notnull().sum(axis=1)
    campus_total = elec_wide.sum(axis=1)
    campus_total[reporting_count < threshold] = np.nan
    
    campus_total_df = pd.DataFrame({
        'timestamp': campus_total.index,
        'kwh': campus_total.values.astype(np.float32),
        'n_reporting': reporting_count.values.astype(np.int32)
    })
    
    # 6. Save outputs
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    long_df.to_parquet(config.CAMPUS_LONG_PARQUET, index=False)
    campus_total_df.to_parquet(config.CAMPUS_TOTAL_PARQUET, index=False)
    kept_meta.to_csv(config.BUILDINGS_META_CSV)
    
    print(f"Saved processed data to {config.PROCESSED_DIR}")

if __name__ == "__main__":
    # Issue 8: Main block handle --build
    if '--build' in sys.argv:
        build_campus_dataset()
