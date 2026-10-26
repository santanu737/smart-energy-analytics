"""
Inspect raw BDG2 CSV files for Phase 0.
Prints file size, shape, headers, dtypes, NaNs, and time range.
"""

import os
import pandas as pd
from src.config import RAW_DIR

def inspect_files():
    if not RAW_DIR.exists():
        print(f"Error: Raw directory not found at {RAW_DIR}")
        return

    csv_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.csv')])
    
    if not csv_files:
        print(f"No CSV files found in {RAW_DIR}")
        return

    print(f"Found {len(csv_files)} files in {RAW_DIR}\n")
    print("-" * 50)

    for filename in csv_files:
        filepath = RAW_DIR / filename
        filesize_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        # Load only a few rows first to get headers and dtypes efficiently for large files
        # then load fully for shape and NaN counts (if memory allows)
        # Note: electricity files are large, but we need total NaN count and shape.
        try:
            # We use low_memory=False to avoid DtypeWarning on large files
            df = pd.read_csv(filepath, low_memory=False)
            
            print(f"File: {filename}")
            print(f"Size: {filesize_mb:.2f} MB")
            print(f"Shape: {df.shape}")
            
            # First 5 columns
            cols = df.columns.tolist()
            print(f"First 5 columns: {cols[:5]}")
            
            # Dtypes summary
            print("Dtypes summary:")
            print(df.dtypes.value_counts())
            
            # NaN count
            print(f"Total NaN count: {df.isnull().sum().sum()}")
            
            # Timestamp range if 'timestamp' exists
            if 'timestamp' in df.columns:
                # Ensure it's sorted or just take first/last as they appear
                first_ts = df['timestamp'].iloc[0]
                last_ts = df['timestamp'].iloc[-1]
                print(f"First timestamp: {first_ts}")
                print(f"Last timestamp: {last_ts}")
            
            print("-" * 50)
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            print("-" * 50)

if __name__ == "__main__":
    inspect_files()
