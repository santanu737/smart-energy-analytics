"""Master training script for data loading, anomaly detection, forecasting, and benchmarking."""

import sys
import pandas as pd
from src import config
from src.data_loader import build_campus_dataset
from src.anomaly import run_phase4
from src.forecast import run_phase5
from src.benchmark import compute_benchmarks

def main():
    print("=== Step 1: Building Campus Dataset ===")
    build_campus_dataset()
    
    print("\n=== Step 2: Training Anomaly Detection Model ===")
    anomaly_metrics = run_phase4(save=True)
    
    print("\n=== Step 3: Training Forecasting Models (Multi-Horizon) ===")
    forecast_metrics = run_phase5(save=True)
    
    print("\n=== Step 4: Computing Building Benchmarks ===")
    benchmarks = compute_benchmarks()
    benchmarks.to_csv(config.PROCESSED_DIR / "building_benchmarks.csv", index=False)
    print(benchmarks.head(10))
    print(f"Total savings candidates identified: {benchmarks['savings_candidate'].sum()}")
    
    print("\n=== Training pipeline complete successfully! ===")

if __name__ == "__main__":
    main()
