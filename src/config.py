"""Configuration paths and project constants for Smart Energy Analytics."""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
TESTS_DIR = BASE_DIR / "tests"
DOCS_DIR = BASE_DIR / "docs"

# Specific file paths
RAW_DATA_PATH = RAW_DATA_DIR / "household_power_consumption.txt"
HOURLY_DATA_PATH = PROCESSED_DATA_DIR / "hourly.csv"
DAILY_DATA_PATH = PROCESSED_DATA_DIR / "daily.csv"
ISOLATION_FOREST_PATH = MODELS_DIR / "isolation_forest.joblib"
FORECAST_MODEL_PATH = MODELS_DIR / "forecast_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Reproducibility
RANDOM_STATE = 42

# Dataset specifications
RAW_DATA_SEP = ";"
RAW_DATA_NA_VALUES = "?"
DATETIME_FORMAT = "%d/%m/%Y %H:%M:%S"

TARGET_COLUMN = "Global_active_power"

NUMERIC_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

SUB_METERING_COLUMNS = [
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
