"""Configuration paths and project constants for Smart Energy Analytics (BDG2 version)."""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "bdg2"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# Raw file paths (verified names in Phase 0)
METADATA_CSV = RAW_DIR / "metadata.csv"
WEATHER_CSV = RAW_DIR / "weather.csv"
ELECTRICITY_CSV = RAW_DIR / "electricity.csv"
ELECTRICITY_CLEANED_CSV = RAW_DIR / "electricity_cleaned.csv"

# Processed output paths
CAMPUS_LONG_PARQUET = PROCESSED_DIR / "campus_long.parquet"
CAMPUS_TOTAL_PARQUET = PROCESSED_DIR / "campus_total.parquet"
BUILDINGS_META_CSV = PROCESSED_DIR / "buildings_meta.csv"

# Global settings
RANDOM_STATE = 42
CAMPUS_SITE_ID = None  # To be chosen in Phase 1

# Financial and Environmental factors (illustrative defaults)
TARIFF_PER_KWH = 8  # Local currency units per kWh
CO2_KG_PER_KWH = 0.7  # kg CO2 per kWh
