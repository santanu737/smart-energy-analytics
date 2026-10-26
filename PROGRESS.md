# Progress

## Done
- Project structure, requirements.txt (pyarrow added), .gitignore, .env.example
- README switched to the campus version (BDG2 dataset)
- BDG2 files in data/raw/bdg2 (git-ignored): metadata.csv, weather.csv, electricity.csv, electricity_cleaned.csv
- Phase 0: src/config.py for BDG2, src/inspect_data.py, docs/DATA_NOTES.md
- Phase 1 (data build): src/data_loader.py; CAMPUS_SITE_ID = "Fox" (131 of 137 buildings kept at >= 90% complete hours). Built and verified: campus_long.parquet (2,298,264 rows), campus_total.parquet (17,544 rows, 7 NaN hours), buildings_meta.csv

## Current phase
Phase 1 leftover: notebooks/01_data_cleaning.ipynb (campus selection justification). Then Phase 2 EDA.

## Notes
- Always read README.md and docs/DATA_NOTES.md before coding.
- Never assume column names: print headers first.
- Model paths (isolation_forest.joblib, forecast_model.joblib, metrics.json) were dropped from config.py; add back in Phase 4 to 7.
- Fox cleaned data: 0.0% zeros, 0.4% NaN. Check raw vs cleaned in Phase 2.
- Campus total minimum 4,681 kWh: check hours with 80-99% reporting (n_reporting).
- data_loader.py still has pandas 3.1 warnings (inplace=True, melt fragmentation); clean up later.
- Swan has no primaryspaceusage; weather rows per site range 16,860 to 17,544.
