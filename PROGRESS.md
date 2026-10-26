# Progress

## Done
- Project structure, requirements.txt (pyarrow added), .gitignore, .env.example
- README switched to the campus version (BDG2 dataset)
- BDG2 files in data/raw/bdg2 (git-ignored): metadata.csv, weather.csv, electricity.csv, electricity_cleaned.csv
- Phase 0: src/config.py rewritten for BDG2, src/inspect_data.py run, docs/DATA_NOTES.md written

## Current phase
Phase 1: data loading and campus selection (src/data_loader.py, notebook 01)

## Notes
- Always read README.md and docs/DATA_NOTES.md before coding.
- Never assume column names: print headers first.
- Run and test after each phase.
- Model paths (isolation_forest.joblib, forecast_model.joblib, metrics.json) were dropped from config.py; add them back in Phase 4 to 7.
- Swan site has no primaryspaceusage values; not a campus candidate.
- Weather rows per site range 16,860 to 17,544; check coverage at the chosen campus site.
