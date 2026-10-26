# Progress
## Done
- Phase 0: config for BDG2, src/inspect_data.py, docs/DATA_NOTES.md
- Phase 1: src/data_loader.py, Fox campus selected (131 buildings), processed data built
- Phase 3: src/features.py with tests (8 passing); no leakage, target aligned at horizon h
- notebooks/01_data_cleaning.ipynb written (not executed yet, jupyter not installed)
## Current phase
Phase 2: EDA notebook (needs jupyter)
## Next
Phases 4+5: anomaly detection (injected-fault validation), forecasting (naive baselines, time split)
## Notes
- Always read README.md before coding. Never assume column names: print headers first.
- Run pytest after each change. Slow internet: use pip --default-timeout 120 --retries 10.
- Cleanup later: encode primaryspaceusage with a fixed mapping before Phase 4.
- Check which electricity file campus_long.parquet is built from (raw or cleaned) and document it.
