# Progress
## Done
- Phase 0: config for BDG2, src/inspect_data.py, docs/DATA_NOTES.md
- Phase 1: src/data_loader.py, Fox campus selected (131 buildings), processed data built
- Phase 2: notebooks/02_eda.ipynb completed
- Phase 3: src/features.py with tests (8 passing); no leakage, target aligned at horizon h
- Phase 4: src/anomaly.py, Isolation Forest trained, validation with injected faults completed
- Phase 5: src/forecast.py, multi-horizon forecasting trained and evaluated against baselines
- Phase 6: src/benchmark.py, per-building metrics, EUI, baseload ratio, cost/carbon, savings candidates
- Phase 7: src/train.py, master training pipeline executed successfully
- Phase 8: FastAPI backend (`backend/`) with auth, endpoints, data service
- Phase 9: Frontend dashboard (`frontend/`) with login, charts, heatmap, anomaly panel, live replay
- Phase 10: Test suite (`tests/`) - all 24 tests passed successfully (`pytest -q`)
- Phase 11: Results documentation, evaluation plot, dashboard screenshots, and README acceptance checklist completed
## Current phase
All phases (0-11) completed successfully.
## Next
No planned phase remains.
## Notes
- All models retrained and verified in the current environment.
- Pytest passing successfully (24 tests passed).
- Frontend API client syntax and dashboard navigation fixed; forecast, building, and anomaly panels are wired to the backend.
- Dashboard navigation now scrolls the selected section into view.
- README results now contain the saved forecast and anomaly metrics.
- Six documentation images are available under `docs/images/`.
- Frontend ranking and anomaly rendering handles missing numeric values and uses the API's `score` field without failing the dashboard.
- Dashboard navigation no longer hides nested Forecasting, Buildings, or Anomalies sections; the login form no longer shows a misleading hard-coded password.
