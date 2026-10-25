# Smart Campus Energy Analytics and Forecasting System

An end-to-end data science and web project that analyses real hourly electricity consumption of **university campus buildings**, detects abnormal usage, forecasts campus load, benchmarks buildings against each other, and shows everything on a web dashboard.

**Author:** Santanu Mondal ([@santanu737](https://github.com/santanu737))
**Context:** Interdisciplinary Project (IDP), B.Tech CSE (AI and ML), Adamas University
**Status:** In development

> **Note for the AI coding assistant (Gemini CLI, Copilot, etc.):** this README is the full specification. Read all of it before writing code. Build **one phase at a time** (see "Build order"), follow the folder structure and API contract, and **never assume dataset column names or file names: print the real headers first (Phase 0) and write down what you find.** Do not invent data, columns, or endpoints that are not described here.

---

## 1. Project goals

1. Clean and analyse real hourly electricity data from university campus buildings (EDA).
2. **Forecast** campus-wide electricity load (next hour and next 24 hours), using weather as an input.
3. **Detect anomalies** in building consumption (unsupervised ML), at building and hour level.
4. **Benchmark buildings**: rank them by total use, use per square metre, and night/weekend baseload (a simple "energy waste" indicator).
5. Serve the results through a **REST API** (FastAPI).
6. Show them on a responsive **web dashboard** (HTML, CSS, JavaScript, Chart.js).
7. Keep the code clean, tested, documented, and easy to explain in an interview.

**Honest framing:** the data comes from open university-building datasets, not from Adamas University's own meters. The architecture is "campus-ready": the loader can later be pointed at Adamas University's meter exports or a live feed. Do not describe it as Adamas data.

**Out of scope:** hardware, ESP32, sensors, MQTT.

---

## 2. Architecture

```
BDG2 files (metadata, weather, electricity meters)
        |
        v
Phase 0: inspect files -> docs/DATA_NOTES.md
        |
        v
Data loading, cleaning, campus site selection  (src/data_loader.py)
        |
        v
Feature engineering  (src/features.py)
        |
        +--> Campus load forecasting (Gradient Boosting)  (src/forecast.py)
        +--> Building anomaly detection (Isolation Forest) (src/anomaly.py)
        +--> Building benchmarking (ranking, EUI, baseload) (src/benchmark.py)
                |
                v
        Saved models + metrics (models/)
                |
                v
        FastAPI backend (backend/) --> JSON API
                |
                v
        Web dashboard (frontend/: HTML + CSS + JS + Chart.js)
```

---

## 3. Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ (project uses 3.13) |
| Data and ML | pandas, numpy, scikit-learn, joblib, pyarrow, matplotlib, seaborn |
| Optional ML extras | statsmodels or Prophet, TensorFlow/Keras (LSTM comparison only) |
| Backend | FastAPI, uvicorn, pydantic, python-jose (JWT), passlib or bcrypt |
| Frontend | HTML5, CSS3 (Flexbox/Grid), vanilla JavaScript (ES6+), Chart.js via CDN |
| Notebooks | Jupyter |
| Testing | pytest, httpx |
| Version control | Git, GitHub (Projects board) |

`pyarrow` must be added to `requirements.txt` (processed data is stored as Parquet because the long-format building data is large).

---

## 4. Dataset: Building Data Genome Project 2 (BDG2)

Open dataset from the ASHRAE Great Energy Predictor III competition:
**https://github.com/buds-lab/building-data-genome-project-2**
Paper: Miller et al., *Scientific Data* 7, 368 (2020).

What it contains (from the official description):
- 3,053 energy meters from 1,636 non-residential buildings at 19 sites (North America and Europe).
- Two full years, **2016 and 2017**, at **hourly** frequency.
- Meter types: electricity, chilled water, hot water, steam, gas, water, irrigation, solar.
- Building metadata and **hourly weather data per site**.
- A large share of the sites are university campuses.

Repository layout (official): `data/metadata/`, `data/weather/`, `data/meters/raw/`, `data/meters/cleaned/` (filtered version), `data/meters/kaggle/`.

**This project uses electricity meters only.**

### Files to copy into this project
Download the BDG2 repository (GitHub **Code, Download ZIP**, or `git clone --depth 1`), then copy these files into `data/raw/bdg2/` **keeping their original names**:
- the buildings **metadata** CSV (from `data/metadata/`)
- the **weather** CSV (from `data/weather/`)
- the **raw electricity** CSV (from `data/meters/raw/`)
- the **cleaned electricity** CSV (from `data/meters/cleaned/`)

These files are large and are git-ignored. Never commit them.

### Expected format (VERIFY in Phase 0, do not trust blindly)
- Meter files are usually **wide**: one `timestamp` column and one column per building (column names are building IDs).
- Metadata is usually one row per building, with fields such as site ID, building ID, primary space usage, floor area (square metres), year built, number of floors, and which meters exist.
- Weather is usually per `site_id` and `timestamp`, with fields such as air temperature, dew temperature, cloud coverage, precipitation, sea level pressure, wind direction, wind speed.
- Missing values (NaN) are common in both raw and weather data. Zeros in meter data can be real or faulty.

Phase 0 must confirm all of the above from the actual files.

---

## 5. Folder structure

Keep the existing structure and add the new files:

```
smart-energy-analytics/
├── README.md
├── PROGRESS.md
├── GEMINI.md
├── requirements.txt
├── .gitignore
├── .env.example
├── data/
│   ├── raw/bdg2/             # BDG2 files (git-ignored)
│   └── processed/            # parquet/csv outputs (git-ignored)
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_anomaly_detection.ipynb
│   └── 04_forecasting.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── inspect_data.py       # Phase 0 helper
│   ├── data_loader.py
│   ├── features.py
│   ├── anomaly.py
│   ├── forecast.py
│   ├── benchmark.py
│   ├── train.py
│   └── evaluate.py
├── models/                   # *.joblib, metrics.json (joblib git-ignored)
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   ├── services.py
│   └── auth.py
├── frontend/
│   ├── index.html
│   ├── login.html
│   ├── css/style.css
│   └── js/ (api.js, charts.js, app.js)
├── tests/
└── docs/
    ├── DATA_NOTES.md         # written in Phase 0
    └── images/
```

---

## 6. Build order (follow strictly, one phase per session)

### Phase 0: Update setup and inspect the data
1. Update `src/config.py` and `requirements.txt` for BDG2: remove the old UCI paths and constants, add paths for `data/raw/bdg2/` and the processed outputs, keep `RANDOM_STATE = 42`, add `pyarrow` to requirements. Keep paths relative with `pathlib`.
2. Write `src/inspect_data.py`. For each file in `data/raw/bdg2/` it prints: file name and size, shape, the first 5 column names, dtypes summary, number of NaNs, and the first and last timestamps where applicable.
3. Run it and write the findings into `docs/DATA_NOTES.md` (a short data dictionary: file, shape, key columns, time range, missing values, anything surprising such as timezone notes).
4. List the **sites** in the metadata, with the number of buildings, how many have an electricity meter, and the most common primary space usages per site.
5. Do not write any other code in this phase.

### Phase 1: Data loading and campus selection (`src/data_loader.py`, notebook 01)
- Functions: `load_metadata()`, `load_weather()`, `load_electricity(cleaned: bool)`, `to_long(df)`, `select_campus(...)`, `build_campus_dataset()`.
- Convert wide meter data to **long format**: `timestamp`, `building_id`, `kwh`. Use `float32` and a categorical `building_id` to save memory.
- **Campus selection:** choose ONE site to act as "the campus" (configurable `CAMPUS_SITE_ID` in `config.py`). Selection criteria, computed in the notebook and justified in markdown: many buildings with electricity meters, high data completeness (for example at least 90% non-missing hours), and mostly education, office, or similar campus-type primary space usage.
- Keep only buildings of that site with enough data. Report how many were dropped and why.
- Join building metadata and the site's weather onto the hourly data (by `timestamp`; weather by the building's `site_id`).
- Missing values:
  - weather: interpolate gaps of up to 3 hours, leave longer gaps as NaN and flag them,
  - meter data: do **not** fill silently. Keep NaN, record per-building completeness, and let models handle or drop NaN explicitly.
- Build a **campus total** hourly series (sum of all selected building readings), computed only on hours where enough buildings reported (define and document the rule).
- Save to `data/processed/`: `campus_long.parquet`, `campus_total.parquet`, `buildings_meta.csv`.
- Check timestamps are hourly, sorted, and unique per building. Document the timezone assumption in `docs/DATA_NOTES.md`, and do not convert time zones without evidence.

### Phase 2: EDA (notebook 02)
Use clear markdown headings and write a short "Observation" under every plot. Include:
- Dataset shape, data types, missing-value summary (overall and per building), completeness heatmap or bar chart.
- Campus total daily profile (median by hour) and weekday vs weekend.
- Monthly and seasonal trend of campus load.
- Load vs outdoor temperature (scatter and binned average), including cooling and heating effects.
- Buildings by primary space usage: count, total use, and average use per square metre.
- Top 10 buildings by total use and by use per square metre.
- Typical daily load profile by building type (for example education vs office).
- Distribution of readings, zeros, and suspicious flat lines.
- **Raw vs cleaned comparison:** count hours where the cleaned file has no value but the raw file does. Describe what kinds of faults these look like. (If this holds, these hours act as weak anomaly labels in Phase 4; verify before relying on it.)
- A summary table of key insights.

### Phase 3: Features (`src/features.py`)
- Time features: `hour`, `dayofweek`, `month`, `is_weekend`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`.
- Weather features: air temperature, dew temperature, wind speed, cloud coverage, precipitation, plus temperature lags (1h, 3h, 24h) and 24h rolling mean temperature.
- Target lags for the campus total: 1, 2, 3, 24, 48, 168 hours.
- Rolling statistics (past values only): 3h mean, 24h mean, 24h standard deviation.
- Building-level features for anomaly detection: reading relative to the building's median for the same **hour of week** (`dev_from_how_median`, computed on training data only), ratio to the building's own mean, and rolling z-score.
- Building metadata features: primary space usage (categorical), floor area, year built, floors.
- `build_forecast_table(campus_total, weather)` and `build_anomaly_features(long_df)`.
- No data leakage: every statistic is computed from past values or from the training period only.

### Phase 4: Anomaly detection (`src/anomaly.py`, notebook 03)
- Model: `IsolationForest(n_estimators=200, random_state=42)` with a small contamination value (for example 0.01 to 0.02), on building-relative features (not raw kWh alone, otherwise large buildings dominate).
- Two options to compare and document: one global model on normalised features, or one model per building type.
- Also implement a rule-based baseline: rolling z-score greater than 3 against the same hour of week.
- Output: `anomaly_flag`, `anomaly_score` per building-hour.
- **Validation (real data has no true labels):**
  1. Inject synthetic anomalies (spikes, drops to near zero, long high stretches) into a copy of the test period and report precision and recall.
  2. If Phase 2 confirms that raw-vs-cleaned differences indicate faults, report how many of those hours the detector flags (as weak labels, with a clear caveat).
  3. Show and describe the top flagged real events (building, time, kWh, score).
- Per-building summary: anomaly rate, and a list of the most anomalous buildings.
- Save the model to `models/isolation_forest.joblib`.

### Phase 5: Forecasting (`src/forecast.py`, notebook 04)
**5a (required): campus total load**
- Target: campus total kWh, 1 hour ahead, then 6 and 24 hours ahead.
- **Time-based split (no shuffling):** train from the start of 2016 to 31 March 2017, validate April to June 2017, test July to December 2017.
- Models to compare:
  1. Naive: same hour yesterday (`lag_24`)
  2. Seasonal naive: same hour last week (`lag_168`)
  3. `GradientBoostingRegressor` or `HistGradientBoostingRegressor` with weather and lag features (main model)
  4. Optional: a model without weather, to measure the value of weather data
- Metrics: MAE, RMSE, R-squared, and MAPE only where the actual value is not near zero.
- Plots: forecast vs actual (one week), residuals, feature importance, error by hour of day, error by month.
- Save the best model to `models/forecast_model.joblib` and metrics to `models/metrics.json`.

**5b (stretch, only after everything else works): per-building forecast**
- One global `HistGradientBoostingRegressor` with building metadata and lag features, evaluated per building. Report the median per-building error and the worst buildings.

### Phase 6: Benchmarking (`src/benchmark.py`)
- Per building: total kWh, kWh per square metre per year, average and peak load.
- **Baseload ratio:** average night (for example 00:00 to 05:00) and weekend load divided by average weekday daytime load. A high ratio suggests equipment left on.
- Rank buildings and flag top "savings candidates" (high use per square metre combined with high baseload ratio). Explain the logic in markdown and in the code docstring.
- Peer comparison: compare each building against the median of its primary space usage group.
- Estimated cost and carbon: configurable `TARIFF_PER_KWH` (default 8, currency units) and `CO2_KG_PER_KWH` (default 0.7). Both are **illustrative defaults** that must be changeable in `config.py` and in the dashboard.

### Phase 7: Training script (`src/train.py`)
- `python -m src.train` runs features, trains the forecast model, fits the anomaly model, computes the benchmark tables, saves models, metrics, and processed tables, and prints a clear results table.

### Phase 8: Backend API (`backend/`)
FastAPI app with CORS enabled, pydantic response models, and clear error messages. Load data and models **once at startup**.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Service status |
| POST | `/api/auth/login` | Returns a JWT for a demo user (credentials from `.env`) |
| GET | `/api/campus/summary` | Total kWh, average and peak load, peak time, anomaly count, building count, for a date range |
| GET | `/api/campus/history` | Campus total time series. Params: `start`, `end`, `resolution` (`hourly` or `daily`) |
| GET | `/api/buildings` | Building list with metadata and totals |
| GET | `/api/buildings/{building_id}/history` | One building's series. Params: `start`, `end`, `resolution` |
| GET | `/api/buildings/{building_id}/profile` | Average load matrix by weekday and hour (for a heatmap) |
| GET | `/api/forecast` | Campus forecast. Params: `horizon` (1, 6, 24), `as_of` |
| GET | `/api/anomalies` | Flagged readings with score. Params: `building_id`, `start`, `end`, `limit` |
| GET | `/api/ranking` | Building ranking. Params: `metric` (`total`, `eui`, `baseload`), `limit` |
| GET | `/api/usage-types` | Consumption share by primary space usage |
| GET | `/api/weather` | Weather series for the campus site. Params: `start`, `end` |
| GET | `/api/cost-carbon` | Cost and carbon estimate. Params: `start`, `end`, `tariff`, `co2_factor` |
| GET | `/api/replay/next` | Next hourly campus record from a moving pointer (demo "live" feed) |
| POST | `/api/replay/reset` | Reset the replay pointer |

Rules:
- Validate every parameter (dates inside the data range, known building IDs, allowed horizons) and return 400 or 422 with a helpful message.
- Protect every endpoint except `/api/health` and `/api/auth/login` with JWT.
- Read secrets from `.env`, never hard-code them. Hash the demo password.
- If `passlib` and `bcrypt` conflict on this Python version, use the `bcrypt` package directly.
- Serve `frontend/` as static files from the same app so one command runs everything:
  `uvicorn backend.main:app --reload`

### Phase 9: Frontend dashboard (`frontend/`)
Plain HTML, CSS, JavaScript with Chart.js from a CDN. No frameworks.

**Pages:** `login.html` (JWT stored in `localStorage`) and `index.html` (dashboard, redirects to login without a token).

**Dashboard sections**
1. **Header:** project name, date-range picker, dark mode toggle, logout.
2. **KPI cards:** campus total energy (kWh), average and peak load (kW) with peak time, number of anomalies, estimated cost, estimated carbon.
3. **Campus load chart (line):** hourly or daily, with a weather overlay option (temperature on a second axis).
4. **Forecast chart:** recent actual values plus the forecast (dashed) for 1h, 6h, or 24h.
5. **Building selector:** a dropdown that loads that building's history chart, profile heatmap, and anomalies.
6. **Weekday x hour heatmap** for the selected building (built with CSS grid and JavaScript, no extra library).
7. **Ranking chart (horizontal bar):** top 10 buildings by total use, use per square metre, or baseload ratio (metric toggle).
8. **Usage-type doughnut chart:** share of consumption by primary space usage.
9. **Anomaly panel:** table of flagged readings (building, time, kWh, score) with red highlight, click a row to jump to that building.
10. **Savings candidates box:** buildings flagged by Phase 6 with a short reason.
11. **Live replay mode:** "Start live demo" button that calls `/api/replay/next` every 2 seconds.
12. **Model performance box:** forecast MAE and RMSE vs the baselines.

**UI requirements**
- Responsive (CSS Grid and Flexbox), works on mobile and desktop.
- Dark and light mode through CSS variables, saved in `localStorage`.
- Loading spinners and friendly error messages.
- Accessible: semantic HTML, labelled inputs, good contrast.
- Clean modern look, no inline styles.
- JavaScript split into `api.js` (all `fetch` calls), `charts.js` (chart creation and updates), and `app.js` (events and logic). Use `async/await` and `try/catch`.

### Phase 10: Tests (`tests/`)
- Loader: wide-to-long conversion, hourly sorted unique index, NaN handling, campus selection rules.
- Features: no future leakage (a lag or rolling feature at time `t` uses only data before `t`).
- Benchmark: baseload ratio and per-square-metre calculations on a small hand-made example.
- API: health, login success and failure, 401 without token, validation errors, forecast response shape.
- Run with `pytest -q`.

### Phase 11: Documentation
- Fill the Results section with real numbers.
- Add 4 to 6 screenshots to `docs/images/` and show them here.
- Keep docstrings and comments short and useful.

---

## 7. Coding standards

- Python: PEP 8, type hints, docstrings on public functions, no wildcard imports.
- Use `pathlib` and `src/config.py` for all paths. No absolute paths.
- `random_state=42` everywhere.
- Watch memory: use `float32`, categorical dtypes, and Parquet; avoid copying the long dataframe unnecessarily.
- No data leakage: fit medians, scalers, and models on training data only.
- Notebooks run top to bottom without errors, with markdown headings and short observations.
- JavaScript: ES6+, `const` and `let`, no inline event handlers.
- Commit small and often (`feat:`, `fix:`, `docs:` prefixes).
- Never print or commit secrets. Never commit anything under `data/raw/` or `data/processed/`.

---

## 8. Setup and run

```bash
# 1. environment (Windows PowerShell)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. copy the BDG2 files into data/raw/bdg2/ (see Dataset section)

# 3. inspect, build data, train
python -m src.inspect_data
python -m src.data_loader
python -m src.train

# 4. copy .env.example to .env, set values, then start the app
uvicorn backend.main:app --reload
# open http://127.0.0.1:8000
```

`.env.example`:
```
SECRET_KEY=change_me
DEMO_USER=admin
DEMO_PASSWORD=change_me
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

---

## 9. Results (fill after training)

| Task | Model | Metric | Value |
|---|---|---|---|
| Campus forecast (1h ahead) | Naive (lag 24) | MAE / RMSE | TBD |
| Campus forecast (1h ahead) | Seasonal naive (lag 168) | MAE / RMSE | TBD |
| Campus forecast (1h ahead) | Gradient Boosting with weather | MAE / RMSE / R2 | TBD |
| Campus forecast (1h ahead) | Gradient Boosting without weather | MAE / RMSE | TBD |
| Anomaly detection | Isolation Forest (injected faults) | Precision / Recall | TBD |
| Anomaly detection | Rolling z-score rule | Precision / Recall | TBD |

Screenshots: add images from `docs/images/` here.

---

## 10. Limitations and future work

- Data is from open university-building datasets (2016 to 2017), not from Adamas University, and the selected campus is in a different climate and country. Results do not transfer directly.
- Real anomaly labels do not exist. Validation uses injected faults and, if confirmed, weak labels from raw-vs-cleaned differences.
- Cost and carbon use flat illustrative factors.
- Only electricity is analysed. Cooling, heating, and water meters are future work.
- Future work: per-building forecasting at scale, LSTM or Prophet comparison, holiday and academic-calendar features, demand-response suggestions, alerts by email or Telegram, Docker deployment, and plugging in Adamas University meter data or a live IoT feed (ESP32 and MQTT).

---

## 11. GitHub Projects board mapping

One **Milestone per phase** (0 to 11) and one **Issue per bullet group**. Custom fields: `Phase`, `Priority`, `Area` (Data, ML, Backend, Frontend, Docs, Testing). Columns: Backlog, To Do, In Progress, Review, Done.

---

## 12. Suggested prompts for the AI assistant (one per session)

0. "Read README.md and PROGRESS.md. Do Phase 0 only: update src/config.py and requirements.txt for BDG2 (add pyarrow), write src/inspect_data.py, and show me the code. Do not run anything yet."
1. "Now run the inspection and write docs/DATA_NOTES.md. List all sites with building counts and electricity meter counts. Do not start Phase 1."
2. "Implement Phase 1 in src/data_loader.py using only the columns and files confirmed in docs/DATA_NOTES.md. Do not write other phases."
3. "Create notebooks/02_eda.ipynb following the Phase 2 list, with an observation under every plot."
4. "Implement src/features.py (Phase 3) with no leakage, and tests/test_features.py."
5. "Implement src/anomaly.py and the validation in Phase 4."
6. "Implement src/forecast.py (Phase 5a) with baselines, the time-based split, metrics, and saved models."
7. "Implement src/benchmark.py (Phase 6) and src/train.py (Phase 7)."
8. "Build the FastAPI backend (Phase 8) exactly following the API table."
9. "Build the frontend (Phase 9) with separate api.js, charts.js, and app.js files."
10. "Write the tests (Phase 10) and fix failures. Then update the Results section from models/metrics.json."

After every phase: run it, fix errors, update `PROGRESS.md`, commit, push.

---

## 13. Acceptance checklist

- [ ] `docs/DATA_NOTES.md` exists and matches the real files
- [ ] `python -m src.data_loader` builds the campus dataset without errors
- [ ] `python -m src.train` saves models and metrics
- [ ] Forecast model beats both naive baselines on the test period
- [ ] Anomaly detector is validated with injected faults
- [ ] Building ranking and baseload analysis work
- [ ] `uvicorn backend.main:app --reload` serves API and dashboard together
- [ ] Login works and protected endpoints reject missing tokens
- [ ] Dashboard shows KPIs, campus chart, forecast, building view, heatmap, ranking, anomalies, and live replay
- [ ] Dashboard works on mobile width and in dark mode
- [ ] `pytest -q` passes
- [ ] README results and screenshots are filled in

---

## 14. License and data citation

Code: MIT (or your choice).
Data: Building Data Genome Project 2. Miller, C., Kathirgamanathan, A., Picchetti, B. et al. *The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition.* Sci Data 7, 368 (2020). Check the dataset repository for its license and cite it in reports and slides.
