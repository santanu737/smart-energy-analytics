# Smart Energy Analytics and Forecasting System

An end-to-end data science and web project that analyses real household electricity consumption, detects abnormal usage, forecasts the next hours of load, and shows everything on a web dashboard.

**Author:** Santanu Mondal ([@santanu737](https://github.com/santanu737))
**Context:** Interdisciplinary Project (IDP), B.Tech CSE (AI and ML), Adamas University
**Status:** In development

> **Note for GitHub Copilot:** this README is the full specification of the project. Read all of it before generating code. Build the project **phase by phase** (see "Build order"), follow the folder structure and API contract exactly, and do not invent dataset columns or endpoints that are not described here.

---

## 1. Project goals

1. Clean and analyse a real smart-meter dataset (EDA).
2. **Detect anomalies** in energy consumption (unsupervised ML).
3. **Forecast** energy load for the next hour(s) (supervised ML) and compare it with simple baselines.
4. Serve the results through a **REST API** (FastAPI).
5. Show them on a responsive **web dashboard** (HTML, CSS, JavaScript, Chart.js).
6. Keep the code clean, tested, documented, and easy to explain in an interview.

**Out of scope:** hardware, ESP32, sensors, MQTT. The system uses a recorded public dataset. The architecture is "IoT-ready": the data loader can later be replaced by a live data source.

---

## 2. Architecture

```
Raw dataset (CSV/TXT)
        |
        v
Data cleaning + hourly resampling  (src/data_loader.py)
        |
        v
Feature engineering  (src/features.py)
        |
        +--> Anomaly detection (Isolation Forest)  (src/anomaly.py)
        +--> Load forecasting (Gradient Boosting)  (src/forecast.py)
                |
                v
        Saved models (models/*.joblib)
                |
                v
        FastAPI backend  (backend/)  --> JSON API
                |
                v
        Web dashboard  (frontend/: HTML + CSS + JS + Chart.js)
```

---

## 3. Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Data and ML | pandas, numpy, scikit-learn, joblib, matplotlib, seaborn |
| Optional ML extras | statsmodels or Prophet, TensorFlow/Keras (LSTM comparison only) |
| Backend | FastAPI, uvicorn, pydantic, python-jose (JWT), passlib |
| Frontend | HTML5, CSS3 (Flexbox/Grid), vanilla JavaScript (ES6+), Chart.js via CDN |
| Notebooks | Jupyter |
| Testing | pytest, httpx (API tests) |
| Version control | Git, GitHub (Projects board for task tracking) |

---

## 4. Dataset

**UCI "Individual Household Electric Power Consumption"**
One household, December 2006 to November 2010, one reading per minute (about 2.07 million rows).

Place the file here (it is large, so it is git-ignored):
`data/raw/household_power_consumption.txt`

File format:
- Separator is `;`
- Missing values are written as `?`
- `Date` is `dd/mm/yyyy`, `Time` is `hh:mm:ss`

| Column | Meaning | Unit |
|---|---|---|
| `Date`, `Time` | Timestamp parts | dd/mm/yyyy, hh:mm:ss |
| `Global_active_power` | Household active power (per-minute average) | kW |
| `Global_reactive_power` | Household reactive power | kW |
| `Voltage` | Minute-averaged voltage | V |
| `Global_intensity` | Minute-averaged current | A |
| `Sub_metering_1` | Kitchen (dishwasher, oven, microwave) | Wh |
| `Sub_metering_2` | Laundry room (washer, dryer, fridge, light) | Wh |
| `Sub_metering_3` | Electric water heater and air conditioner | Wh |

Only these columns exist. **Do not add or assume other columns** (such as temperature) unless an external dataset is explicitly merged in a later phase.

Derived column: `other_wh = Global_active_power * 1000 / 60 - (Sub_metering_1 + Sub_metering_2 + Sub_metering_3)` (energy not covered by the three sub-meters, in Wh per minute).

---

## 5. Folder structure

Create exactly this layout:

```
smart-energy-analytics/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── data/
│   ├── raw/                  # household_power_consumption.txt (git-ignored)
│   └── processed/            # hourly.csv, daily.csv (git-ignored)
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_anomaly_detection.ipynb
│   └── 04_forecasting.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py             # paths and constants
│   ├── data_loader.py
│   ├── features.py
│   ├── anomaly.py
│   ├── forecast.py
│   ├── train.py              # trains and saves all models
│   └── evaluate.py
├── models/                   # *.joblib (git-ignored except .gitkeep)
├── backend/
│   ├── __init__.py
│   ├── main.py               # FastAPI app
│   ├── schemas.py            # pydantic models
│   ├── services.py           # data + model access
│   └── auth.py               # JWT login
├── frontend/
│   ├── index.html            # dashboard
│   ├── login.html
│   ├── css/style.css
│   └── js/
│       ├── api.js            # fetch wrappers
│       ├── charts.js         # Chart.js setup
│       └── app.js            # page logic
├── tests/
│   ├── test_data_loader.py
│   ├── test_features.py
│   └── test_api.py
└── docs/
    └── images/               # screenshots and plots for this README
```

---

## 6. Build order (follow strictly)

### Phase 1: Data ingestion and cleaning (`src/data_loader.py`, notebook 01)
- Read the raw file with `sep=';'`, `na_values='?'`, `low_memory=False`.
- Combine `Date` and `Time` into one `datetime` index (format `%d/%m/%Y %H:%M:%S`), sort it, remove duplicates.
- Convert numeric columns to float.
- Report missing values (count and percentage, and the longest gap).
- Missing-value policy:
  - interpolate gaps of up to 3 consecutive minutes (time-based),
  - leave longer gaps as `NaN` and flag them in a `was_missing` column.
- Resample to **hourly**:
  - `mean` for `Global_active_power`, `Global_reactive_power`, `Voltage`, `Global_intensity`,
  - `sum` for `Sub_metering_1/2/3` (Wh per hour),
  - drop hours that have more than 20% missing minutes.
- Add `energy_kwh` (hourly kWh) = hourly mean of `Global_active_power` (kW) for one hour.
- Save `data/processed/hourly.csv` and `data/processed/daily.csv`.
- Provide `load_raw()`, `clean()`, `resample_hourly()`, and `load_hourly()` functions with docstrings and type hints.

### Phase 2: EDA (notebook 02)
Use clear markdown headings and write a short "Observation" under every plot. Include:
- Dataset shape, `info()`, `describe()`, missing-value summary.
- Hourly load profile (median by hour of day).
- Weekday vs weekend profile.
- Monthly and seasonal trend (average kWh per month, per year).
- Sub-metering breakdown (kitchen, laundry, water heater and AC, other) as a share of total.
- Distribution of power (histogram and boxplot).
- Correlation heatmap.
- Voltage behaviour and its relation to load.
- A summary table of key insights (peak hour, busiest month, weekend effect, biggest appliance group).

### Phase 3: Features (`src/features.py`)
- Time features: `hour`, `dayofweek`, `month`, `is_weekend`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`.
- Lags of target power: 1, 2, 3, 24, 48, 168 hours.
- Rolling statistics: 3h mean, 24h mean, 24h std (computed using past values only, no leakage).
- `dev_from_hour_median`: deviation from the median for the same hour of day (computed on training data only).
- A function `build_forecast_table(df)` and a function `build_anomaly_features(df)`.

### Phase 4: Anomaly detection (`src/anomaly.py`, notebook 03)
- Model: `IsolationForest(n_estimators=200, contamination=0.01 to 0.03, random_state=42)`.
- Features: power, `dev_from_hour_median`, hour sin/cos, voltage, current, and the three sub-metering values.
- The real dataset has **no ground-truth anomaly labels**. Therefore:
  1. Validate the detector by **injecting synthetic anomalies** (spikes, near-zero drops, long high-load stretches) into a copy of the test period and reporting precision and recall.
  2. Inspect the top flagged real events (date, power, hour) and describe them in the notebook.
- Output for each row: `anomaly_flag` (0/1) and `anomaly_score`.
- Add a simple rule-based comparison (z-score > 3 per hour-of-day) and compare it with Isolation Forest.
- Save the fitted model to `models/isolation_forest.joblib`.

### Phase 5: Forecasting (`src/forecast.py`, notebook 04)
- Target: next-hour `Global_active_power` (kW).
- **Time-based split** (never shuffle): train on 2007 to 2009, validate on the first half of 2010, test on the rest of 2010.
- Remove flagged anomalies from the training target and interpolate before fitting.
- Models to compare:
  1. Naive baseline: same hour yesterday (`lag_24`)
  2. Seasonal naive: same hour last week (`lag_168`)
  3. `GradientBoostingRegressor` (main model)
  4. Optional: LSTM or Prophet as a bonus comparison
- Metrics: MAE, RMSE, R-squared (also report MAPE only where the actual value is not near zero).
- Plots: forecast vs actual (3-day window), residuals histogram, feature importance, error by hour of day.
- Multi-step horizon: support horizons of 1, 6, and 24 hours (recursive or direct approach, explain the choice).
- Save the best model to `models/forecast_model.joblib` and metrics to `models/metrics.json`.

### Phase 6: Training script (`src/train.py`)
- One command trains everything and saves models and metrics:
  `python -m src.train`
- Print a clear results table at the end.

### Phase 7: Backend API (`backend/`)
FastAPI app with CORS enabled for the frontend, pydantic response models, and clear error messages.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Service status |
| POST | `/api/auth/login` | Returns a JWT for a demo user (credentials from `.env`) |
| GET | `/api/summary` | Total kWh, average load, peak hour, peak load, anomaly count, for a date range |
| GET | `/api/history` | Time series. Query params: `start`, `end`, `resolution` (`hourly` or `daily`) |
| GET | `/api/forecast` | Forecast. Query params: `horizon` (1, 6, or 24) and `as_of` (timestamp) |
| GET | `/api/anomalies` | List of flagged readings with score. Query params: `start`, `end`, `limit` |
| GET | `/api/appliances` | Sub-metering share (kitchen, laundry, water heater and AC, other) for a range |
| GET | `/api/cost` | Estimated cost. Query params: `start`, `end`, `tariff` (currency per kWh, default 8) |
| GET | `/api/replay/next` | Returns the next hourly record from a moving pointer, to imitate a live feed |
| POST | `/api/replay/reset` | Resets the replay pointer |

Rules:
- Load the data and the models **once at startup**, not on every request.
- Validate all query parameters (dates, ranges, horizon values) and return HTTP 422 or 400 with a helpful message.
- Protect every endpoint except `/api/health` and `/api/auth/login` with JWT.
- Hash the demo password, read secrets from `.env`, never hard-code them.
- Run with: `uvicorn backend.main:app --reload`
- Serve the `frontend/` folder as static files from the same app, so the whole project runs from one command.

### Phase 8: Frontend dashboard (`frontend/`)
Plain HTML, CSS, JavaScript with Chart.js from a CDN. No frameworks.

**Pages**
- `login.html`: username and password form, stores the JWT in `localStorage`, redirects to the dashboard, shows errors.
- `index.html`: the dashboard. Redirect to login if no valid token.

**Dashboard sections**
1. **Header:** project name, date-range picker (start and end), dark mode toggle, logout button.
2. **KPI cards:** total energy (kWh), average load (kW), peak load and peak hour, number of anomalies, estimated cost.
3. **Load chart (line):** actual power over the selected range, with a toggle between hourly and daily.
4. **Forecast chart (line):** recent actual values plus the forecast (dashed line) for the chosen horizon (1h, 6h, 24h).
5. **Daily profile chart:** average load by hour of day, weekday vs weekend.
6. **Appliance breakdown (doughnut chart):** kitchen, laundry, water heater and AC, other.
7. **Anomaly panel:** table of flagged readings (time, power, score) with red highlight, and red markers on the load chart.
8. **Live replay mode:** a "Start live demo" button that calls `/api/replay/next` every 2 seconds and updates a live card and chart.
9. **Model performance box:** shows MAE and RMSE of the model vs the baseline (read from the metrics file through an endpoint or from `/api/summary`).

**UI requirements**
- Responsive layout (CSS Grid and Flexbox), works on mobile and desktop.
- Dark mode and light mode using CSS variables, preference saved in `localStorage`.
- Loading spinners and friendly error messages when the API fails.
- Accessible: semantic HTML, labels for inputs, sufficient colour contrast.
- Clean, modern look (cards, soft shadows, consistent spacing). No inline styles.
- JavaScript split into `api.js` (all `fetch` calls), `charts.js` (chart creation and update), and `app.js` (events and page logic). Use `async/await` and `try/catch`.

### Phase 9: Tests (`tests/`)
- Data loader: handles `?` values, builds a correct datetime index, hourly resample has expected columns.
- Features: no future leakage (a lag feature at time `t` uses only data before `t`).
- API: health, login success and failure, protected route without token returns 401, validation errors return 4xx, forecast returns the expected shape.
- Run with: `pytest -q`

### Phase 10: Documentation
- Fill the "Results" section of this README with real numbers and screenshots after training.
- Add 4 to 6 screenshots to `docs/images/` (dashboard, forecast chart, anomaly table, EDA plots).
- Keep docstrings and comments short and useful.

---

## 7. Coding standards

- Python: PEP 8, type hints, docstrings on public functions, no wildcard imports.
- Use `pathlib` and `src/config.py` for all paths. No absolute paths.
- Set `random_state=42` everywhere for reproducibility.
- No data leakage: fit scalers, medians, and models on training data only.
- Keep notebooks clean: markdown headings, short observations, outputs visible, run top to bottom without errors.
- JavaScript: ES6+, `const`/`let`, no global variable clutter, no inline event handlers.
- Commit small and often with clear messages, for example `feat: add hourly resampling` or `fix: handle missing values`.

---

## 8. Setup and run

```bash
# 1. clone and enter the project
git clone https://github.com/santanu737/smart-energy-analytics.git
cd smart-energy-analytics

# 2. virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. install dependencies
pip install -r requirements.txt

# 4. put the dataset in data/raw/ (see Dataset section), then:
python -m src.data_loader       # builds data/processed/hourly.csv
python -m src.train             # trains models and saves metrics

# 5. copy .env.example to .env and set values, then start the app
uvicorn backend.main:app --reload
# open http://127.0.0.1:8000
```

`requirements.txt` should include: `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`, `jupyter`, `fastapi`, `uvicorn[standard]`, `python-jose[cryptography]`, `passlib[bcrypt]`, `python-dotenv`, `pytest`, `httpx`.

`.env.example`:
```
SECRET_KEY=change_me
DEMO_USER=admin
DEMO_PASSWORD=change_me
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

`.gitignore` must exclude: `.venv/`, `.env`, `__pycache__/`, `.ipynb_checkpoints/`, `data/raw/*`, `data/processed/*`, `models/*.joblib` (keep `.gitkeep` files).

---

## 9. Results (fill after training)

| Task | Model | Metric | Value |
|---|---|---|---|
| Forecast (1h ahead) | Naive (lag 24) | MAE / RMSE | TBD |
| Forecast (1h ahead) | Seasonal naive (lag 168) | MAE / RMSE | TBD |
| Forecast (1h ahead) | Gradient Boosting | MAE / RMSE / R2 | TBD |
| Anomaly detection | Isolation Forest (injected faults) | Precision / Recall | TBD |
| Anomaly detection | Z-score rule | Precision / Recall | TBD |

Screenshots: add images from `docs/images/` here.

---

## 10. Limitations and future work

- One household only, so results may not generalise to other homes or climates.
- No weather data in this dataset. Merging temperature could improve the forecast.
- No real anomaly labels, so anomaly performance is validated on injected faults.
- Cost uses a flat tariff (the dataset is from a French home, so the amount is illustrative).
- Future work: LSTM/Prophet comparison, weather features, per-appliance disaggregation, alerts by email or Telegram, Docker deployment, and connecting a live IoT source (ESP32 + MQTT) in place of the dataset.

---

## 11. GitHub Projects board mapping

Create one **Milestone per phase** and one **Issue per bullet group**. Suggested custom fields: `Phase`, `Priority`, `Area` (Data, ML, Backend, Frontend, Docs, Testing).

Board columns: **Backlog, To Do, In Progress, Review, Done**.

---

## 12. Suggested prompts for Copilot (use one at a time)

1. "Read README.md. Create the folder structure, `requirements.txt`, `.gitignore`, `.env.example`, and `src/config.py` exactly as described."
2. "Implement Phase 1 in `src/data_loader.py` with the cleaning and resampling rules in README.md. Add docstrings and type hints."
3. "Create `notebooks/02_eda.ipynb` following the Phase 2 list, with an observation under every plot."
4. "Implement `src/features.py` for Phase 3 with no data leakage, then write `tests/test_features.py`."
5. "Implement `src/anomaly.py` and the validation with injected anomalies as described in Phase 4."
6. "Implement `src/forecast.py` and `src/train.py` for Phase 5 and 6, with baselines, time-based split, metrics, and saved models."
7. "Build the FastAPI backend in `backend/` exactly following the API table in README.md, with JWT auth and input validation."
8. "Build the frontend in `frontend/` following the Phase 8 sections and UI requirements, using Chart.js and separate `api.js`, `charts.js`, `app.js` files."
9. "Write the tests from Phase 9 and fix any failing ones."
10. "Update the Results section of README.md using `models/metrics.json`."

---

## 13. Acceptance checklist

- [ ] `python -m src.train` runs without errors and saves models and metrics
- [ ] Forecast model beats both baselines on the test set
- [ ] Anomaly detector is validated with injected faults
- [ ] `uvicorn backend.main:app --reload` serves API and dashboard together
- [ ] Login works and protected endpoints reject missing tokens
- [ ] Dashboard shows KPIs, load chart, forecast chart, appliance chart, anomalies, and live replay
- [ ] Dashboard works on mobile width and in dark mode
- [ ] `pytest -q` passes
- [ ] README results and screenshots are filled in

---

## 14. License

MIT (or your choice). Dataset license: see the UCI Machine Learning Repository page for the dataset.
