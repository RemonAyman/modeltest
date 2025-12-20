# AI Predictor — Bus Delay Project

Short description
- A small end-to-end project to clean bus trip data, train models that predict delay minutes, and serve predictions through a Flask dashboard. Includes training and evaluation scripts, model artifacts, and a web UI for interactive predictions and charts.

Table of contents
- Project structure
- Quick start (run locally)
- Data cleaning & training
- Model files and decisions
- Scripts (what each does)
- API / Endpoints
- Metrics explained
- Troubleshooting
- Notes & next steps

Project structure
- `app.py` — Flask web application (dashboard, /predict, /charts-data endpoints).
- `clean_and_train.py` — Production-ready cleaning + training pipeline. Produces `cleaned_transport_dataset.csv` and `model_data.pkl`.
- `new_train_model.py` — original/experimental training script (kept for reference and extra experiments).
- `retrain_rf.py` — utility script to retrain RandomForest and update `model_data.pkl` with `rf_model` and `rf_metrics`.
- `optimize_r2.py` — model/hyperparameter search script that chooses best model by CV R2 and saves `serving_model`.
- `optimize_rf_r2.py` — experiments comparing RF/HGB variants (raw/log1p/trim/ensemble) to improve RF R2.
- `check_data_models.py` — quick sanity-check script for `cleaned_transport_dataset.csv` and `model_data.pkl`.
- `dirty_transport_dataset.csv` — original raw dataset (input).
- `cleaned_transport_dataset.csv` — cleaned dataset produced by `clean_and_train.py`.
- `model_data.pkl` — joblib dump containing trained models, scalers, feature columns and metrics.
- `templates/` — HTML templates for the dashboard pages (login, signup, dashboard).
- `static/` — static assets (CSS etc.).

Quick start (run locally)
1. Install dependencies (recommended in a venv):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell on Windows
pip install -r requirements.txt  # create if you need (see Dependencies below)
```

2. If you have raw data, produce the cleaned dataset and models:

```powershell
python clean_and_train.py
```

This will produce `cleaned_transport_dataset.csv` and `model_data.pkl` (contains `rf_model`, `lr_model`, `hgb_model` and metrics). The script runs cross-validation and selects a `best_model` by CV RMSE and saves it.

3. (Optional) Run the RF retraining or optimization scripts:

```powershell
python retrain_rf.py         # retrain RF and update rf_model / rf_metrics
python optimize_r2.py        # performs hyperparameter search and sets best serving model by CV R2
python optimize_rf_r2.py     # experiments for RF: log1p, trimming, ensemble
```

4. Start the web app:

```powershell
python app.py
```

Open http://127.0.0.1:5000 in your browser. Log in (or sign up) and use the dashboard to input route/time/weather and get predictions. Charts are available via the CHARTS button once you have a prediction.

Dependencies
- Primary Python libs used: `pandas`, `numpy`, `scikit-learn`, `joblib`, `flask`, `werkzeug`.
- Create `requirements.txt` with versions you prefer. Minimal example:

```
pandas
numpy
scikit-learn
joblib
flask
werkzeug
scipy
```

Data cleaning & feature engineering notes
- `clean_and_train.py` performs:
  - Parsing and normalizing time fields (`scheduled_time`, `actual_time`).
  - Cleaning passenger counts and validating GPS coordinates.
  - Creating features: `delay_minutes` (target), `hour`, `is_weekend`, `hour_sin`, `hour_cos` (cyclic encoding), per-route aggregates (`route_mean_delay`, `route_median_delay`, `route_std_delay`, `route_count`), `route_hour_mean`, and `weather_mean_delay`.
  - One-hot encoding for categorical fields used during model training.
  - Scaling numeric features with a saved `scaler` (stored in `model_data.pkl`).

Model artifacts and `model_data.pkl`
- `model_data.pkl` is a joblib dict that contains:
  - `rf_model`, `lr_model`, `hgb_model` (when available)
  - `rf_metrics`, `lr_metrics`, `hgb_metrics` — dictionaries with `r2`, `rmse`, `mae`, `rae` (and `r2_clipped` in some scripts)
  - `feature_columns` — canonical column order for serving (used to align input DF)
  - `scaler` — StandardScaler used for numeric features
  - `num_features` — list of numeric feature names used for scaling
  - `best_model`, `best_model_name`, `serving_model` — chosen best by `clean_and_train.py` / `optimize_r2.py`
  - `route_stats` — per-route aggregate stats used by the app for lookup

How the Flask app serves predictions
- `app.py` loads `model_data.pkl` at startup and builds lookup maps from `cleaned_transport_dataset.csv` for `route_hour_mean` and `weather_mean_delay`.
- `/predict` builds features from the request JSON, fills any engineered features (e.g., `hour_sin`/`hour_cos`, route stats) if missing, scales numeric features, one-hot encodes categorical columns, aligns to `feature_columns` and calls the `serving_model` for predictions.
- The app returns both `serving_prediction` and `rf_prediction` (frontend expects `rf_prediction`), plus human-readable accuracy strings and per-model explanation snippets.

Metrics explained
- R2 (coefficient of determination): measures proportion of variance explained by the model. On small noisy datasets single-split R2 can be negative; always rely on CV R2 for stability.
- RMSE: root mean squared error (same units as target minutes).
- MAE: mean absolute error.
- RAE: relative absolute error = sum(|y-yhat|) / sum(|y-mean(y)|) — helpful to compare to a naive mean predictor.
- `r2_clipped` (used in some scripts) clips negative R2 to 0 for display when a negative R2 is not meaningful for dashboard users.

Common issues & troubleshooting
- Indentation or syntax errors: ensure Python file edits preserved original indentation. Use the repository `app.py` and restart with `python app.py`.
- Missing engineered columns errors in `/predict`: The app attempts to fill engineered numeric features; ensure `model_data.pkl` includes `feature_columns` and `num_features`, and `cleaned_transport_dataset.csv` is present for lookups.
- Very large predictions: If model was trained on transformed targets or with leakage, predictions may be extreme. Consider clipping predictions in `app.py` before sending to UI (e.g., clamp to [0, 240]).
- Negative R2 on single-split evaluations: prefer cross-validation metrics. Use `optimize_r2.py` to evaluate by CV R2.

Developer notes & next steps
- To force the app to serve a specific model, set `model_data['serving_model']` to the desired model and set `model_data['best_model_name']` accordingly, then save `model_data.pkl`.
- To perform more robust target encoding (avoid leakage), apply out-of-fold target encoding within CV folds, or use cross-validated target-encoding libraries.
- Consider adding unit tests for feature-building and a `requirements.txt` pinned to specific compatible versions.

Contact / contribution
- This repository is intended for experimentation. If you want changes (UI tweaks, caps for predictions, stronger HPO with XGBoost/LightGBM), tell me which part you want and I can implement it.

