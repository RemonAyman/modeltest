"""
Centralized ML training utilities.

- Builds features from cleaned dataset
- Evaluates candidate models with cross-validated predictions
- Selects best model by mean CV R2
- Safely persists `model_data.pkl` only when candidate is as-good-or-better
  than the existing saved best (prevents accidental downgrade)

This module avoids data leakage by applying scaling inside CV for models
that require it (LinearRegression) via a Pipeline + ColumnTransformer.
"""
from typing import Dict, Tuple, Any
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    denom = np.sum(np.abs(y_true - np.mean(y_true)))
    rae = float(np.sum(np.abs(y_true - y_pred)) / denom) if denom != 0 else float("nan")
    return {"r2": r2, "rmse": rmse, "mae": mae, "rae": rae}


def _ensure_engineered(df: pd.DataFrame) -> Tuple[pd.DataFrame, list, list]:
    # Ensure common engineered features are present, returning df, num_features, cat_features
    df = df.copy()
    if 'scheduled_dt' in df.columns:
        try:
            df['hour'] = pd.to_datetime(df['scheduled_dt']).dt.hour
        except Exception:
            df['hour'] = 0
    elif 'scheduled_time' in df.columns:
        try:
            df['scheduled_dt'] = pd.to_datetime(df['scheduled_time'], errors='coerce')
            df['hour'] = df['scheduled_dt'].dt.hour.fillna(0).astype(int)
        except Exception:
            df['hour'] = 0
    elif 'actual_time' in df.columns:
        try:
            df['hour'] = pd.to_datetime(df['actual_time']).dt.hour
        except Exception:
            df['hour'] = 0
    else:
        df['hour'] = 0

    if 'dow' in df.columns and 'is_weekend' not in df.columns:
        df['is_weekend'] = df['dow'].astype(int).apply(lambda x: 1 if x >= 5 else 0)
    elif 'is_weekend' not in df.columns:
        df['is_weekend'] = 0

    # cyclical encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    cat_features = ['route_id', 'weather', 'time_of_day']
    num_features = ['passenger_count', 'hour', 'is_weekend', 'hour_sin', 'hour_cos']
    if 'latitude' in df.columns and 'longitude' in df.columns:
        num_features += ['latitude', 'longitude']

    return df, num_features, cat_features


def compute_oof_aggregates(df: pd.DataFrame, cv: KFold) -> pd.DataFrame:
    """Compute out-of-fold aggregated features to avoid leakage during CV.

    Adds route-level stats, route-hour mean, and weather mean delay as OOF features.
    """
    df = df.copy()
    # prepare columns
    df['route_mean_delay'] = np.nan
    df['route_median_delay'] = np.nan
    df['route_std_delay'] = np.nan
    df['route_count'] = 0
    df['route_hour_mean'] = np.nan
    df['weather_mean_delay'] = np.nan

    # ensure required cols
    has_route = 'route_id' in df.columns
    has_hour = 'hour' in df.columns
    has_weather = 'weather' in df.columns

    for train_idx, val_idx in cv.split(df):
        tr = df.iloc[train_idx]
        if has_route:
            g = tr.groupby('route_id')['delay_minutes']
            mean_map = g.mean()
            med_map = g.median()
            std_map = g.std().fillna(0)
            count_map = g.count()
            df.iloc[val_idx, df.columns.get_loc('route_mean_delay')] = df.iloc[val_idx]['route_id'].map(mean_map).fillna(0).values
            df.iloc[val_idx, df.columns.get_loc('route_median_delay')] = df.iloc[val_idx]['route_id'].map(med_map).fillna(0).values
            df.iloc[val_idx, df.columns.get_loc('route_std_delay')] = df.iloc[val_idx]['route_id'].map(std_map).fillna(0).values
            # route_count
            df.iloc[val_idx, df.columns.get_loc('route_count')] = df.iloc[val_idx]['route_id'].map(count_map).fillna(0).values

        if has_route and has_hour:
            g2 = tr.groupby(['route_id', 'hour'])['delay_minutes'].mean()
            def _map_route_hour(row):
                return g2.get((row['route_id'], row['hour']), np.nan)
            df.iloc[val_idx, df.columns.get_loc('route_hour_mean')] = df.iloc[val_idx].apply(_map_route_hour, axis=1).fillna(0).values

        if has_weather:
            wm = tr.groupby('weather')['delay_minutes'].mean()
            df.iloc[val_idx, df.columns.get_loc('weather_mean_delay')] = df.iloc[val_idx]['weather'].map(wm).fillna(0).values

    # For any remaining missing values (e.g., rare categories), fill with global mean
    global_mean = df['delay_minutes'].mean()
    for c in ['route_mean_delay', 'route_median_delay', 'route_hour_mean', 'weather_mean_delay']:
        if c in df.columns:
            df[c] = df[c].fillna(global_mean)

    return df


def build_feature_matrices(df: pd.DataFrame, num_features: list, cat_features: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Tree models: keep all dummies
    available_cat = [c for c in cat_features if c in df.columns]
    df_cat = pd.get_dummies(df[available_cat], drop_first=False) if available_cat else pd.DataFrame(index=df.index)
    # Fit a scaler to numeric features (fit on full df here for app-serving; LR CV will scale inside pipeline)
    scaler = StandardScaler()
    # include any aggregate numeric features if present
    agg_cols = [c for c in ['route_mean_delay','route_median_delay','route_std_delay','route_count','route_hour_mean','weather_mean_delay','dow'] if c in df.columns and c not in num_features]
    num_features_local = list(num_features) + agg_cols
    df_num = df[num_features_local].fillna(0)
    scaler.fit(df_num)
    df_num_scaled = pd.DataFrame(scaler.transform(df_num), columns=num_features_local, index=df.index)

    X_tree = pd.concat([df_cat, df_num_scaled], axis=1)

    # LR: drop_first to avoid dummy trap, but keep same numeric scaled features
    X_lr_cat = pd.get_dummies(df[available_cat], drop_first=True) if available_cat else pd.DataFrame(index=df.index)
    X_lr = pd.concat([X_lr_cat, df_num_scaled], axis=1)

    return X_tree, X_lr, scaler


def evaluate_candidates(X_tree: pd.DataFrame, X_lr: pd.DataFrame, y: np.ndarray, cv_splits: int = 5) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    results = {}
    models = {}
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # Linear Regression pipeline: scale numeric columns inside CV to avoid leakage
    numeric_cols_lr = [c for c in X_lr.columns if c in ['passenger_count', 'hour', 'is_weekend', 'hour_sin', 'hour_cos', 'latitude', 'longitude']]

    if len(numeric_cols_lr) == 0:
        # fallback: assume last N columns are numeric by checking dtype
        numeric_cols_lr = [c for c in X_lr.columns if np.issubdtype(X_lr[c].dtype, np.number)]

    lr_preproc = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols_lr)
    ], remainder='passthrough')

    lr_pipeline = Pipeline([('pre', lr_preproc), ('lr', LinearRegression())])

    # Evaluate LR with cross_val_predict on X_lr
    try:
        preds_lr = cross_val_predict(lr_pipeline, X_lr, y, cv=cv, n_jobs=-1)
        results['lr'] = compute_metrics(y, preds_lr)
        models['lr'] = lr_pipeline
    except Exception as e:
        results['lr'] = {'error': str(e)}

    # Random Forest with light hyperparameter tuning (RandomizedSearchCV)
    try:
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [6, 10, 15, None],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.5]
        }
        rs = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_dist, n_iter=8, cv=cv, n_jobs=-1, random_state=42)
        rs.fit(X_tree, y)
        best_rf = rs.best_estimator_
        preds_rf = cross_val_predict(best_rf, X_tree, y, cv=cv, n_jobs=-1)
        results['rf'] = compute_metrics(y, preds_rf)
        results['rf']['best_params'] = rs.best_params_
        models['rf'] = best_rf
    except Exception as e:
        results['rf'] = {'error': str(e)}

    # HistGradientBoosting
    hgb = HistGradientBoostingRegressor(random_state=42)
    try:
        preds_hgb = cross_val_predict(hgb, X_tree, y, cv=cv, n_jobs=-1)
        results['hgb'] = compute_metrics(y, preds_hgb)
        models['hgb'] = hgb
    except Exception as e:
        results['hgb'] = {'error': str(e)}

    return results, models


def safe_save_model_data(candidate_md: Dict, candidate_best_r2: float, model_path: str = 'model_data.pkl', tol: float = 1e-6) -> Tuple[bool, str]:
    """
    Save `candidate_md` to `model_path` only if candidate_best_r2 is >= existing best.
    If existing model is better, write candidate to a timestamped backup and return False.
    Returns (saved: bool, message: str)
    """
    existing = {}
    if os.path.exists(model_path):
        try:
            existing = joblib.load(model_path)
        except Exception:
            existing = {}

    existing_best = existing.get('best_cv_r2', float('-inf'))
    now = datetime.utcnow().isoformat()
    if existing_best is not None and (candidate_best_r2 + tol) < existing_best:
        # Candidate is worse: do not overwrite main model_data.pkl
        fname = f"model_data_candidate_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump(candidate_md, fname)
        msg = f"Candidate best R2 {candidate_best_r2:.4f} is worse than existing {existing_best:.4f}. Saved candidate to {fname}"
        return False, msg

    # Candidate is good enough: persist and keep a backup of existing
    if os.path.exists(model_path):
        bak = f"model_data_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl"
        try:
            os.replace(model_path, bak)
        except Exception:
            # best-effort backup
            pass

    # record metadata
    candidate_md['best_cv_r2'] = float(candidate_best_r2)
    candidate_md['last_trained'] = now
    joblib.dump(candidate_md, model_path)
    return True, f"Saved new model_data.pkl with best_cv_r2={candidate_best_r2:.4f}"


def run_full_training(cleaned_csv: str = 'cleaned_transport_dataset.csv', cv_splits: int = 5) -> Tuple[bool, Dict]:
    df = pd.read_csv(cleaned_csv)
    if 'delay_minutes' not in df.columns:
        raise SystemExit('cleaned dataset must contain target column `delay_minutes`')

    df, num_features, cat_features = _ensure_engineered(df)
    # compute out-of-fold aggregates to avoid leakage during CV
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    df = compute_oof_aggregates(df, cv)

    X_tree, X_lr, scaler = build_feature_matrices(df, num_features, cat_features)
    y = df['delay_minutes'].values

    results, models = evaluate_candidates(X_tree, X_lr, y, cv_splits=cv_splits)

    # pick best by r2 (only among those that computed r2)
    best_name = None
    best_r2 = float('-inf')
    for name, res in results.items():
        if isinstance(res, dict) and 'r2' in res:
            if res['r2'] > best_r2:
                best_r2 = res['r2']
                best_name = name

    # Fit selected models on full data. First compute aggregates on full data to
    # provide stable features for the final fit.
    df_full = compute_oof_aggregates(df.copy(), KFold(n_splits=cv_splits, shuffle=True, random_state=42))
    # For full-data fit we actually want aggregates computed on full dataset
    # so recompute direct groupby stats (no OOF needed)
    if 'route_id' in df_full.columns:
        grp = df_full.groupby('route_id')['delay_minutes']
        df_full['route_mean_delay'] = df_full['route_id'].map(grp.mean()).fillna(df_full['delay_minutes'].mean())
        df_full['route_median_delay'] = df_full['route_id'].map(grp.median()).fillna(df_full['delay_minutes'].mean())
        df_full['route_std_delay'] = df_full['route_id'].map(grp.std()).fillna(0)
        df_full['route_count'] = df_full['route_id'].map(grp.count()).fillna(0)
        df_full['route_hour_mean'] = df_full.apply(lambda r: df_full[(df_full['route_id']==r['route_id']) & (df_full['hour']==r['hour'])]['delay_minutes'].mean() if True else np.nan, axis=1).fillna(df_full['delay_minutes'].mean())
    if 'weather' in df_full.columns:
        wf = df_full.groupby('weather')['delay_minutes'].mean()
        df_full['weather_mean_delay'] = df_full['weather'].map(wf).fillna(df_full['delay_minutes'].mean())

    X_tree_full, X_lr_full, scaler_full = build_feature_matrices(df_full, num_features, cat_features)

    # Fit selected models on full data
    model_objects = {}
    if 'lr' in models and best_name == 'lr':
        # fit LR pipeline on full lr feature matrix
        lr_pipeline = models['lr']
        lr_pipeline.fit(X_lr_full, y)
        model_objects['serving_model'] = lr_pipeline
        model_objects['lr_model'] = lr_pipeline
    elif 'lr' in models:
        # still fit lr for comparison
        models['lr'].fit(X_lr_full, y)
        model_objects['lr_model'] = models['lr']

    if 'rf' in models:
        models['rf'].fit(X_tree_full, y)
        model_objects['rf_model'] = models['rf']
    if 'hgb' in models:
        models['hgb'].fit(X_tree_full, y)
        model_objects['hgb_model'] = models['hgb']

    if best_name in model_objects:
        model_objects['serving_model'] = model_objects[best_name + '_model'] if best_name != 'lr' else model_objects['lr_model']
    else:
        # fallback: if lr existed, serve lr; else choose rf if available
        model_objects['serving_model'] = model_objects.get('lr_model') or model_objects.get('rf_model') or model_objects.get('hgb_model')

    # Build model_data structure
    model_data = {}
    model_data.update(model_objects)
    model_data['cv_metrics'] = results
    # Also store per-model metrics for compatibility
    model_data['lr_metrics'] = results.get('lr', {})
    model_data['rf_metrics'] = results.get('rf', {})
    model_data['hgb_metrics'] = results.get('hgb', {})
    # features for app alignment are based on tree X (used to build df_final in app)
    model_data['feature_columns'] = X_tree_full.columns.tolist()
    model_data['num_features'] = num_features
    model_data['scaler'] = scaler_full
    model_data['best_model_name'] = best_name

    saved, msg = safe_save_model_data(model_data, best_r2)
    return saved, {"message": msg, "best_name": best_name, "best_r2": best_r2, "cv_metrics": results}
