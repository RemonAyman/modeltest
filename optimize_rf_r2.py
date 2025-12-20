import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def rae(y_true, y_pred):
    denom = np.sum(np.abs(y_true - np.mean(y_true)))
    if denom == 0:
        return np.nan
    return np.sum(np.abs(y_true - y_pred)) / denom


def metrics(y, yhat):
    mse = mean_squared_error(y, yhat)
    return {
        'r2': float(r2_score(y, yhat)),
        'rmse': float(np.sqrt(mse)),
        'mae': float(mean_absolute_error(y, yhat)),
        'rae': float(rae(y, yhat)),
    }


def build_features(df, md):
    cols = md.get('feature_columns')
    # ensure engineered fields
    if 'hour' not in df.columns:
        try:
            df['hour'] = pd.to_datetime(df.get('scheduled_time', df.get('actual_time'))).dt.hour.fillna(0).astype(int)
        except Exception:
            df['hour'] = 0
    if 'is_weekend' not in df.columns and 'dow' in df.columns:
        df['is_weekend'] = df['dow'].astype(int).apply(lambda x: 1 if x >= 5 else 0)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    num_features = md.get('num_features', ['passenger_count', 'hour', 'is_weekend'])
    for c in num_features:
        if c not in df.columns:
            df[c] = 0

    cats = [c for c in ['route_id', 'weather', 'time_of_day'] if c in df.columns]
    Xcat = pd.get_dummies(df[cats]) if cats else pd.DataFrame(index=df.index)
    Xnum = df[num_features]
    X = pd.concat([Xcat, Xnum], axis=1)
    if cols:
        X = X.reindex(columns=cols, fill_value=0)
    return X


def run_experiments():
    df = pd.read_csv('cleaned_transport_dataset.csv')
    if 'delay_minutes' not in df.columns:
        raise SystemExit('cleaned dataset missing delay_minutes')

    try:
        md = joblib.load('model_data.pkl')
    except Exception:
        md = {}

    X = build_features(df, md)
    y = df['delay_minutes'].values

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    candidates = {
        'rf': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        'hgb': HistGradientBoostingRegressor(random_state=42),
    }

    results = {}

    # Baseline on raw target
    for name, model in candidates.items():
        preds = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
        results.setdefault(name, {})['raw'] = metrics(y, preds)

    # Try log1p target (train on log1p(y), invert predictions with expm1)
    y_log = np.log1p(y)
    for name, model in candidates.items():
        preds_log = cross_val_predict(model, X, y_log, cv=cv, n_jobs=-1)
        preds = np.expm1(preds_log)
        results.setdefault(name, {})['log1p'] = metrics(y, preds)

    # Try trimming extremes (remove top 1% and bottom 1%)
    q_low, q_high = np.percentile(y, [1, 99])
    mask = (y >= q_low) & (y <= q_high)
    X_trim = X[mask]
    y_trim = y[mask]
    for name, model in candidates.items():
        preds = cross_val_predict(model, X_trim, y_trim, cv=cv, n_jobs=-1)
        results.setdefault(name, {})['trim_1pct'] = metrics(y_trim, preds)

    # Ensemble average with LinearRegression (avg of LR + model)
    lr = LinearRegression()
    lr_preds = cross_val_predict(lr, X, y, cv=cv, n_jobs=-1)
    for name, model in candidates.items():
        m_preds = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
        avg = 0.5 * (lr_preds + m_preds)
        results.setdefault(name, {})['ensemble_lr_avg'] = metrics(y, avg)

    # Summarize and optionally persist best rf variant
    print('Experiments summary:')
    for name, info in results.items():
        print('\nModel:', name)
        for k, v in info.items():
            print(f'  {k}: r2={v["r2"]:.4f}, rmse={v["rmse"]:.2f}, mae={v["mae"]:.2f}, rae={v["rae"]:.3f}')

    # Choose best RF variant (by r2) and save rf_model trained on that transform/full data
    best_rf_variant = max(results['rf'].items(), key=lambda kv: kv[1]['r2'])[0]
    print('\nBest RF variant:', best_rf_variant, 'metrics:', results['rf'][best_rf_variant])

    # Fit final RF on full data using chosen strategy
    final_rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    if best_rf_variant == 'raw':
        final_rf.fit(X, y)
        final_preds = final_rf.predict(X)
    elif best_rf_variant == 'log1p':
        final_rf.fit(X, np.log1p(y))
        final_preds = np.expm1(final_rf.predict(X))
    elif best_rf_variant == 'trim_1pct':
        final_rf.fit(X_trim, y_trim)
        final_preds = final_rf.predict(X)
    elif best_rf_variant == 'ensemble_lr_avg':
        # save RF trained on raw (for consistency) and we store ensemble metric only
        final_rf.fit(X, y)
        final_preds = final_rf.predict(X)
    else:
        final_rf.fit(X, y)
        final_preds = final_rf.predict(X)

    # compute final metrics on full data (where applicable)
    final_metrics = metrics(y, final_preds)
    print('\nFinal RF trained on', best_rf_variant, '->', final_metrics)

    # persist to model_data.pkl (update rf_model and rf_metrics)
    md['rf_model'] = final_rf
    md['rf_metrics'] = final_metrics
    joblib.dump(md, 'model_data.pkl')
    print('Updated model_data.pkl with rf_model and rf_metrics')


if __name__ == '__main__':
    run_experiments()
