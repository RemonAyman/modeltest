import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')


def rae(y_true, y_pred):
    denom = np.sum(np.abs(y_true - np.mean(y_true)))
    if denom == 0:
        return np.nan
    return np.sum(np.abs(y_true - y_pred)) / denom


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"r2": r2, "rmse": rmse, "mae": mae, "rae": float(rae(y_true, y_pred))}


def load_and_prepare():
    df = pd.read_csv('cleaned_transport_dataset.csv')
    if 'delay_minutes' not in df.columns:
        raise SystemExit('missing target delay_minutes in cleaned CSV')

    # ensure hour/is_weekend and cyclic features exist
    if 'hour' not in df.columns:
        try:
            df['hour'] = pd.to_datetime(df.get('scheduled_time', df.get('actual_time'))).dt.hour.fillna(0).astype(int)
        except Exception:
            df['hour'] = 0
    if 'dow' in df.columns and 'is_weekend' not in df.columns:
        df['is_weekend'] = df['dow'].astype(int).apply(lambda x: 1 if x >= 5 else 0)

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    md = {}
    try:
        md = joblib.load('model_data.pkl')
    except Exception:
        md = {}

    num_features = md.get('num_features', ['passenger_count', 'hour', 'is_weekend'])
    # ensure num_features present
    for c in num_features:
        if c not in df.columns:
            df[c] = 0

    # one-hot cats
    cats = [c for c in ['route_id', 'weather', 'time_of_day'] if c in df.columns]
    X_cat = pd.get_dummies(df[cats]) if cats else pd.DataFrame(index=df.index)
    X_num = df[num_features]
    X = pd.concat([X_cat, X_num], axis=1)
    y = df['delay_minutes'].values
    return X, y, df, md


def run_search(X, y):
    # Use KFold CV and R2 as scoring
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scorer = make_scorer(r2_score)

    results = {}

    # 1) RandomForest baseline + randomized search optimizing R2
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_param = {
        'n_estimators': randint(80, 400),
        'max_depth': randint(4, 30),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 6),
        'max_features': ['sqrt', 'log2', None]
    }
    rsearch = RandomizedSearchCV(rf, rf_param, n_iter=40, scoring=r2_scorer, cv=cv, random_state=42, n_jobs=-1)
    rsearch.fit(X, y)
    best_rf = rsearch.best_estimator_
    preds = cross_val_score(best_rf, X, y, cv=cv, scoring=r2_scorer)
    results['rf'] = {'model': best_rf, 'cv_r2_mean': float(np.mean(preds)), 'cv_r2_scores': preds, 'search': rsearch}

    # 2) HistGradientBoostingRegressor (fast) search
    try:
        hgb_param = {'max_iter': randint(50, 400), 'max_depth': randint(3, 15), 'learning_rate': uniform(0.01, 0.5)}
        hgb = HistGradientBoostingRegressor(random_state=42)
        hsearch = RandomizedSearchCV(hgb, hgb_param, n_iter=30, scoring=r2_scorer, cv=cv, random_state=42, n_jobs=-1)
        hsearch.fit(X, y)
        best_hgb = hsearch.best_estimator_
        preds = cross_val_score(best_hgb, X, y, cv=cv, scoring=r2_scorer)
        results['hgb'] = {'model': best_hgb, 'cv_r2_mean': float(np.mean(preds)), 'cv_r2_scores': preds, 'search': hsearch}
    except Exception as e:
        results['hgb'] = {'error': str(e)}

    # 3) LinearRegression baseline (no search) - check performance
    lr = LinearRegression()
    scores = cross_val_score(lr, X, y, cv=cv, scoring=r2_scorer)
    results['lr'] = {'model': lr.fit(X, y), 'cv_r2_mean': float(np.mean(scores)), 'cv_r2_scores': scores}

    return results


def evaluate_results(results, X, y):
    # Compute metrics on CV predictions for each candidate
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    final = {}
    for name, info in results.items():
        if 'model' not in info:
            final[name] = info
            continue
        model = info['model']
        from sklearn.model_selection import cross_val_predict
        preds = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
        metrics = compute_metrics(y, preds)
        final[name] = {'model': model, 'metrics': metrics}
    return final


def main():
    print('Preparing data...')
    X, y, df, md = load_and_prepare()
    print('Data shape:', X.shape)

    print('Running model search optimizing CV R2...')
    results = run_search(X, y)

    print('Evaluating candidates with CV predict metrics...')
    final = evaluate_results(results, X, y)

    # choose best by CV R2
    best_name = None
    best_r2 = -999
    for k, v in final.items():
        if 'metrics' in v and v['metrics'] and v['metrics'].get('r2') is not None:
            r2v = v['metrics']['r2']
            if r2v > best_r2:
                best_r2 = r2v
                best_name = k

    print('Final CV R2 by model:')
    for k, v in final.items():
        print(k, ':', v.get('metrics'))

    print('Selected best by CV R2:', best_name, 'r2=', best_r2)

    # update model_data.pkl serving model to best candidate
    md.setdefault('rf_metrics', md.get('rf_metrics'))
    md.setdefault('lr_metrics', md.get('lr_metrics'))
    md.setdefault('hgb_metrics', md.get('hgb_metrics'))
    # store candidate models + metrics
    for k, v in final.items():
        if 'metrics' in v:
            md[f'{k}_metrics'] = v['metrics']
            md[f'{k}_model'] = v['model']

    if best_name:
        md['serving_model'] = md.get(f'{best_name}_model')
        md['best_model_name'] = best_name

    joblib.dump(md, 'model_data.pkl')
    print('Saved model_data.pkl with updated serving_model and metrics')


if __name__ == '__main__':
    main()
