import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ml_pipeline import safe_save_model_data


def rae(y_true, y_pred):
    denom = np.sum(np.abs(y_true - np.mean(y_true)))
    if denom == 0:
        return np.nan
    return np.sum(np.abs(y_true - y_pred)) / denom


def compute_metrics(y, yhat):
    mse = mean_squared_error(y, yhat)
    return {
        'r2': float(r2_score(y, yhat)),
        'rmse': float(np.sqrt(mse)),
        'mae': float(mean_absolute_error(y, yhat)),
        'rae': float(rae(y, yhat)),
    }


def build_X(df, md):
    num_features = md.get('num_features', ['passenger_count', 'hour', 'is_weekend'])
    for c in num_features:
        if c not in df.columns:
            df[c] = 0

    cats = [c for c in ['route_id', 'weather', 'time_of_day'] if c in df.columns]
    Xcat = pd.get_dummies(df[cats]) if cats else pd.DataFrame(index=df.index)
    Xnum = df[num_features]
    X = pd.concat([Xcat, Xnum], axis=1)
    # align to saved feature_columns if present
    cols = md.get('feature_columns')
    if cols:
        X = X.reindex(columns=cols, fill_value=0)
    return X


def main():
    print('Retraining LinearRegression on full cleaned dataset...')
    df = pd.read_csv('cleaned_transport_dataset.csv')
    if 'delay_minutes' not in df.columns:
        raise SystemExit('cleaned dataset missing delay_minutes')

    md = {}
    try:
        md = joblib.load('model_data.pkl')
    except Exception:
        md = {}

    # ensure basic engineered features
    if 'hour' not in df.columns:
        try:
            df['hour'] = pd.to_datetime(df.get('scheduled_time', df.get('actual_time'))).dt.hour.fillna(0).astype(int)
        except Exception:
            df['hour'] = 0
    if 'is_weekend' not in df.columns and 'dow' in df.columns:
        df['is_weekend'] = df['dow'].astype(int).apply(lambda x: 1 if x >= 5 else 0)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    X = build_X(df, md)
    y = df['delay_minutes'].values

    # Use log1p transform to ensure LR predictions are positive after inverse transform.
    use_log = True
    if use_log:
        y_train = np.log1p(y)
    else:
        y_train = y

    # If a scaler exists in model_data, apply it to numeric features so LR is
    # trained on the same scaled features used at prediction time in the app.
    scaler = md.get('scaler')
    num_features = md.get('num_features', ['passenger_count', 'hour', 'is_weekend'])
    try:
        if scaler is not None and num_features:
            # ensure columns exist in X
            for c in num_features:
                if c not in X.columns:
                    X[c] = df.get(c, 0)
            X[num_features] = scaler.transform(df[num_features])
    except Exception:
        pass

    lr = LinearRegression()
    lr.fit(X, y_train)

    # compute in-sample predictions and metrics on original scale
    if use_log:
        preds_trans = lr.predict(X)
        preds = np.expm1(preds_trans)
    else:
        preds = lr.predict(X)
    m = compute_metrics(y, preds)
    print('LR full-fit metrics:', m)

    md['lr_model'] = lr
    md['lr_metrics'] = m
    # record that lr was trained on log1p target
    md['lr_target_transform'] = 'log1p' if use_log else 'none'
    # keep feature_columns if present
    if 'feature_columns' not in md:
        md['feature_columns'] = X.columns.tolist()

    # Use safe save to avoid overwriting a better existing model
    saved, msg = safe_save_model_data(md, candidate_best_r2=m.get('r2', float('-inf')))
    if saved:
        print('Updated model_data.pkl with lr_model and lr_metrics')
    else:
        print('Did NOT overwrite model_data.pkl:', msg)


if __name__ == '__main__':
    main()
