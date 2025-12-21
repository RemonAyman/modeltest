import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from ml_pipeline import safe_save_model_data


def rae(y_true, y_pred):
    import joblib
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from ml_pipeline import safe_save_model_data


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


    def build_feature_matrix(df, num_features, feature_columns=None):
        # ensure num features exist
        for c in num_features:
            if c not in df.columns:
                df[c] = 0

        cat_cols = [c for c in ['route_id', 'weather', 'time_of_day'] if c in df.columns]
        df_cat = pd.get_dummies(df[cat_cols]) if cat_cols else pd.DataFrame(index=df.index)
        df_num = df[num_features]
        X = pd.concat([df_cat, df_num], axis=1)
        if feature_columns:
            X = X.reindex(columns=feature_columns, fill_value=0)
        return X


    def main():
        print('Loading cleaned dataset...')
        df = pd.read_csv('cleaned_transport_dataset.csv')
        if 'delay_minutes' not in df.columns:
            raise SystemExit('cleaned_transport_dataset.csv missing target column delay_minutes')

        # Load existing model_data if available to preserve columns/scaler
        try:
            md = joblib.load('model_data.pkl')
            print('Loaded existing model_data.pkl')
        except Exception:
            md = {}

        num_features = md.get('num_features', ['passenger_count', 'hour', 'is_weekend'])

        # Ensure engineered time features exist in cleaned DF
        if 'scheduled_time' in df.columns:
            try:
                df['hour'] = pd.to_datetime(df['scheduled_time']).dt.hour
            except Exception:
                df['hour'] = 0
        elif 'actual_time' in df.columns:
            try:
                df['hour'] = pd.to_datetime(df['actual_time']).dt.hour
            except Exception:
                df['hour'] = 0
        else:
            df['hour'] = 0

        # is_weekend from dow if available (dow: 0-6), else default 0
        if 'dow' in df.columns:
            try:
                df['is_weekend'] = df['dow'].astype(int).apply(lambda x: 1 if x >= 5 else 0)
            except Exception:
                df['is_weekend'] = 0
        else:
            df['is_weekend'] = 0

        # cyclic hour features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Fit scaler for numeric features
        scaler = md.get('scaler')
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(df[num_features])

        df[num_features] = scaler.transform(df[num_features])

        # Build X and y
        feature_columns = md.get('feature_columns')
        X = build_feature_matrix(df, num_features, feature_columns)
        y = df['delay_minutes'].values

        # If feature_columns were missing, capture current columns
        if feature_columns is None:
            feature_columns = X.columns.tolist()

        print(f'Training RandomForest on {X.shape[0]} samples, {X.shape[1]} features')
        rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)

        # cross-validated predictions to compute reliable metrics
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        preds = cross_val_predict(rf, X, y, cv=cv, n_jobs=-1)

        metrics = compute_metrics(y, preds)
        print('CV RF metrics:', metrics)

        # fit on full data
        rf.fit(X, y)

        # prepare candidate model_data
        md['rf_model'] = rf
        md['rf_metrics'] = metrics
        md['feature_columns'] = feature_columns
        md['scaler'] = scaler
        md['num_features'] = num_features

        # Use safe save to avoid overwriting a better existing model
        saved, msg = safe_save_model_data(md, candidate_best_r2=metrics.get('r2', float('-inf')))
        if saved:
            print('Updated model_data.pkl with retrained rf_model and rf_metrics')
        else:
            print('Did NOT overwrite model_data.pkl:', msg)


    if __name__ == '__main__':
        main()
