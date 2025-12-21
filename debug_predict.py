import joblib
import pandas as pd
import numpy as np

# Sample input matching the dashboard screenshot
sample = {
    'route': 'R1',
    'weather': 'sunny',
    'time': '18:39',
    'day_type': 'weekday',
    'passengers': 50,
    'latitude': 0,
    'longitude': 0,
}

md = joblib.load('model_data.pkl')
serving_model = md.get('serving_model')
rf_model = md.get('rf_model')
lr_model = md.get('lr_model')
feature_columns = md.get('feature_columns', [])
scaler = md.get('scaler')
num_features = md.get('num_features', ['passenger_count', 'hour', 'is_weekend'])
route_stats = md.get('route_stats')

# Build input as in app.py
hour = int(sample['time'].split(':')[0])
if 5 <= hour < 12:
    tod_str = 'Morning'
elif 12 <= hour < 17:
    tod_str = 'Afternoon'
elif 17 <= hour < 21:
    tod_str = 'Evening'
else:
    tod_str = 'Night'

input_data = {
    'route_id': [sample['route']],
    'weather': [sample['weather'].lower()],
    'time_of_day': [tod_str],
    'passenger_count': [int(sample.get('passengers', 56))],
    'hour': [hour],
    'is_weekend': [1 if sample.get('day_type') == 'weekend' else 0]
}

df_input = pd.DataFrame(input_data)

# cyclical
if 'hour_sin' in num_features and 'hour_sin' not in df_input.columns:
    df_input['hour_sin'] = np.sin(2 * np.pi * df_input['hour'] / 24)
if 'hour_cos' in num_features and 'hour_cos' not in df_input.columns:
    df_input['hour_cos'] = np.cos(2 * np.pi * df_input['hour'] / 24)

# route aggregates from model_data if present
# model_data may have route_stats or we can try to use cleaned csv maps
route_hour_map = {}
weather_mean_map = {}
try:
    df_clean = pd.read_csv('cleaned_transport_dataset.csv')
    route_hour_map = df_clean.set_index(['route_id', 'hour'])['route_hour_mean'].to_dict()
    weather_mean_map = df_clean.groupby('weather')['weather_mean_delay'].mean().to_dict()
except Exception:
    pass

if 'route_mean_delay' in num_features:
    # we don't have route_stats_df here; try to read maps
    def _route_mean(r):
        return route_hour_map.get((r, hour), 0)
    df_input['route_mean_delay'] = df_input['route_id'].apply(lambda r: 0)
    df_input['route_median_delay'] = df_input['route_id'].apply(lambda r: 0)
    df_input['route_std_delay'] = df_input['route_id'].apply(lambda r: 0)
    df_input['route_count'] = df_input['route_id'].apply(lambda r: 0)

if 'route_hour_mean' in num_features:
    df_input['route_hour_mean'] = df_input.apply(lambda row: route_hour_map.get((row['route_id'], int(row['hour'])), row.get('route_mean_delay', 0)), axis=1)

if 'weather_mean_delay' in num_features:
    df_input['weather_mean_delay'] = df_input['weather'].apply(lambda w: weather_mean_map.get(w, 0))

# fillers for lat/long
if 'latitude' in num_features and 'latitude' not in df_input.columns:
    df_input['latitude'] = float(sample.get('latitude', 0) or 0)
if 'longitude' in num_features and 'longitude' not in df_input.columns:
    df_input['longitude'] = float(sample.get('longitude', 0) or 0)

# scale numeric features if scaler present
if scaler:
    for c in num_features:
        if c not in df_input.columns:
            df_input[c] = 0
    try:
        df_input[num_features] = scaler.transform(df_input[num_features])
    except Exception:
        pass

# one-hot and align
cat_features = ['route_id', 'weather', 'time_of_day']
df_encoded = pd.get_dummies(df_input[cat_features])
df_final = pd.concat([df_encoded, df_input[num_features]], axis=1)

# align to feature columns
if not feature_columns:
    print('No feature_columns in model_data.pkl')
else:
    df_final = df_final.reindex(columns=feature_columns, fill_value=0)

print('INPUT RAW:')
print(input_data)
print('\nFEATURE VALUES (pre-scaled where relevant):')
print(df_input.to_dict(orient='records')[0])
print('\nALIGNED FEATURES (after one-hot & scale):')
print(df_final.T[df_final.index if hasattr(df_final,'index') else df_final.columns].iloc[:50])

# Predictions
if serving_model is not None:
    try:
        svc_raw = serving_model.predict(df_final)[0]
        print('\nSERVING MODEL PREDICTION:', svc_raw)
    except Exception as e:
        print('Serving model predict error:', e)

if rf_model is not None:
    try:
        rf_raw = rf_model.predict(df_final)[0]
        print('RF PREDICTION:', rf_raw)
    except Exception as e:
        print('RF predict error:', e)

if lr_model is not None:
    try:
        lr_raw = lr_model.predict(df_final)[0]
        print('LR PREDICTION (raw):', lr_raw)
        # compute LR contributions if coefficients available
        if hasattr(lr_model, 'coef_'):
            coefs = np.array(lr_model.coef_).ravel()
            cols = feature_columns
            vals = df_final[cols].iloc[0].values.astype(float)
            contrib = coefs * vals
            idxs = np.argsort(np.abs(contrib))[::-1][:10]
            print('\nTop LR contributions (coef * value):')
            for i in idxs:
                print(f"{cols[i]}: {contrib[i]:+.3f} (coef={coefs[i]:.3f}, val={vals[i]:.3f})")
        elif hasattr(lr_model, 'named_steps'):
            # pipeline case
            try:
                lr_step = lr_model.named_steps['lr']
                pre = lr_model.named_steps['pre']
                # use pre.transform to get transformed features
                Xt = pre.transform(df_final)
                coefs = np.array(lr_step.coef_).ravel()
                vals = Xt[0]
                contrib = coefs * vals
                idxs = np.argsort(np.abs(contrib))[::-1][:10]
                print('\nTop LR contributions (pipeline coef * value):')
                for i in idxs:
                    print(f"feat_idx_{i}: {contrib[i]:+.3f} (coef={coefs[i]:.3f}, val={vals[i]:.3f})")
            except Exception as e:
                print('Could not compute pipeline contributions:', e)
    except Exception as e:
        print('LR predict error:', e)

# RF top feature importances and sample values
if rf_model is not None and hasattr(rf_model, 'feature_importances_'):
    try:
        fi = np.array(rf_model.feature_importances_)
        cols = feature_columns
        idxs = np.argsort(fi)[::-1][:10]
        print('\nRF top feature importances and sample values:')
        for i in idxs:
            feat = cols[i]
            print(f"{feat}: importance={fi[i]:.4f}, value={df_final.get(feat, pd.Series([0])).iloc[0]}")
    except Exception as e:
        print('RF explanation error:', e)
