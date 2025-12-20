import pandas as pd
import numpy as np
import joblib

ok = True
print('Check started')
# Load cleaned data
try:
    df = pd.read_csv('cleaned_transport_dataset.csv')
    print('cleaned df shape:', df.shape)
except Exception as e:
    print('Failed to load cleaned CSV:', e)
    raise SystemExit(1)

# Checks
checks = []
# delay_minutes
if 'delay_minutes' in df.columns:
    mn = df['delay_minutes'].min()
    mx = df['delay_minutes'].max()
    checks.append(('delay_minutes_min', mn))
    checks.append(('delay_minutes_max', mx))
    neg_count = int((df['delay_minutes'] < 0).sum())
    checks.append(('delay_minutes_negative_count', neg_count))
else:
    checks.append(('delay_minutes_missing', True))

# passenger_count
if 'passenger_count' in df.columns:
    mn = df['passenger_count'].min()
    mx = df['passenger_count'].max()
    checks.append(('passenger_count_min', mn))
    checks.append(('passenger_count_max', mx))
    bad = int(((df['passenger_count'] < 1) | (df['passenger_count'] > 200)).sum())
    checks.append(('passenger_count_out_of_bounds', bad))

# lat/lon
if 'latitude' in df.columns and 'longitude' in df.columns:
    checks.append(('latitude_min', float(df['latitude'].min())))
    checks.append(('latitude_max', float(df['latitude'].max())))
    checks.append(('longitude_min', float(df['longitude'].min())))
    checks.append(('longitude_max', float(df['longitude'].max())))
    lat_bad = int(((df['latitude'].abs() > 90) | (df['longitude'].abs() > 180)).sum())
    checks.append(('latlon_out_of_bounds', lat_bad))

# dow
if 'dow' in df.columns:
    unique = sorted(df['dow'].dropna().unique().tolist())
    checks.append(('dow_unique', unique))

# route_count
if 'route_count' in df.columns:
    checks.append(('route_count_min', int(df['route_count'].min())))

# NaNs in key features
nan_counts = df.isna().sum()
checks.append(('nan_counts_head', nan_counts[nan_counts>0].to_dict()))

# Print checks
for k,v in checks:
    print(k, v)

# Load model_data
try:
    md = joblib.load('model_data.pkl')
    print('model_data keys:', list(md.keys()))
except Exception as e:
    print('Failed to load model_data.pkl:', e)
    raise SystemExit(1)

# metrics sanity
for mname in ['rf_metrics','hgb_metrics','lr_metrics']:
    if mname in md:
        m = md[mname]
        rmse = m.get('rmse')
        mae = m.get('mae')
        rae = m.get('rae')
        r2 = m.get('r2')
        print(f"{mname}: r2={r2}, rmse={rmse}, mae={mae}, rae={rae}")

# serving model predictions check
serv = md.get('serving_model') or md.get('best_model') or md.get('rf_model')
lr_model = md.get('lr_model')
if serv is None:
    print('No serving model in model_data')
else:
    try:
        Xcols = md.get('feature_columns', [])
        # Load features from cleaned df (align)
        X = pd.read_csv('cleaned_transport_dataset.csv')
        # Prepare dataframe with only model features, fill missing with 0
        Xf = pd.DataFrame(columns=Xcols)
        for c in Xcols:
            if c in X.columns:
                Xf[c] = X[c]
            else:
                Xf[c] = 0
        # ensure numeric
        Xf = Xf.fillna(0)
        preds = serv.predict(Xf)
        print('serving_model preds min/max:', float(np.min(preds)), float(np.max(preds)))
        negp = int((preds < 0).sum())
        print('serving_model negative_predictions_count:', negp)
    except Exception as e:
        print('Error running serving_model on cleaned data:', e)

if lr_model is not None:
    try:
        Xcols = md.get('feature_columns', [])
        Xf = pd.DataFrame(columns=Xcols)
        X = pd.read_csv('cleaned_transport_dataset.csv')
        for c in Xcols:
            if c in X.columns:
                Xf[c] = X[c]
            else:
                Xf[c] = 0
        Xf = Xf.fillna(0)
        preds = lr_model.predict(Xf)
        print('lr_model preds min/max:', float(np.min(preds)), float(np.max(preds)))
        negp = int((preds < 0).sum())
        print('lr_model negative_predictions_count:', negp)
    except Exception as e:
        print('Error running lr_model on cleaned data:', e)

print('Check finished')
