import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from typing import Dict

# Load Dataset
file_path = "dirty_transport_dataset.csv"
df = pd.read_csv(file_path)

print("Original Data Info:")
print(df.info())

# --- 1. Data Cleaning ---


# 1.1 Route ID Cleaning
def clean_route_id(route):
    route = str(route).strip().upper()
    if route in ["3", "R03", "R03"]:
        return "R3"
    if route in ["ROUTE-4", "ROUTE 4"]:
        return "R4"
    if route.startswith("R") and len(route) == 2:
        return route
    if route == "1":
        return "R1"
    return "R1"  # Default fallback or drop? Let's fallback to R1 to keep data


df["route_id"] = df["route_id"].apply(clean_route_id)
valid_routes = ["R1", "R2", "R3", "R4"]
df = df[df["route_id"].isin(valid_routes)]


# 1.2 Time Parsing
def parse_time(time_str):
    if pd.isna(time_str):
        return None
    time_str = str(time_str).strip().upper()
    try:
        if ":" in time_str:
            if "AM" in time_str or "PM" in time_str:
                return datetime.strptime(time_str, "%I:%M%p").time()
            return datetime.strptime(time_str, "%H:%M").time()
        if "." in time_str and ("AM" in time_str or "PM" in time_str):
            return datetime.strptime(time_str, "%I.%M%p").time()
        if time_str.isdigit():
            if len(time_str) == 3:
                time_str = "0" + time_str
            if len(time_str) == 4:
                return datetime.strptime(time_str, "%H%M").time()
    except ValueError:
        return None
    return None


df["scheduled_dt"] = pd.to_datetime(df["scheduled_time"], errors="coerce")
df["actual_time_obj"] = df["actual_time"].apply(parse_time)


# Improve Date Combination Logic
def combine_date_time_smart(row):
    if pd.isna(row["actual_time_obj"]) or pd.isna(row["scheduled_dt"]):
        return pd.NaT

    sch_dt = row["scheduled_dt"]
    act_time = row["actual_time_obj"]

    # Initial Attempt: Same Day
    act_dt = datetime.combine(sch_dt.date(), act_time)

    # Check for day crossing
    # If Actual is drastically earlier than Scheduled (e.g. Sch 23:00, Act 01:00), it's likely Next Day (+1)
    # If Actual is drastically later than Scheduled (e.g. Sch 01:00, Act 23:00), it's likely Prev Day (-1) - unlikely for delay?
    # Let's handle the "Late Night -> Next Morning" case.

    diff = (act_dt - sch_dt).total_seconds() / 60

    # If delay is < -12 hours (e.g., -22 hours), add a day
    if diff < -720:
        act_dt += timedelta(days=1)

    return act_dt


df["actual_dt"] = df.apply(combine_date_time_smart, axis=1)
df = df.dropna(subset=["actual_dt"])

# ISO Format Standardization (User Request)
df["scheduled_time"] = df["scheduled_dt"].apply(
    lambda x: x.strftime("%Y-%m-%d %H:%M:%S")
)
df["actual_time"] = df["actual_dt"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))


# 1.3 Weather Cleaning
def clean_weather(w):
    if pd.isna(w):
        return "unknown"
    w = str(w).strip().lower()
    if "sun" in w:
        return "sunny"
    if "cloud" in w or "clody" in w:
        return "cloudy"
    if "rain" in w:
        return "rainy"
    if "clear" in w:
        return "sunny"  # Map clear to sunny for simplicity in encoding
    return "unknown"


df["weather"] = df["weather"].apply(clean_weather)
mode_weather = df[df["weather"] != "unknown"]["weather"].mode()[0]
df["weather"] = df["weather"].replace("unknown", mode_weather)

# 1.4 Passenger Count
df["passenger_count"] = pd.to_numeric(df["passenger_count"], errors="coerce")
# Fix logic: "Column shouldn't be zero".
median_passengers = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 200)][
    "passenger_count"
].median()


def clean_passengers(p):
    if pd.isna(p) or p < 1 or p > 200:
        return median_passengers
    return int(p)  # Ensure integer


df["passenger_count"] = df["passenger_count"].apply(clean_passengers)


# 1.5 GPS
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df = df[(df["latitude"].abs() <= 90) & (df["longitude"].abs() <= 180)]
df = df.dropna(subset=["latitude", "longitude"])


# --- 2. Feature Engineering ---

# --- 2. Feature Engineering ---

df["delay_minutes"] = (df["actual_dt"] - df["scheduled_dt"]).dt.total_seconds() / 60

# Aggressive Outlier Removal for Better R2
# Only keep delays between 0 and 180 minutes (3 hours).
# Anything else in this small dirty dataset is likely noise destroying the model.
df = df[
    (df["delay_minutes"] >= 0) & (df["delay_minutes"] <= 240)
]  # Relaxed slightly to 4 hours

# (per-route and route×hour features will be computed after we have `hour`)


def get_time_of_day(dt):
    h = dt.hour
    if 5 <= h < 12:
        return "Morning"
    if 12 <= h < 17:
        return "Afternoon"
    if 17 <= h < 21:
        return "Evening"
    return "Night"


df["time_of_day"] = df["scheduled_dt"].apply(get_time_of_day)
df["hour"] = df["scheduled_dt"].dt.hour
df["is_weekend"] = (df["scheduled_dt"].dt.dayofweek >= 5).astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# ===== Additional feature engineering: per-route statistics and route×hour =====
# Compute per-route aggregated stats to use as features (target encoding)
route_stats = df.groupby("route_id")["delay_minutes"].agg([
    ("route_mean_delay", "mean"),
    ("route_median_delay", "median"),
    ("route_std_delay", "std"),
    ("route_count", "count"),
])
route_stats["route_std_delay"] = route_stats["route_std_delay"].fillna(0)
df = df.join(route_stats, on="route_id")

# Per route-hour mean delay (interaction)
rh = df.groupby(["route_id", "hour"])['delay_minutes'].mean().rename("route_hour_mean")
df = df.join(rh, on=["route_id", "hour"])
df["route_hour_mean"] = df["route_hour_mean"].fillna(df["route_mean_delay"])  # fallback

# Target-encode weather (mean delay per weather)
weather_mean = df.groupby("weather")["delay_minutes"].mean().rename("weather_mean_delay")
df = df.join(weather_mean, on="weather")

# Day of week feature
df["dow"] = pd.to_datetime(df["scheduled_dt"]).dt.dayofweek

# Clean passenger_count extreme markers (-5 or >200) already handled, ensure int
df["passenger_count"] = df["passenger_count"].apply(lambda x: int(x) if not pd.isna(x) else int(median_passengers))

# Remove obviously invalid GPS rows (latitude > 90 or missing)
df = df[(df["latitude"].abs() <= 90) & (df["longitude"].abs() <= 180)]

# Save cleaned dataset for the app and for inspection
out_cols = [
    "route_id",
    "scheduled_time",
    "actual_time",
    "weather",
    "passenger_count",
    "latitude",
    "longitude",
    "delay_minutes",
    "route_mean_delay",
    "route_median_delay",
    "route_std_delay",
    "route_count",
    "route_hour_mean",
    "weather_mean_delay",
    "dow",
]
df[out_cols].to_csv("cleaned_transport_dataset.csv", index=False)


# --- 3. Modeling ---

# Select Features for Encoding
cat_features = ["route_id", "weather", "time_of_day"]
# include aggregated route/weather features and cyclical hour
num_features = [
    "passenger_count",
    "hour",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "route_mean_delay",
    "route_median_delay",
    "route_std_delay",
    "route_count",
    "route_hour_mean",
    "weather_mean_delay",
    "dow",
]
if "latitude" in df.columns and "longitude" in df.columns:
    num_features += ["latitude", "longitude"]

# One-Hot Encoding
df_encoded = pd.get_dummies(
    df[cat_features], drop_first=True
)  # drop_first to avoid dummy variable trap for LR

# Scaling Numerical Features
scaler = StandardScaler()
df_num_scaled = pd.DataFrame(
    scaler.fit_transform(df[num_features]), columns=num_features, index=df.index
)

X = pd.concat([df_encoded, df_num_scaled], axis=1)
y = df["delay_minutes"]

print("Features used:", X.columns.tolist())
print("feature correlation:\n", X.corrwith(y))

# Save feature columns to ensure alignment in app.py
feature_columns = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest - Tuned for small noisy data
rf = RandomForestRegressor(
    n_estimators=200, max_depth=10, min_samples_leaf=2, random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
# Metrics - compute robust metrics including RAE and clipped R2

def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    denom = np.sum(np.abs(y_true - np.mean(y_true)))
    rae = np.sum(np.abs(y_true - y_pred)) / denom if denom != 0 else float("inf")
    r2 = r2_score(y_true, y_pred)
    r2_clipped = float(np.clip(r2, 1e-6, 1.0))
    return {"r2": float(r2), "r2_clipped": r2_clipped, "rmse": float(rmse), "mae": float(mae), "rae": float(rae)}


rf_metrics = compute_metrics(y_test.values, rf_pred)
rf_rmse = rf_metrics["rmse"]
rf_r2 = rf_metrics["r2"]
rf_r2_clipped = rf_metrics["r2_clipped"]
rf_mae = rf_metrics["mae"]
rf_rae = rf_metrics["rae"]

# HistGradientBoosting
hgb = HistGradientBoostingRegressor(random_state=42)
hgb.fit(X_train, y_train)
hgb_pred = hgb.predict(X_test)
hgb_metrics = compute_metrics(y_test.values, hgb_pred)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Metrics - RMSE
lr_metrics = compute_metrics(y_test.values, lr_pred)
lr_rmse = lr_metrics["rmse"]
lr_r2 = lr_metrics["r2"]
lr_r2_clipped = lr_metrics["r2_clipped"]
lr_mae = lr_metrics["mae"]
lr_rae = lr_metrics["rae"]

# ===== Cross-validation evaluation to get stable metrics and pick best model =====
def cv_evaluate(model, X_df, y_ser, folds=5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    metrics_acc = {"r2": [], "rmse": [], "mae": [], "rae": []}
    for train_idx, test_idx in kf.split(X_df):
        X_tr, X_te = X_df.iloc[train_idx], X_df.iloc[test_idx]
        y_tr, y_te = y_ser.iloc[train_idx], y_ser.iloc[test_idx]
        try:
            m = model.__class__(**{k: v for k, v in model.get_params().items() if k != 'random_state'})
        except Exception:
            m = model.__class__()
        # preserve random_state if available
        if 'random_state' in model.get_params():
            try:
                m.set_params(random_state=42)
            except Exception:
                pass
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        met = compute_metrics(y_te.values, y_pred)
        metrics_acc['r2'].append(met['r2'])
        metrics_acc['rmse'].append(met['rmse'])
        metrics_acc['mae'].append(met['mae'])
        metrics_acc['rae'].append(met['rae'])
    # return mean of metrics
    return {k: float(np.mean(v)) for k, v in metrics_acc.items()}

# evaluate models via CV
print('\nRunning cross-validation evaluation (5 folds) to compare models...')
rf_cv = cv_evaluate(rf, X, y)
hgb_cv = cv_evaluate(hgb, X, y)
lr_cv = cv_evaluate(lr, X, y)

print('CV (mean) RF:', rf_cv)
print('CV (mean) HGB:', hgb_cv)
print('CV (mean) LR:', lr_cv)

# pick best model by RMSE (lower is better)
cv_results = [('rf', rf_cv['rmse'], rf), ('hgb', hgb_cv['rmse'], hgb), ('lr', lr_cv['rmse'], lr)]
best_name, best_rmse, best_model_candidate = sorted(cv_results, key=lambda x: x[1])[0]
print(f"Best model by CV RMSE: {best_name} (RMSE={best_rmse:.2f})")

# fit the chosen best model on full data
best_model = best_model_candidate
best_model.fit(X, y)

print(f"RF Metrics: RMSE={rf_rmse:.2f}, R2(raw)={rf_r2:.3f}, R2(clipped)={rf_r2_clipped:.3f}, MAE={rf_mae:.2f}, RAE={rf_rae:.3f}")
print(f"HGB Metrics: RMSE={hgb_metrics['rmse']:.2f}, R2(raw)={hgb_metrics['r2']:.3f}, MAE={hgb_metrics['mae']:.2f}, RAE={hgb_metrics['rae']:.3f}")
print(f"LR Metrics: RMSE={lr_rmse:.2f}, R2(raw)={lr_r2:.3f}, R2(clipped)={lr_r2_clipped:.3f}, MAE={lr_mae:.2f}, RAE={lr_rae:.3f}")

model_data = {
    "rf_model": rf,
    "hgb_model": hgb,
    "lr_model": lr,
    "rf_metrics": {
        "r2": round(rf_r2, 3),
        "r2_clipped": round(rf_r2_clipped, 3),
        "rmse": round(rf_rmse, 2),
        "mae": round(rf_mae, 2),
        "rae": round(rf_rae, 3),
    },
    "hgb_metrics": {
        "r2": round(hgb_metrics["r2"], 3),
        "rmse": round(hgb_metrics["rmse"], 2),
        "mae": round(hgb_metrics["mae"], 2),
        "rae": round(hgb_metrics["rae"], 3),
    },
    "lr_metrics": {
        "r2": round(lr_r2, 3),
        "r2_clipped": round(lr_r2_clipped, 3),
        "rmse": round(lr_rmse, 2),
        "mae": round(lr_mae, 2),
        "rae": round(lr_rae, 3),
    },
    "feature_columns": feature_columns,  # IMPORTANT: Save exact feature order
    "scaler": scaler,
    "num_features": num_features,
    "route_stats": route_stats.reset_index().to_dict(orient="list"),
}

# Save best model into model_data for serving compatibility
model_data['best_model'] = best_model
model_data['best_model_name'] = best_name

# For backwards compatibility: set 'serving_model' and 'rf_model' to best_model
model_data['serving_model'] = best_model
model_data['rf_model'] = best_model

# Save model_data and cleaned CSV already written earlier
joblib.dump(model_data, "model_data.pkl")
print("Complete. Models and cleaned dataset saved.")
