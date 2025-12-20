import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from typing import Dict

# ===============================
# 1. Load Dataset
# ===============================
file_path = "dirty_transport_dataset.csv"
df = pd.read_csv(file_path)

print("Original Data Info:")
print(df.info())

# ===============================
# 2. Data Cleaning
# ===============================


# ---- Route ID ----
def clean_route_id(route):
    route = str(route).strip().upper()
    if route in ["3", "R03"]:
        return "R3"
    if route in ["ROUTE-4", "ROUTE 4"]:
        return "R4"
    if route == "1":
        return "R1"
    if route.startswith("R") and len(route) == 2:
        return route
    return None


df["route_id"] = df["route_id"].apply(clean_route_id)
df = df.dropna(subset=["route_id"])


# ---- Time Parsing ----
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


def combine_date_time(row):
    if pd.isna(row["actual_time_obj"]) or pd.isna(row["scheduled_dt"]):
        return pd.NaT

    sch = row["scheduled_dt"]
    act = datetime.combine(sch.date(), row["actual_time_obj"])
    diff = (act - sch).total_seconds() / 60

    if diff < -720:
        act += timedelta(days=1)

    return act


df["actual_dt"] = df.apply(combine_date_time, axis=1)
df = df.dropna(subset=["actual_dt"])


# ---- Weather ----
def clean_weather(w):
    if pd.isna(w):
        return "unknown"
    w = str(w).lower()
    if "sun" in w or "clear" in w:
        return "sunny"
    if "cloud" in w or "clody" in w:
        return "cloudy"
    if "rain" in w:
        return "rainy"
    return "unknown"


df["weather"] = df["weather"].apply(clean_weather)
df["weather"] = df["weather"].replace(
    "unknown", df[df["weather"] != "unknown"]["weather"].mode()[0]
)

# ---- Passenger Count ----
df["passenger_count"] = pd.to_numeric(df["passenger_count"], errors="coerce")
median_passengers = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 200)][
    "passenger_count"
].median()

df["passenger_count"] = df["passenger_count"].apply(
    lambda x: median_passengers if pd.isna(x) or x < 1 or x > 200 else int(x)
)

# ---- GPS (if present) ----
if "latitude" in df.columns and "longitude" in df.columns:
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df[(df["latitude"].abs() <= 90) & (df["longitude"].abs() <= 180)]
    df = df.dropna(subset=["latitude", "longitude"])

# ===============================
# 3. Feature Engineering
# ===============================

df["delay_minutes"] = (df["actual_dt"] - df["scheduled_dt"]).dt.total_seconds() / 60

# Remove extreme outliers
df = df[(df["delay_minutes"] >= 0) & (df["delay_minutes"] <= 240)]


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

# Cyclical encoding for hour (helps tree & linear models handle wrap-around)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# ===============================
# 4. Encoding & Scaling
# ===============================

cat_features = ["route_id", "weather", "time_of_day"]
# include gps and cyclic hour features in numeric list if present
num_features = ["passenger_count", "hour", "is_weekend", "hour_sin", "hour_cos"]
if "latitude" in df.columns and "longitude" in df.columns:
    num_features += ["latitude", "longitude"]

# For tree models we can keep all dummy columns
df_cat = pd.get_dummies(df[cat_features], drop_first=False)

scaler = StandardScaler()
df_num = pd.DataFrame(
    scaler.fit_transform(df[num_features]), columns=num_features, index=df.index
)

# X for tree models
X = pd.concat([df_cat, df_num], axis=1)
y = df["delay_minutes"]

# Prepare X for linear regression (avoid dummy trap)
X_lr = pd.get_dummies(df[cat_features], drop_first=True)
X_lr = pd.concat([X_lr, df_num], axis=1)

feature_columns = X.columns.tolist()

# ===============================
# 5. Train / Test (index-based split so all X variants align)
# ===============================

train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)

X_train, X_test = X.loc[train_idx], X.loc[test_idx]
X_lr_train, X_lr_test = X_lr.loc[train_idx], X_lr.loc[test_idx]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]

# ---- Random Forest (with RandomizedSearchCV tuning) ----
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5, None],
}

rs = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_dist,
    n_iter=10,
    cv=3,
    n_jobs=-1,
    random_state=42,
    verbose=0,
)

rs.fit(X_train, y_train)
rf_best = rs.best_estimator_
rf_pred = rf_best.predict(X_test)


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    denom = np.sum(np.abs(y_true - np.mean(y_true)))
    rae = np.sum(np.abs(y_true - y_pred)) / denom if denom != 0 else float("inf")
    r2 = r2_score(y_true, y_pred)
    r2_clipped = float(np.clip(r2, 1e-6, 1.0))
    return {
        "r2": float(r2),
        "r2_clipped": r2_clipped,
        "rmse": float(rmse),
        "mae": float(mae),
        "rae": float(rae),
    }


rf_metrics = compute_metrics(y_test.values, rf_pred)
rf_rmse = rf_metrics["rmse"]
rf_r2 = rf_metrics["r2"]
rf_r2_clipped = rf_metrics["r2_clipped"]
rf_mae = rf_metrics["mae"]
rf_rae = rf_metrics["rae"]


# ---- Random Forest on log1p(target) experiment ----
rf_log = RandomForestRegressor(random_state=42, n_estimators=rf_best.n_estimators, max_depth=rf_best.max_depth, min_samples_leaf=rf_best.min_samples_leaf, max_features=rf_best.max_features)
rf_log.fit(X_train, np.log1p(y_train))
rf_log_pred = np.expm1(rf_log.predict(X_test))
rf_log_metrics = compute_metrics(y_test.values, rf_log_pred)

# ---- HistGradientBoosting (stronger tree model) ----
hgb = HistGradientBoostingRegressor(random_state=42)
hgb.fit(X_train, y_train)
hgb_pred = hgb.predict(X_test)
hgb_metrics = compute_metrics(y_test.values, hgb_pred)

# ---- Linear Regression (use X_lr inputs) ----
lr = LinearRegression()
lr.fit(X_lr_train, y_train)
lr_pred = lr.predict(X_lr_test)
lr_metrics = compute_metrics(y_test.values, lr_pred)
lr_rmse = lr_metrics["rmse"]
lr_r2 = lr_metrics["r2"]
lr_r2_clipped = lr_metrics["r2_clipped"]
lr_mae = lr_metrics["mae"]
lr_rae = lr_metrics["rae"]

print("\nRandom Forest (tuned):")
print(f"R2 (raw) = {rf_r2:.3f}")
print(f"R2 (clipped 0-1) = {rf_r2_clipped:.3f}")
print(f"RMSE = {rf_rmse:.2f}")
print(f"MAE = {rf_mae:.2f}")
print(f"RAE = {rf_rae:.3f}")

print("\nRandom Forest (tuned) on log1p target:")
print(f"RMSE = {rf_log_metrics['rmse']:.2f}")
print(f"MAE = {rf_log_metrics['mae']:.2f}")
print(f"RAE = {rf_log_metrics['rae']:.3f}")

print("\nHistGradientBoosting:")
print(f"R2 (raw) = {hgb_metrics['r2']:.3f}")
print(f"RMSE = {hgb_metrics['rmse']:.2f}")
print(f"MAE = {hgb_metrics['mae']:.2f}")
print(f"RAE = {hgb_metrics['rae']:.3f}")

print("\nLinear Regression:")
print(f"R2 (raw) = {lr_r2:.3f}")
print(f"R2 (clipped 0-1) = {lr_r2_clipped:.3f}")
print(f"RMSE = {lr_rmse:.2f}")
print(f"MAE = {lr_mae:.2f}")
print(f"RAE = {lr_rae:.3f}")

# ===============================
# 6. Save Model
# ===============================

model_data = {
    "rf_model": rf_best,
    "rf_log_model": rf_log,
    "hgb_model": hgb,
    "lr_model": lr,
    "rf_metrics": {
        "r2": round(rf_r2, 3),
        "r2_clipped": round(rf_r2_clipped, 3),
        "rmse": round(rf_rmse, 2),
        "mae": round(rf_mae, 2),
        "rae": round(rf_rae, 3),
        "best_params": rs.best_params_,
    },
    "rf_log_metrics": {
        "rmse": round(rf_log_metrics["rmse"], 2),
        "mae": round(rf_log_metrics["mae"], 2),
        "rae": round(rf_log_metrics["rae"], 3),
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
    "feature_columns": feature_columns,
    "scaler": scaler,
    "num_features": num_features,
}

joblib.dump(model_data, "model_data_no_gps.pkl")
print("\nModel saved successfully.")
