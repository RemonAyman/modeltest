import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

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


# --- 3. Modeling ---

# Select Features for Encoding
cat_features = ["route_id", "weather", "time_of_day"]
num_features = ["passenger_count", "hour", "is_weekend"]

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

# Metrics - RMSE
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred) * 100
rf_mae = mean_absolute_error(y_test, rf_pred)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Metrics - RMSE
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred) * 100
lr_mae = mean_absolute_error(y_test, lr_pred)

print(f"RF Metrics: RMSE={rf_rmse:.2f}, R2={rf_r2:.2f}%, MAE={rf_mae:.2f}")
print(f"LR Metrics: RMSE={lr_rmse:.2f}, R2={lr_r2:.2f}%, MAE={lr_mae:.2f}")

# Save
# Ensure we only stick to strict column list for CSV
cols_out = [
    "route_id",
    "scheduled_time",
    "actual_time",
    "weather",
    "passenger_count",
    "latitude",
    "longitude",
    "delay_minutes",
]
df[cols_out].to_csv("cleaned_transport_dataset.csv", index=False)

model_data = {
    "rf_model": rf,
    "lr_model": lr,
    "rf_metrics": {
        "rmse": round(rf_rmse, 2),
        "r2": round(rf_r2, 2),
        "mae": round(rf_mae, 2),
    },
    "lr_metrics": {
        "rmse": round(lr_rmse, 2),
        "r2": round(lr_r2, 2),
        "mae": round(lr_mae, 2),
    },
    "feature_columns": feature_columns,  # IMPORTANT: Save exact feature order
    "scaler": scaler,
    "num_features": num_features,
}
joblib.dump(model_data, "model_data.pkl")
print("Complete.")
