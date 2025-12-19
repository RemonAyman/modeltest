from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    jsonify,
    flash,
)
import joblib
import pandas as pd
import numpy as np
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "super_secret_key_for_demo_purposes"

# Database Setup
DB_NAME = "users.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT UNIQUE NOT NULL, 
                  email TEXT UNIQUE NOT NULL, 
                  password TEXT NOT NULL)"""
    )
    conn.commit()
    conn.close()


init_db()

# Load Model
try:
    model_data = joblib.load("model_data.pkl")
    rf_model = model_data["rf_model"]
    lr_model = model_data["lr_model"]
    rf_metrics = model_data["rf_metrics"]
    lr_metrics = model_data["lr_metrics"]
    # Feature columns for alignment
    model_data.get("feature_columns", [])
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback for dev if model not trained yet
    rf_model = lr_model = None


@app.route("/")
def index():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        hashed_pw = generate_password_hash(password, method="pbkdf2:sha256")

        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (username, email, hashed_pw),
            )
            conn.commit()
            conn.close()
            flash("Account created! Please login.")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username or Email already exists.")
            return redirect(url_for("signup"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        login_id = request.form["login_id"]  # Can be username or email
        password = request.form["password"]

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        # Check against username or email
        c.execute(
            "SELECT * FROM users WHERE username=? OR email=?", (login_id, login_id)
        )
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            session["user"] = user[1]  # Store username
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials.")

    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", user=session["user"])


@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if not rf_model:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json

    try:
        route = data.get("route")
        weather = data.get("weather")
        time_val = data.get("time")
        is_weekend_str = data.get("day_type")

        # 1. Feature Engineering (Match training logic)
        hour = int(time_val.split(":")[0])
        tod_str = "Night"
        if 5 <= hour < 12:
            tod_str = "Morning"
        elif 12 <= hour < 17:
            tod_str = "Afternoon"
        elif 17 <= hour < 21:
            tod_str = "Evening"

        # 2. Construct Input DataFrame
        input_data = {
            "route_id": [route],
            "weather": [weather.lower()],
            "time_of_day": [tod_str],
            "passenger_count": [int(data.get("passengers", 56))],
            "hour": [hour],
            "is_weekend": [1 if is_weekend_str == "weekend" else 0],
        }
        df_input = pd.DataFrame(input_data)

        # 3. Scaling Numerical Features
        scaler = model_data.get("scaler")
        num_features = model_data.get(
            "num_features", ["passenger_count", "hour", "is_weekend"]
        )

        if scaler:
            # Only scale what was scaled during training
            df_input[num_features] = scaler.transform(df_input[num_features])

        # 4. One-Hot Encoding & Alignment
        cat_features = ["route_id", "weather", "time_of_day"]
        df_encoded = pd.get_dummies(df_input[cat_features])

        df_final = pd.concat([df_encoded, df_input[num_features]], axis=1)

        feature_columns = model_data.get("feature_columns", [])
        if not feature_columns:
            return jsonify({"error": "Model feature columns missing"}), 500

        df_final = df_final.reindex(columns=feature_columns, fill_value=0)

        # Predict
        rf_raw = rf_model.predict(df_final)[0]
        lr_raw = lr_model.predict(df_final)[0]

        # Clip negative predictions
        rf_pred = round(max(0, rf_raw), 1)
        lr_pred = round(max(0, lr_raw), 1)

        # Return RMSE as accuracy metric
        return jsonify(
            {
                "rf_prediction": rf_pred,
                "lr_prediction": lr_pred,
                "rf_accuracy": f"RMSE: {rf_metrics.get('rmse', 'N/A')}",
                "lr_accuracy": f"RMSE: {lr_metrics.get('rmse', 'N/A')}",
            }
        )

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
