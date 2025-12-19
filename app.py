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


@app.route('/charts-data')
def charts_data():
    # Returns JSON used by frontend charts: dataset distribution, model metrics,
    # top features and brief reasons for model differences.
    try:
        # Dataset summary (if cleaned CSV exists)
        ds_stats = {}
        try:
            df = pd.read_csv('cleaned_transport_dataset.csv')
            total = len(df)
            mean_delay = float(df['delay_minutes'].mean()) if 'delay_minutes' in df else None
            median_delay = float(df['delay_minutes'].median()) if 'delay_minutes' in df else None
            buckets = {
                '0-5': int(((df['delay_minutes'] >= 0) & (df['delay_minutes'] < 5)).sum()),
                '5-15': int(((df['delay_minutes'] >= 5) & (df['delay_minutes'] < 15)).sum()),
                '15-30': int(((df['delay_minutes'] >= 15) & (df['delay_minutes'] < 30)).sum()),
                '30+': int((df['delay_minutes'] >= 30).sum()),
            }
            buckets_pct = {k: round(v / total * 100, 1) if total > 0 else 0 for k, v in buckets.items()}
            ds_stats = {
                'total_samples': int(total),
                'mean_delay': mean_delay,
                'median_delay': median_delay,
                'buckets': buckets,
                'buckets_pct': buckets_pct,
                'pct_over_15': round((df['delay_minutes'] >= 15).sum() / total * 100, 1) if total > 0 else 0,
            }
        except Exception:
            ds_stats = {'error': 'cleaned dataset not found or unreadable'}

        # Model metrics
        rf_m = model_data.get('rf_metrics', {}) if model_data else {}
        lr_m = model_data.get('lr_metrics', {}) if model_data else {}

        # Feature importance and coefficients
        top_features = []
        lr_coefs = []
        try:
            cols = model_data.get('feature_columns', []) if model_data else []
            if rf_model is not None and hasattr(rf_model, 'feature_importances_'):
                fi = list(rf_model.feature_importances_)
                pairs = sorted(list(zip(cols, fi)), key=lambda x: x[1], reverse=True)[:8]
                top_features = [{'feature': p[0], 'importance': round(float(p[1]), 4)} for p in pairs]

            if lr_model is not None and hasattr(lr_model, 'coef_'):
                coefs = list(lr_model.coef_)
                pairs = sorted(list(zip(cols, coefs)), key=lambda x: abs(x[1]), reverse=True)[:8]
                lr_coefs = [{'feature': p[0], 'coef': round(float(p[1]), 4)} for p in pairs]
        except Exception:
            top_features = []
            lr_coefs = []

        # Short textual reasons (automatic): based on RMSE/R2 and feature importance
        reasons = []
        try:
            if rf_m and lr_m:
                rf_rmse = rf_m.get('rmse')
                lr_rmse = lr_m.get('rmse')
                rf_r2 = rf_m.get('r2')
                lr_r2 = lr_m.get('r2')

                if rf_rmse and lr_rmse:
                    if rf_rmse < lr_rmse:
                        reasons.append('Random Forest has lower RMSE — better at capturing non-linear patterns and outliers.')
                    else:
                        reasons.append('Linear Regression has lower RMSE — data may be mostly linear.')

                if rf_r2 and lr_r2:
                    if rf_r2 > lr_r2:
                        reasons.append('Random Forest higher R2% — explains more variance across features.')
                    else:
                        reasons.append('Linear Regression higher R2% — simpler linear relationships dominate.')

                if top_features:
                    reasons.append('Top features affecting delay: ' + ', '.join([t['feature'] for t in top_features[:5]]))
        except Exception:
            pass

        payload = {
            'dataset': ds_stats,
            'rf_metrics': rf_m,
            'lr_metrics': lr_m,
            'rf_top_features': top_features,
            'lr_top_coefs': lr_coefs,
            'reasons': reasons,
        }

        return jsonify(payload)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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

        # Build simple per-prediction explanations
        rf_reason = 'RF explanation not available.'
        lr_reason = 'LR explanation not available.'

        try:
            cols = model_data.get('feature_columns', [])
            # For LR: use coefficients * feature values as contribution proxy
            if lr_model is not None and hasattr(lr_model, 'coef_'):
                coefs = np.array(lr_model.coef_)
                # ensure df_final columns align with cols
                vals = df_final[cols].iloc[0].values.astype(float)
                contrib = coefs * vals
                # get top contributors by absolute impact
                idxs = np.argsort(np.abs(contrib))[::-1][:5]
                parts = []
                for i in idxs:
                    parts.append(f"{cols[i]}: {contrib[i]:+.2f}")
                lr_reason = f"Linear model contributions (coef*value): {', '.join(parts)}"

            # For RF: show top feature importances and sample values
            if rf_model is not None and hasattr(rf_model, 'feature_importances_'):
                fi = np.array(rf_model.feature_importances_)
                idxs = np.argsort(fi)[::-1][:5]
                parts = []
                for i in idxs:
                    feat = cols[i]
                    importance = fi[i]
                    val = df_final.get(feat, pd.Series([0])).iloc[0] if feat in df_final.columns else 0
                    parts.append(f"{feat} (imp={importance:.3f}) val={val}")
                rf_reason = f"RF top features for this sample: {', '.join(parts)}"
        except Exception as e:
            print('Explanation generation error:', e)

        # Return predictions, accuracies and per-model reasons
        return jsonify(
            {
                "rf_prediction": rf_pred,
                "lr_prediction": lr_pred,
                "rf_accuracy": f"RMSE: {rf_metrics.get('rmse', 'N/A')}",
                "lr_accuracy": f"RMSE: {lr_metrics.get('rmse', 'N/A')}",
                "rf_reason": rf_reason,
                "lr_reason": lr_reason,
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
