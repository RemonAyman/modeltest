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
    # restore saved objects (do NOT refit anything here)
    lr_model = model_data.get("lr_model")
    rf_model = model_data.get('rf_model')
    hgb_model = model_data.get('hgb_model')
    # metrics and metadata
    rf_metrics = model_data.get("rf_metrics", {})
    lr_metrics = model_data.get("lr_metrics", {})
    best_cv_r2 = model_data.get('best_cv_r2')
    best_model_name = model_data.get('best_model_name')
    # Feature columns and scaler to be used for prediction (no fitting)
    feature_columns = model_data.get("feature_columns", [])
    scaler = model_data.get('scaler')
    num_features = model_data.get('num_features', ["passenger_count", "hour", "is_weekend"])

    # determine serving model by best_model_name or best_cv_r2
    serving_model = None
    serving_name = None
    if best_model_name:
        if best_model_name == 'lr' and lr_model is not None:
            serving_model = lr_model
            serving_name = 'lr'
        elif best_model_name == 'rf' and rf_model is not None:
            serving_model = rf_model
            serving_name = 'rf'
        elif best_model_name == 'hgb' and hgb_model is not None:
            serving_model = hgb_model
            serving_name = 'hgb'

    # fallback: choose the model with highest saved per-model CV r2 if best_model_name missing
    if serving_model is None:
        candidates = []
        if lr_metrics and 'r2' in lr_metrics:
            candidates.append(('lr', lr_metrics.get('r2')))
        if rf_metrics and 'r2' in rf_metrics:
            candidates.append(('rf', rf_metrics.get('r2')))
        hgb_metrics = model_data.get('hgb_metrics', {})
        if hgb_metrics and 'r2' in hgb_metrics:
            candidates.append(('hgb', hgb_metrics.get('r2')))
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            serving_name = candidates[0][0]
            serving_model = {'lr': lr_model, 'rf': rf_model, 'hgb': hgb_model}.get(serving_name)

    # route_stats (dict of lists) -> DataFrame for quick lookup
    route_stats_df = None
    if model_data.get('route_stats'):
        try:
            route_stats_df = pd.DataFrame(model_data.get('route_stats'))
        except Exception:
            route_stats_df = None

    # load cleaned dataset for route_hour_mean and weather_mean lookups if saved maps absent
    df_clean = None
    try:
        df_clean = pd.read_csv('cleaned_transport_dataset.csv')
        # build lookup maps only if not present in model_data
        route_hour_map = model_data.get('route_hour_map') or df_clean.set_index(['route_id', 'hour'])['route_hour_mean'].to_dict()
        weather_mean_map = model_data.get('weather_mean_map') or df_clean.groupby('weather')['weather_mean_delay'].mean().to_dict()
    except Exception:
        route_hour_map = model_data.get('route_hour_map', {}) or {}
        weather_mean_map = model_data.get('weather_mean_map', {}) or {}

    print(f"Loaded model_data.pkl: serving_name={serving_name}, best_cv_r2={best_cv_r2}")
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback for dev if model not trained yet
    serving_model = rf_model = lr_model = None
    feature_columns = []
    scaler = None
    num_features = ["passenger_count", "hour", "is_weekend"]
    route_hour_map = {}
    weather_mean_map = {}


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
        hgb_m = model_data.get('hgb_metrics', {}) if model_data else {}

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
            # Compare available model metrics
            comps = []
            if rf_m:
                comps.append(('RF', rf_m))
            if hgb_m:
                comps.append(('HGB', hgb_m))
            if lr_m:
                comps.append(('LR', lr_m))

            # Simple comparisons by RMSE and R2
            best_rmse = None
            for name, m in comps:
                rmse = m.get('rmse')
                if rmse is not None:
                    if best_rmse is None or rmse < best_rmse[0]:
                        best_rmse = (rmse, name)

            if best_rmse:
                reasons.append(f"Best RMSE: {best_rmse[1]} (RMSE={best_rmse[0]})")

            # Top features hint
            if top_features:
                reasons.append('Top features affecting delay: ' + ', '.join([t['feature'] for t in top_features[:5]]))
        except Exception:
            pass

        # Dataset correlations (numeric) from cleaned csv if available
        correlations = {}
        try:
            df_clean = pd.read_csv('cleaned_transport_dataset.csv')
            num_cols = df_clean.select_dtypes(include=[float, int]).columns.tolist()
            corr = df_clean[num_cols].corrwith(df_clean['delay_minutes']).abs().sort_values(ascending=False)
            correlations = corr.head(12).to_dict()
        except Exception:
            correlations = {}

        payload = {
            'dataset': ds_stats,
            'rf_metrics': rf_m,
            'lr_metrics': lr_m,
            'hgb_metrics': hgb_m,
            'rf_top_features': top_features,
            'lr_top_coefs': lr_coefs,
            'reasons': reasons,
            'correlations': correlations,
            'route_stats': model_data.get('route_stats', {}),
        }

        return jsonify(payload)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if not serving_model:
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

        # 3. Ensure all numeric features expected by the model exist (fill or compute)
        scaler = model_data.get("scaler")
        num_features = model_data.get("num_features", ["passenger_count", "hour", "is_weekend"])

        # compute cyclical hour features
        if "hour_sin" in num_features and "hour_sin" not in df_input.columns:
            df_input["hour_sin"] = np.sin(2 * np.pi * df_input["hour"] / 24)
        if "hour_cos" in num_features and "hour_cos" not in df_input.columns:
            df_input["hour_cos"] = np.cos(2 * np.pi * df_input["hour"] / 24)

        # route-based aggregated features from route_stats_df
        if "route_mean_delay" in num_features:
            if "route_id" in df_input.columns and route_stats_df is not None:
                rmap = route_stats_df.set_index('route_id').to_dict(orient='index')
                df_input["route_mean_delay"] = df_input["route_id"].apply(lambda r: rmap.get(r, {}).get('route_mean_delay', 0))
                df_input["route_median_delay"] = df_input["route_id"].apply(lambda r: rmap.get(r, {}).get('route_median_delay', 0))
                df_input["route_std_delay"] = df_input["route_id"].apply(lambda r: rmap.get(r, {}).get('route_std_delay', 0))
                df_input["route_count"] = df_input["route_id"].apply(lambda r: rmap.get(r, {}).get('route_count', 0))
            else:
                for c in ["route_mean_delay", "route_median_delay", "route_std_delay", "route_count"]:
                    if c in num_features and c not in df_input.columns:
                        df_input[c] = 0

        # route_hour_mean: try lookup in route_hour_map else fallback to route_mean_delay
        if "route_hour_mean" in num_features:
            def _route_hour_val(row):
                key = (row['route_id'], int(row['hour']))
                return route_hour_map.get(key, row.get('route_mean_delay', 0))

            df_input['route_hour_mean'] = df_input.apply(_route_hour_val, axis=1)

        # weather mean
        if "weather_mean_delay" in num_features:
            df_input['weather_mean_delay'] = df_input['weather'].apply(lambda w: weather_mean_map.get(w, 0))

        # dow (if missing) infer from day_type or default 0
        if 'dow' in num_features and 'dow' not in df_input.columns:
            if is_weekend_str == 'weekend':
                df_input['dow'] = 6
            else:
                df_input['dow'] = 0

        # latitude/longitude defaults
        if 'latitude' in num_features and 'latitude' not in df_input.columns:
            df_input['latitude'] = float(data.get('latitude', 0) or 0)
        if 'longitude' in num_features and 'longitude' not in df_input.columns:
            df_input['longitude'] = float(data.get('longitude', 0) or 0)

        # Now scale numeric features if scaler present
        if scaler:
            # ensure all num_features exist
            for c in num_features:
                if c not in df_input.columns:
                    df_input[c] = 0
            df_input[num_features] = scaler.transform(df_input[num_features])

        # 4. One-Hot Encoding & Alignment
        cat_features = ["route_id", "weather", "time_of_day"]
        df_encoded = pd.get_dummies(df_input[cat_features])

        df_final = pd.concat([df_encoded, df_input[num_features]], axis=1)

        feature_columns = model_data.get("feature_columns", [])
        if not feature_columns:
            return jsonify({"error": "Model feature columns missing"}), 500

        df_final = df_final.reindex(columns=feature_columns, fill_value=0)

        # Predict with serving model (best model)
        svc_raw = serving_model.predict(df_final)[0]
        svc_pred = svc_raw
        try:
            # If serving model is LR and lr was trained on log1p, apply inverse
            if 'serving_name' in globals() and serving_name == 'lr' and model_data.get('lr_target_transform') == 'log1p':
                svc_pred = np.expm1(svc_raw)
                print(f"Serving (LR) raw: {svc_raw}, after inverse np.expm1: {svc_pred}")
            else:
                print(f"Serving model ({globals().get('serving_name', 'unknown')}) raw prediction: {svc_raw}")
        except Exception as e:
            print('Error applying serving inverse transform:', e)
        svc_pred = round(max(0, float(svc_pred)), 1)

        # Also predict with RF (explicit) if available
        rf_pred = None
        try:
            if rf_model is not None:
                rf_raw = rf_model.predict(df_final)[0]
                rf_pred = round(max(0, rf_raw), 1)
        except Exception:
            rf_pred = None

        # Also predict with LR if available (for comparison)
        lr_pred = None
        if lr_model is not None:
            try:
                lr_raw = lr_model.predict(df_final)[0]
                # Log raw LR prediction and apply inverse transform if needed (no refit)
                lr_target_transform = model_data.get('lr_target_transform') if model_data else None
                print(f"LR raw prediction before inverse transform: {lr_raw}, lr_target_transform={lr_target_transform}")
                if lr_target_transform == 'log1p':
                    lr_inv = np.expm1(lr_raw)
                    print(f"LR prediction after np.expm1: {lr_inv}")
                    lr_pred = round(max(0, lr_inv), 1)
                else:
                    lr_pred = round(max(0, lr_raw), 1)
            except Exception:
                lr_pred = None

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
        # Include both generic `serving_*` and explicit `rf_*` keys because
        # the frontend expects `rf_prediction`, `rf_accuracy`, `rf_reason`.
        return jsonify(
            {
                "serving_prediction": svc_pred,
                "serving_model": model_data.get("best_model_name", "serving"),
                # provide RF-named fields for frontend compatibility (prefer explicit rf_model prediction)
                "rf_prediction": (rf_pred if rf_pred is not None else svc_pred),
                "rf_accuracy": f"RMSE: {rf_metrics.get('rmse', 'N/A')}",
                "rf_reason": rf_reason,
                # LR comparator fields
                "lr_prediction": lr_pred,
                "lr_accuracy": f"RMSE: {lr_metrics.get('rmse', 'N/A')}",
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
