import urllib.request
import urllib.parse
import json
import http.cookiejar
import random

base_url = "http://127.0.0.1:5000"
cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

user_suffix = random.randint(1000, 9999)
username = f"user_{user_suffix}"
email = f"user_{user_suffix}@test.com"
password = "password123"

print(f"Testing with User: {username}")

# 1. Signup
signup_data = urllib.parse.urlencode(
    {"username": username, "email": email, "password": password}
).encode("utf-8")

try:
    resp = opener.open(f"{base_url}/signup", data=signup_data)
    print("Signup Request Sent")
    # Should redirect or show Login page
    if "/login" in resp.geturl():
        print("Signup Successful (Redirected to Login)")
    else:
        print(f"Signup Result URL: {resp.geturl()}")
except Exception as e:
    print(f"Signup Failed: {e}")

# 2. Login
login_data = urllib.parse.urlencode(
    {"login_id": username, "password": password}  # Test with username
).encode("utf-8")

try:
    resp = opener.open(f"{base_url}/login", data=login_data)
    if "/dashboard" in resp.geturl():
        print("Login Successful")
    else:
        print("Login Failed")
except Exception as e:
    print(f"Login Error: {e}")

# 3. Predict
predict_data = json.dumps(
    {"route": "R1", "weather": "sunny", "time": "08:00", "day_type": "weekday"}
).encode("utf-8")

req = urllib.request.Request(
    f"{base_url}/predict",
    data=predict_data,
    headers={"Content-Type": "application/json"},
)
try:
    resp = opener.open(req)
    data = json.load(resp)
    print("Prediction Response:", data)
    if "rf_prediction" in data and "lr_prediction" in data:
        print("Dual Prediction Verified Success")
    else:
        print("Dual Prediction Failed keys")
except Exception as e:
    print(f"Prediction Error: {e}")
