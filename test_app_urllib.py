import urllib.request
import urllib.parse
import json
import http.cookiejar

base_url = "http://127.0.0.1:5000"
cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

# 1. Login
login_data = urllib.parse.urlencode({"username": "test", "password": "pw"}).encode(
    "utf-8"
)
try:
    resp = opener.open(f"{base_url}/login", data=login_data)
    print(f"Login Response URL: {resp.geturl()}")
    if "/dashboard" in resp.geturl():
        print("Login Successful")
    else:
        print("Login Failed")
except Exception as e:
    print(f"Login Error: {e}")

# 2. Predict
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
    print(f"Prediction Response: {data}")
    if "delay_minutes" in data:
        print("Prediction Test Passed")
except Exception as e:
    print(f"Prediction Error: {e}")
