import requests
import json

base_url = "http://127.0.0.1:5000"

# 1. Test Login (Simulate session)
session = requests.Session()
login_data = {"username": "test", "password": "pw"}
r_login = session.post(f"{base_url}/login", data=login_data)

print(f"Login Status: {r_login.status_code}")
if r_login.url == f"{base_url}/dashboard":
    print("Login Successful: Redirected to dashboard")
else:
    print(f"Login Failed? Url: {r_login.url}")

# 2. Test Predict API
predict_data = {
    "route": "R1",
    "weather": "sunny",
    "time": "08:00",
    "day_type": "weekday",
}

r_predict = session.post(f"{base_url}/predict", json=predict_data)
print(f"Predict Status: {r_predict.status_code}")
print(f"Predict Response: {r_predict.text}")

if r_predict.status_code == 200:
    print("Prediction Test Passed")
else:
    print("Prediction Test Failed")
