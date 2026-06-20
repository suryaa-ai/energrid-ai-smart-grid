import requests

TOKEN = "8642450221:AAFcR0ck6P1eu5D_KhbPs01Y-Z4DVjHcERw"
CHAT_ID = "5031994606"

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
payload = {"chat_id": CHAT_ID, "text": "Direct test from Python"}

try:
    r = requests.post(url, json=payload, timeout=30)
    print("Status code:", r.status_code)
    print("Response:", r.text)
except Exception as e:
    print("Error:", e)