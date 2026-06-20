import requests

TOKEN = "8642450221:AAFcR0ck6P1eu5D_KhbPs01Y-Z4DVjHcERw"
CHAT_ID = "5031994606"   # as string

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
payload = {"chat_id": CHAT_ID, "text": "Hello from HTTP request!"}

response = requests.post(url, json=payload)
if response.status_code == 200:
    print("✅ Message sent successfully!")
else:
    print("❌ Failed:", response.text)