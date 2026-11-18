import os
import requests
import json

api_key = None
with open('d:/python/RR/api.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if 'gemini_api_key' in line:
            # extract key between quotes
            parts = line.split('=')
            if len(parts) >= 2:
                api_key = parts[1].strip().strip('"').strip("'")

print("API Key extracted:", api_key[:6] + "..." + api_key[-4:] if api_key else "None")

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
headers = {"Content-Type": "application/json"}
payload = {
    "contents": [{
        "parts": [{"text": "Hello, answer in 5 words."}]
    }]
}

try:
    resp = requests.post(url, headers=headers, json=payload, timeout=10)
    print("Status code:", resp.status_code)
    print("Response:", resp.json())
except Exception as e:
    print("Request exception:", e)

