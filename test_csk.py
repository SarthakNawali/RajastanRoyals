import requests
import json
import sys

with open('d:/python/RR/api.txt', 'r', encoding='utf-8') as f:
    key = [line.split('=')[1].strip(' "\'\n') for line in f if 'gemini_api_key' in line][0]

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={key}"
headers = {"Content-Type": "application/json"}
payload = {
    "contents": [{"parts": [{"text": "Name 3 retained players for Chennai Super Kings in IPL. Return JSON list: [{\"player_name\": \"...\", \"team_name\": \"Chennai Super Kings\", \"retention_status\": \"Retained\", \"reason\": \"...\"}]"}]}],
    "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}
}

print("Sending request...", flush=True)
try:
    r = requests.post(url, headers=headers, json=payload, timeout=15)
    print("Status:", r.status_code, flush=True)
    print("Result:", r.text[:300], flush=True)
except Exception as e:
    print("Error:", e, flush=True)

