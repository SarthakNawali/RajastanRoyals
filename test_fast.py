import requests

with open('d:/python/RR/api.txt', 'r', encoding='utf-8') as f:
    key = [line.split('=')[1].strip(' "\'\n') for line in f if 'gemini_api_key' in line][0]

for m in ['gemini-2.5-flash-lite', 'gemini-flash-latest', 'gemini-3.5-flash-lite', 'gemini-2.5-pro']:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={key}"
    r = requests.post(url, json={'contents': [{'parts': [{'text': 'Say hello in 3 words'}]}]}, timeout=10)
    print(m, "status:", r.status_code)
    if r.status_code == 200:
        print("Success! Output:", r.json()['candidates'][0]['content']['parts'][0]['text'])
        break

