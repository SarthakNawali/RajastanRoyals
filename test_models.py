import requests

with open('d:/python/RR/api.txt', 'r', encoding='utf-8') as f:
    key = [line.split('=')[1].strip(' "\'\n') for line in f if 'gemini_api_key' in line][0]

print("Testing with key:", key[:6] + "...")
for model in ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro']:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    try:
        r = requests.post(url, json={'contents': [{'parts': [{'text': 'Say hello'}]}]}, timeout=10)
        print(model, "status:", r.status_code)
        if r.status_code == 200:
            print("Response:", r.json()['candidates'][0]['content']['parts'][0]['text'])
            break
        else:
            print("Response error:", r.text[:150])
    except Exception as e:
        print(model, "error:", e)

