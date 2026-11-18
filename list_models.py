import requests

with open('d:/python/RR/api.txt', 'r', encoding='utf-8') as f:
    key = [line.split('=')[1].strip(' "\'\n') for line in f if 'gemini_api_key' in line][0]

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
r = requests.get(url, timeout=10)
print("Status:", r.status_code)
if r.status_code == 200:
    models = [m['name'] for m in r.json().get('models', []) if 'generateContent' in m.get('supportedGenerationMethods', [])]
    print("Available generateContent models:")
    for m in models:
        print(" ", m)
else:
    print(r.text)

