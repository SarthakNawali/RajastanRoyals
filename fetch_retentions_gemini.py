import os
import requests
import json
import pandas as pd
import time

DATA_DIR = r"d:/python/RR"
api_key = None
with open(f"{DATA_DIR}/api.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if "gemini_api_key" in line:
            parts = line.split("=")
            if len(parts) >= 2:
                api_key = parts[1].strip().strip('"').strip("'")

if not api_key:
    raise ValueError("Gemini API key not found in api.txt")

ctpm = pd.read_csv(f"{DATA_DIR}/competition_team_player_mapping.csv")
ipl2026 = ctpm[ctpm["comp_id"] == 3298][["player_id", "player_name", "team_name"]].drop_duplicates()

print(f"Loaded {len(ipl2026)} IPL 2026 player records across {ipl2026['team_name'].nunique()} teams.")

teams = sorted(ipl2026["team_name"].unique())
all_retention_results = []

model_name = "gemini-3.5-flash-lite"
url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
headers = {"Content-Type": "application/json"}

for team in teams:
    team_players = ipl2026[ipl2026["team_name"] == team]["player_name"].tolist()
    print(f"\nQuerying Gemini ({model_name}) for {team} ({len(team_players)} players)...")
    
    prompt = f"""You are an official IPL (Indian Premier League) cricket analyst and franchise auction strategist.
For the IPL team '{team}', here is their exact squad list from the 2026 season:
{json.dumps(team_players)}

Based on official IPL retention rules ahead of the upcoming IPL 2027 auction cycle:
1. Identify which players are RETAINED (the core 4 to 6 marquee/franchise players held back).
2. All other players in this squad are NOT RETAINED (released into the IPL 2027 auction pool to free up purse).

Return your response strictly as a JSON array of objects without markdown formatting.
Each object must have:
- "player_name": string (exact player name matching the input list)
- "team_name": "{team}"
- "retention_status": "Retained" or "Not Retained"
- "reason": concise reason (e.g., "Franchise captain & marquee batter", "Released to auction pool to recover purse", "Overseas pace enforcer retained", "Squad rotation bowler released")

Include all {len(team_players)} players from the list.
"""

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json"
        }
    }
    
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                text_content = data['candidates'][0]['content']['parts'][0]['text']
                parsed = json.loads(text_content)
                print(f"  Successfully retrieved {len(parsed)} player classifications for {team}")
                all_retention_results.extend(parsed)
                break
            else:
                print(f"  Attempt {attempt+1} failed with status {resp.status_code}: {resp.text[:150]}")
                time.sleep(2)
        except Exception as e:
            print(f"  Attempt {attempt+1} error: {e}")
            time.sleep(2)

print(f"\nTotal Gemini classifications received: {len(all_retention_results)}")

# Save to JSON and CSV
out_json = f"{DATA_DIR}/outputs/gemini_ipl_retention.json"
out_csv = f"{DATA_DIR}/outputs/gemini_ipl_retention.csv"

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(all_retention_results, f, indent=2)

df_ret = pd.DataFrame(all_retention_results)
df_ret.to_csv(out_csv, index=False)

print(f"Saved Gemini retention database to:\n  - {out_json}\n  - {out_csv}")
print("\nRetention Breakdown:")
print(df_ret["retention_status"].value_counts())
print("\nSample Retained Players:")
print(df_ret[df_ret["retention_status"] == "Retained"][["player_name", "team_name", "reason"]].head(10).to_string(index=False))
