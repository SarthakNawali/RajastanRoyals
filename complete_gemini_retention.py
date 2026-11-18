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

out_json = f"{DATA_DIR}/outputs/gemini_ipl_retention.json"
existing_data = []
if os.path.exists(out_json):
    try:
        with open(out_json, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    except:
        pass

processed_teams = set(x['team_name'] for x in existing_data)
print(f"Already processed teams ({len(processed_teams)}):", processed_teams)

ctpm = pd.read_csv(f"{DATA_DIR}/competition_team_player_mapping.csv")
ipl2026 = ctpm[ctpm["comp_id"] == 3298][["player_id", "player_name", "team_name"]].drop_duplicates()
all_teams = sorted(ipl2026["team_name"].unique())

remaining_teams = [t for t in all_teams if t not in processed_teams]
print(f"Remaining teams to process ({len(remaining_teams)}):", remaining_teams)

# Try models with gentle backoff
models_to_try = ["gemini-3.5-flash-lite", "gemini-2.5-pro", "gemini-3.1-pro-preview"]
headers = {"Content-Type": "application/json"}

for team in remaining_teams:
    team_players = ipl2026[ipl2026["team_name"] == team]["player_name"].tolist()
    print(f"\nQuerying Gemini for {team} ({len(team_players)} players)...")
    
    prompt = f"""You are an official IPL (Indian Premier League) cricket analyst and franchise auction strategist.
For the IPL franchise '{team}', here is their exact squad list from the 2026 season:
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
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}
    }
    
    success = False
    for model in models_to_try:
        if success:
            break
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        for attempt in range(2):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=25)
                if resp.status_code == 200:
                    data = resp.json()
                    text_content = data['candidates'][0]['content']['parts'][0]['text']
                    parsed = json.loads(text_content)
                    print(f"  Successfully retrieved {len(parsed)} classifications for {team} using {model}")
                    existing_data.extend(parsed)
                    success = True
                    break
                else:
                    print(f"  Model {model} attempt {attempt+1} code {resp.status_code}")
                    time.sleep(3)
            except Exception as e:
                print(f"  Model {model} attempt {attempt+1} error: {e}")
                time.sleep(3)
    
    time.sleep(2)

# Save combined data
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(existing_data, f, indent=2)

df_all = pd.DataFrame(existing_data).drop_duplicates(subset=["player_name", "team_name"])
df_all.to_csv(f"{DATA_DIR}/outputs/gemini_ipl_retention.csv", index=False)

print(f"\nCompleted! Total Gemini classifications: {len(df_all)}")
print("Breakdown by status:")
print(df_all["retention_status"].value_counts())
print("\nTeams covered:", df_all["team_name"].nunique())

