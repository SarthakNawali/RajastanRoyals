# 🏏 Cricket Clutch Analytics & Decision Hub

A Streamlit web app that measures how T20 players (batters, bowlers and all-rounders) perform under **high-pressure match situations** compared with **normal situations**. It is built for team selection, auction planning, scouting and role assignment.

---

## 1. What the app does

Every ball in the dataset has a **Composite Pressure Index (CPI)** from 0 to 100. Balls with `CPI >= threshold` (default 60) are treated as **High Pressure**; all others are **Normal**. The app then compares each player's numbers across the two groups.

### Tabs

| Tab | Purpose |
|---|---|
| 🏏 **Batters Decision Hub** | Scatter plot of Normal SR vs Pressure SR, ranked table with PBI, top death-over finishers, CSV export |
| 🎯 **Bowlers Lockdown Hub** | Scatter plot of Normal Economy vs Pressure Economy, ranked table with PoI, top death-over bowlers, CSV export |
| ⚖️ **All-Rounders Matrix** | Players who qualify as both batter and bowler, ranked by a dual impact score |
| ⚔️ **Head-to-Head Duel** | Side-by-side comparison of any two players with a tactical verdict |
| 🔍 **Player Deep-Dive Scout Card** | One player's season-by-season SR, phase splits and 10 highest-leverage moments |
| 👤 **Player Pressure Lookup** | Pick one or more players and see only their Normal vs High-Pressure batting and bowling tables, with CSV export |

### Metrics

**Composite Pressure Index (CPI)** (pre-computed in the data, 0–100):

```
CPI = 100 × (0.35·Equation + 0.25·Wickets + 0.20·Phase + 0.10·Urgency + 0.10·Stakes)
```

**Pressure Batting Index (PBI)**

```
PBI = 0.50·ΔSR + 0.30·ΔAvg + 0.20·ΔBoundary% − 0.20·ΔDot%
```

Tiers: 👑 Elite Clutch (≥45), 🔥 High Impact (≥20), ⚖️ Balanced (≥0), ⚠️ Choke Risk (<0)

**Pressure Bowling Index (PoI)**

```
PoI = 4.0·(Eco_Normal − Eco_Pressure) + 0.40·ΔDot% + 20·(ΔWktRate/100) − 0.20·ΔBoundary%
```

Tiers: ❄️ Ice in Veins (≥15), 🎯 Lockdown Specialist (≥7), ⚖️ Steady (≥0), ⚠️ High Pressure Leaker (<0)

**Dual Impact Score (all-rounders)** = `0.5 × PBI + 1.5 × PoI`

(Δ = Pressure value − Normal value, except for economy, where a positive Δ means tighter bowling under pressure.)

---

## 2. Requirements

- Python 3.9 or newer
- Packages:

```
streamlit
pandas
numpy
altair
matplotlib
```

> `matplotlib` is needed because the tables use pandas `background_gradient` styling.

---

## 3. Data files needed

The app reads from the folder set in `DATA_DIR` (default `d:/python/RR`):

```
d:/python/RR/
├── ball_by_ball.csv
├── matches.csv
├── competitions.csv
├── teams.csv
└── outputs/
    └── enriched_deliveries.csv    ← main file the app uses
```

- **`outputs/enriched_deliveries.csv`** is the file the app loads first. It must already contain the computed `CPI` column and the helper flags the app uses (see below).
- If it is missing, the app falls back to merging `ball_by_ball.csv + matches.csv + competitions.csv`. That raw merge does **not** contain `CPI`, `is_legal`, `phase` and similar columns, so the app will fail with a `KeyError`. In practice, generate `enriched_deliveries.csv` first.

### Columns the app expects in the enriched data

`match_id`, `comp_id`, `season`, `stage`, `knock_out`, `is_close_match`, `innings`, `overid`, `over_ball_num`, `phase`, `CPI`, `batting_team_id`, `bowling_team_id`, `striker_id`, `striker_name`, `batting_type`, `batting_order`, `bowler_id`, `bowler_name`, `pace_or_spin`, `bowling_type`, `runs`, `extras_type`, `is_legal`, `is_four`, `is_six`, `is_wicket`, `dismissal_type`, `dismissed_player_id`

`teams.csv` needs `team_id`, `team_name`. `matches.csv` needs `home_team_id`, `home_team`, `away_team_id`, `away_team`. `competitions.csv` needs `comp_id`, `comp_name`.

---

## 4. How to run

**Step 1 – Put the code in a file**, for example `app.py`.

**Step 2 – (Recommended) create a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**Step 3 – Install dependencies**

```bash
pip install streamlit pandas numpy altair matplotlib
```

**Step 4 – Set your data folder**

Open `app.py` and edit this line so it points to your own data folder:

```python
DATA_DIR = r"d:/python/RR"
```

**Step 5 – Start the app**

```bash
streamlit run app.py
```

Streamlit opens the dashboard in your browser, usually at **http://localhost:8501**. If it doesn't open, copy that address into your browser.

To stop the app, press `Ctrl + C` in the terminal.

---

## 5. Using the dashboard

1. **Pick a scenario in the sidebar.** Use the Quick Scenario presets, or set seasons, tournament level, match stage, innings and "Tight Contests Only" yourself.
2. **Tune the model.** Change the **CPI Pressure Threshold** (40–80) and the minimum pressure / normal ball counts. Higher minimums mean fewer but more reliable players.
3. **Filter by role.** Choose batting order, batting hand (RHB/LHB) and bowling style (Pace/Spin).
4. **Explore the tabs.** Hover over chart points for details; use the 📥 buttons to download CSVs.
5. **Use the Lookup tab** for a quick answer on specific players. Tick *"Ignore sidebar filters"* to use the full history.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError` | Check `DATA_DIR` and that all CSV files are in it |
| `KeyError: 'CPI'` (or `is_legal`, `phase`) | `enriched_deliveries.csv` is missing or incomplete; generate it first |
| `ImportError: matplotlib` | `pip install matplotlib` |
| "No batters/bowlers meet the criteria" | Lower the **Min Pressure/Normal Balls** sliders or loosen the filters |
| Old results after changing the CSV | Press `C` in the app and choose *Clear cache*, or restart Streamlit |
| Season presets show nothing | The presets look for seasons named `'2026'`, `'2025'`, `'2024/25'`; make sure your `season` column uses those labels |

---

## 7. Notes & known limitations

- `DATA_DIR` is a hard-coded Windows path; change it to match your machine.
- The "Latest Season (2026)" preset and the Duel tab's 2026 form rows are hard-coded to the season label `'2026'`.
- Some chart axis titles show "CPI < 60 / ≥ 60" as fixed text even when you move the threshold slider. The actual calculations do use the slider value.
- The Head-to-Head Duel and Scout Card tabs use the **full history**, not the sidebar filters, but they do respect the CPI threshold.
- Small samples give noisy results; keep the minimum-ball sliders reasonably high before drawing conclusions.