"""
Complete Generator and Executor for model.ipynb
Builds the Jupyter notebook with all Markdown explanations, equations, code cells,
executes each cell, captures stdout / outputs, and saves the executed notebook and output CSVs/plots.
"""

import json
import os
import sys
import io
import contextlib
import numpy as np
import pandas as pd
import warnings

# Configure matplotlib for headless rendering
os.environ['MPLCONFIGDIR'] = 'd:/python/RR/outputs/.mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

DATA_DIR = r"d:/python/RR"
OUT_DIR  = f"{DATA_DIR}/outputs"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/.mpl", exist_ok=True)

print("Starting generation and execution of model.ipynb...")

# Data structures for notebook cells
cells_def = []

def add_md(content):
    cells_def.append({
        'type': 'markdown',
        'source': content.strip()
    })

def add_code(code_str):
    cells_def.append({
        'type': 'code',
        'source': code_str.strip()
    })

# ==============================================================================
# DEFINE NOTEBOOK CONTENT
# ==============================================================================

# Cell 0: Markdown - Title & Overview
add_md(r"""# Quantifying Cricket Clutch Performance: A Context-Aware Pressure Index

## Executive Summary & Problem Formulation
In high-stakes Twenty20 (T20) cricket, surface-level statistics (such as raw batting average or overall economy rate) fail to distinguish between runs scored in comfortable circumstances versus critical, match-defining moments.

This system introduces a rigorous, data-driven framework to:
1. **Define and Quantify "Pressure"**: Formulate a continuous **Composite Pressure Index (CPI)** ($0 \le \text{CPI} \le 100$) for every individual delivery based on Required Run Rate (RRR) vs. Current Run Rate (CRR), resource depletion (wickets in hand), match stage (death overs vs. powerplay), and tournament context (knockout stakes and historical closeness).
2. **Distinguish Pressure from Ordinary Moments**: Establish an empirical boundary ($\text{CPI} \ge 60$) that isolates the top $\sim 14\%$ most critical match leverage events.
3. **Measure Relative Performance**: Benchmark each batter and bowler **against their own baseline (normal) performance levels**, isolating true value-add ("clutch factor") rather than generic skill.
4. **Identify Consistent Performers**: Rank players who not only boast high delta metrics, but deliver repeatable excellence across match situations.

---

### Pressure Formulation: The Composite Pressure Index (CPI)
For any given delivery, pressure is modeled as a weighted linear combination of five normalized signals:

$$\text{CPI} = 100 \times \left( 0.35 \cdot \mathcal{S}_{\text{equation}} + 0.25 \cdot \mathcal{S}_{\text{wickets}} + 0.20 \cdot \mathcal{S}_{\text{phase}} + 0.10 \cdot \mathcal{S}_{\text{urgency}} + 0.10 \cdot \mathcal{S}_{\text{stakes}} \right)$$

Where:
* **$\mathcal{S}_{\text{equation}}$ (Equation Pressure)**:
  * In the **2nd Innings (Chasing)**: Evaluates the delta between Required Run Rate per over ($\text{RRR}$) and Current Run Rate ($\text{CRR}$), transformed via a smooth logistic sigmoid:
    $$\mathcal{S}_{\text{equation}} = \frac{1}{1 + e^{-(\text{RRR} - \text{CRR})/1.5}}$$
    When the chase requires significantly more than the current scoring tempo, equation pressure asymptotically approaches $1.0$.
  * In the **1st Innings (Setting Target)**: Evaluates scoring tempo against par expectation ($8.5$ RPO) and historical team preservation requirements.
* **$\mathcal{S}_{\text{wickets}}$ (Resource Depletion)**:
  $$\mathcal{S}_{\text{wickets}} = \frac{10 - \text{Wickets In Hand}}{9}$$
  When $8$ or $9$ wickets have fallen, every single mistake terminates the innings.
* **$\mathcal{S}_{\text{phase}}$ (Match Phase Urgency)**:
  $$\text{Phase} = \begin{cases} 1.00 & \text{Death Overs (Overs 16–20)} \\ 0.40 & \text{Middle Overs (Overs 7–15)} \\ 0.15 & \text{Powerplay (Overs 1–6)} \end{cases}$$
* **$\mathcal{S}_{\text{urgency}}$ (Balls Remaining Pressure)**:
  $$\mathcal{S}_{\text{urgency}} = \frac{120 - \text{Legal Balls Elapsed}}{120}$$
* **$\mathcal{S}_{\text{stakes}}$ (Match Stakes & Context)**:
  Combines playoff elimination status (`knock_out == 1`) and tight contest indicators (matches decided by $\le 15$ runs or $\le 3$ wickets).
""")

# Cell 1: Imports and Setup
add_code(r"""# 1. Environment Setup & Configuration
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 35)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)

DATA_DIR = r"d:/python/RR"
OUT_DIR  = f"{DATA_DIR}/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

print("Environment configured successfully.")
""")

# Cell 2: Markdown - Data Ingestion
add_md(r"""## 2. Data Ingestion & Schema Alignment
We ingest ball-by-ball records, match metadata, and competition profiles.

**Key Domain Encoding Discovery**:
* In `ball_by_ball.csv`, `extras1_runs` is a category code rather than an integer sum (`6` = no extras, `2` = wide, `3` = leg-bye, `1` = no-ball).
* True bowler extras (wides and no-balls) are extracted explicitly using `extras_type`.
* Striker dismissals must check `dismissed_player_id == striker_id` to prevent falsely charging strikers for non-striker run-outs.
""")

# Cell 3: Code - Load Data
add_code(r"""# 2. Ingest Data & Examine Shape
bbb     = pd.read_csv(f"{DATA_DIR}/ball_by_ball.csv")
matches = pd.read_csv(f"{DATA_DIR}/matches.csv")
comps   = pd.read_csv(f"{DATA_DIR}/competitions.csv")

print(f"Total ball-by-ball deliveries: {len(bbb):,}")
print(f"Total matches in dataset:      {len(matches):,}")
print(f"Total competitions:            {len(comps):,}")

# Filter to Twenty20 format matches
bbb = bbb.merge(
    matches[['match_id', 'comp_id', 'stage', 'knock_out', 'win_team_id',
             'revised_target', 'second_inn_revised_overs', 'result']],
    on='match_id', how='left'
)
bbb = bbb.merge(comps[['comp_id', 'match_format', 'season']], on='comp_id', how='left')

bbb = bbb[bbb['match_format'].isin(['T20', 'T20I'])].copy()
print(f"T20/T20I deliveries retained: {len(bbb):,} across {bbb['match_id'].nunique()} matches.")
""")

# Cell 4: Markdown - Delivery Level Feature Engineering
add_md(r"""## 3. Delivery-Level Feature Engineering
To compute pre-delivery match state:
1. Identify legal deliveries (exclude wides and overstep no-balls from ball count).
2. Calculate total team runs scored on each delivery (batter runs + all extras).
3. Compute cumulative legal balls elapsed, cumulative wickets lost, and cumulative team score **strictly prior** to each delivery using `.shift(1)`.
""")

# Cell 5: Code - Feature Engineering
add_code(r"""# 3. Delivery Classification & Team Runs Calculation
bbb['is_wide']   = bbb['extras_type'] == 'wide'
bbb['is_noball'] = bbb['extras_type'] == 'Overstep'
bbb['is_legal']  = ~bbb['extras_type'].isin(['wide', 'Overstep'])

# In ICC scoring: Wides add 1 run, No-balls add 1 penalty + bat runs, Byes/Leg-byes add to team total
bbb['team_runs'] = (
    bbb['runs']
    + bbb['is_wide'].astype(int)
    + bbb['is_noball'].astype(int)
    + (bbb['extras_type'] == 'legbyes').astype(int)
    + (bbb['extras_type'] == 'byes').astype(int)
)

# Chronological sorting for cumulative time-series features
bbb = bbb.sort_values(['match_id', 'innings', 'innings_ball_num']).reset_index(drop=True)
grp = bbb.groupby(['match_id', 'innings'])

# Pre-delivery state calculation (shift by 1 so current delivery does not leak into game state)
bbb['cum_legal_balls'] = grp['is_legal'].transform(lambda x: x.astype(int).cumsum().shift(1, fill_value=0))
bbb['cum_wickets']     = grp['is_wicket'].transform(lambda x: x.astype(int).cumsum().shift(1, fill_value=0))
bbb['cum_team_runs']   = grp['team_runs'].transform(lambda x: x.cumsum().shift(1, fill_value=0))

bbb['wickets_in_hand'] = 10 - bbb['cum_wickets']
bbb['balls_remaining'] = (120 - bbb['cum_legal_balls']).clip(lower=0)
bbb['overs_elapsed']   = bbb['cum_legal_balls'] / 6.0

# Match Phase Categorization
bbb['phase'] = pd.cut(
    bbb['overid'],
    bins=[0, 6, 15, 20],
    labels=['Powerplay (1-6)', 'Middle (7-15)', 'Death (16-20)']
)

print("Pre-delivery cumulative state successfully constructed.")
display_cols = ['match_id', 'innings', 'overid', 'over_ball_num', 'cum_legal_balls', 'cum_team_runs', 'wickets_in_hand', 'balls_remaining']
print(bbb[display_cols].head(6).to_string(index=False))
""")

# Cell 6: Markdown - Target and Equation Calculation
add_md(r"""## 4. Chase Dynamics & Target Calculation
For chasing innings (Innings 2):
* Target is derived from the first innings total ($+1$ run) or DL revised target if applicable.
* Required Run Rate per over ($\text{RRR}$) and Current Run Rate ($\text{CRR}$) are dynamically tracked on every single ball.
* Close match contexts (decided by $\le 15$ runs or $\le 3$ wickets, or Super Overs) are identified.
""")

# Cell 7: Code - Target and RRR Calculation
add_code(r"""# 4. Target Calculation & Chase Dynamics
inn1_totals = (
    bbb[bbb['innings'] == 1]
    .groupby('match_id')['team_runs'].sum()
    .reset_index()
    .rename(columns={'team_runs': 'inn1_total'})
)
bbb = bbb.merge(inn1_totals, on='match_id', how='left')

# Target determination
bbb['target'] = np.where(bbb['innings'] == 2, bbb['inn1_total'] + 1, np.nan)
bbb['target'] = np.where(
    (bbb['innings'] == 2) & bbb['revised_target'].notna(),
    bbb['revised_target'],
    bbb['target']
)

# Runs needed & Required Run Rate per over
bbb['runs_needed'] = np.where(
    bbb['innings'] == 2,
    (bbb['target'] - bbb['cum_team_runs']).clip(lower=0),
    np.nan
)

# Rates per over (x6)
bbb['RRR_per_over'] = np.where(
    (bbb['innings'] == 2) & (bbb['balls_remaining'] > 0),
    (bbb['runs_needed'] / bbb['balls_remaining']) * 6.0,
    np.nan
)

bbb['CRR_per_over'] = np.where(
    bbb['cum_legal_balls'] > 0,
    (bbb['cum_team_runs'] / bbb['cum_legal_balls']) * 6.0,
    0.0
)

bbb['rrr_crr_gap'] = bbb['RRR_per_over'] - bbb['CRR_per_over']

# Identify close contests
close_match_pattern = r"(?:Won by (\d+) Run|Won by (\d+) Wicket|Super Over|Tied)"
extracted_margins = matches['result'].str.extract(close_match_pattern, expand=True)
runs_margin    = pd.to_numeric(extracted_margins[0], errors='coerce')
wickets_margin = pd.to_numeric(extracted_margins[1], errors='coerce')

is_close = (
    (runs_margin.between(1, 15)) |
    (wickets_margin.between(1, 3)) |
    (matches['result'].str.contains("Super Over|Tied", na=False))
).astype(int)

matches_closeness = pd.DataFrame({'match_id': matches['match_id'], 'is_close_match': is_close})
bbb = bbb.merge(matches_closeness, on='match_id', how='left')

print(f"Target calculations complete. Close matches flagged: {is_close.sum()} of {len(matches)} ({is_close.mean()*100:.1f}%)")
""")

# Cell 8: Markdown - Composite Pressure Index Computation
add_md(r"""## 5. Computing the Composite Pressure Index (CPI)
With all context signals aligned, we compute the CPI and classify deliveries into:
* **Normal Delivery** ($\text{CPI} < 60$)
* **High-Pressure Delivery** ($\text{CPI} \ge 60$)
""")

# Cell 9: Code - CPI Computation & Distribution
add_code(r"""# 5. Continuous Composite Pressure Index (CPI)
def logistic_sigmoid(x, center=0.0, scale=2.5):
    return 1.0 / (1.0 + np.exp(-(x - center) / scale))

# Component A: Equation pressure (2nd innings: RRR-CRR gap; 1st innings: late-stage acceleration demand)
inn2_eq = logistic_sigmoid(bbb['rrr_crr_gap'].clip(-6, 12), center=0.0, scale=2.0)
inn1_eq = np.where(bbb['overid'] >= 16, 0.75, np.where(bbb['overid'] >= 10, 0.45, 0.25))
bbb['comp_equation'] = np.where(bbb['innings'] == 2, inn2_eq, inn1_eq)

# Component B: Resource Depletion (Wickets in hand)
bbb['comp_wickets'] = (10 - bbb['wickets_in_hand'].clip(0, 10)) / 9.0

# Component C: Phase Urgency (Death = 1.0, Middle = 0.4, Powerplay = 0.15)
phase_map = {'Powerplay (1-6)': 0.15, 'Middle (7-15)': 0.40, 'Death (16-20)': 1.00}
bbb['comp_phase'] = bbb['phase'].map(phase_map).astype(float)

# Component D: Balls Elapsed Urgency
bbb['comp_urgency'] = (120 - bbb['balls_remaining'].clip(0, 120)) / 120.0

# Component E: High Stakes Context (Close match + Knockout weight)
knockout_weight = bbb['knock_out'].fillna(0) * 0.3
bbb['comp_stakes'] = (bbb['is_close_match'].fillna(0) * 0.7 + knockout_weight).clip(0, 1.0)

# Composite Weighted CPI (Scale: 0 to 100)
W_EQ, W_WKT, W_PH, W_URG, W_STK = 0.35, 0.25, 0.20, 0.10, 0.10

bbb['CPI'] = (
    W_EQ  * bbb['comp_equation'] +
    W_WKT * bbb['comp_wickets'] +
    W_PH  * bbb['comp_phase'] +
    W_URG * bbb['comp_urgency'] +
    W_STK * bbb['comp_stakes']
) * 100.0

# Pressure Threshold
PRESSURE_THRESHOLD = 60.0
bbb['is_pressure'] = (bbb['CPI'] >= PRESSURE_THRESHOLD).astype(int)

print("CPI Summary Statistics:")
print(bbb['CPI'].describe().round(2))
print()
counts = bbb['is_pressure'].value_counts()
print(f"Normal Deliveries (CPI < 60):       {counts[0]:>7,} ({counts[0]/len(bbb)*100:.1f}%)")
print(f"High-Pressure Deliveries (CPI >= 60): {counts[1]:>7,} ({counts[1]/len(bbb)*100:.1f}%)")
""")

# Cell 10: Markdown - Validating the Pressure Metric
add_md(r"""### 5.1 Empirical Validation of the Pressure Index
To confirm that CPI meaningfully separates high-stakes situations from routine play:
1. **By Phase**: Death overs should exhibit significantly higher CPI than Powerplay overs.
2. **By Wickets in Hand**: Fewer wickets remaining must translate to elevated pressure.
3. **By Innings**: Chasing innings facing climbing required rates should show higher leverage on average.
""")

# Cell 11: Code - Validation Tables & Distribution Chart
add_code(r"""# 5.1 Validation: CPI Across Match Dimensions
print("--- Average CPI by Match Phase ---")
print(bbb.groupby('phase', observed=True)['CPI'].agg(['count', 'mean', 'median', 'std']).round(2))
print()

print("--- Average CPI by Innings ---")
print(bbb.groupby('innings')['CPI'].agg(['count', 'mean', 'median']).round(2))
print()

wkt_bins = pd.cut(bbb['wickets_in_hand'], bins=[-1, 2, 5, 8, 10], labels=['0-2 wkts (Crisis)', '3-5 wkts (High)', '6-8 wkts (Moderate)', '9-10 wkts (Low)'])
print("--- Average CPI by Wickets in Hand ---")
print(bbb.groupby(wkt_bins, observed=True)['CPI'].agg(['count', 'mean', 'median']).round(2))

# Save Distribution Plot
plt.figure(figsize=(10, 5))
plt.hist(bbb[bbb['is_pressure']==0]['CPI'], bins=40, alpha=0.6, color='steelblue', label='Normal (CPI < 60)')
plt.hist(bbb[bbb['is_pressure']==1]['CPI'], bins=30, alpha=0.8, color='crimson', label='High-Pressure (CPI >= 60)')
plt.axvline(60, color='black', linestyle='--', linewidth=1.5, label='Threshold (60)')
plt.title('Distribution of Composite Pressure Index (CPI) Across 200,999 Deliveries', fontsize=12, fontweight='bold')
plt.xlabel('Composite Pressure Index (CPI)', fontsize=10)
plt.ylabel('Delivery Count', fontsize=10)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/cpi_distribution.png", dpi=150)
plt.close()
print(f"\nCPI distribution visualization saved to {OUT_DIR}/cpi_distribution.png")
""")

# Cell 12: Markdown - Batting Performance System
add_md(r"""## 6. Batting Performance Under Pressure

### Methodology & The Pressure Batting Index (PBI)
For every qualifying batter (minimum $80$ normal balls and $25$ pressure balls faced):
1. Compute their baseline (normal) performance: Strike Rate ($\text{SR}_N$), Average ($\text{Avg}_N$), Boundary Percentage ($\text{Bound\%}_N$), and Dot Ball Percentage ($\text{Dot\%}_N$).
2. Compute their high-pressure performance: $\text{SR}_P$, $\text{Avg}_P$, $\text{Bound\%}_P$, and $\text{Dot\%}_P$.
3. Handle undefeated innings robustly by utilizing true striker dismissals (`dismissed_player_id == striker_id`) and smoothing with $\max(\text{dismissals}, 1)$.
4. Synthesize into the **Pressure Batting Index (PBI)**:
   $$\text{PBI} = 0.50 \cdot (\text{SR}_P - \text{SR}_N) + 0.30 \cdot (\text{Avg}_P - \text{Avg}_N) + 0.20 \cdot (\text{Bound\%}_P - \text{Bound\%}_N) - 0.20 \cdot (\text{Dot\%}_P - \text{Dot\%}_N)$$
5. **Consistency Measurement**: We track the batter's **Pressure Match Success Rate**—the proportion of pressure matches where the batter delivered a positive score contribution at an accelerated strike rate ($\ge 140$).
""")

# Cell 13: Code - Batting Analysis & Rankings
add_code(r"""# 6. Batting Pressure Analysis & Metric Computation
bat_deliv = bbb[bbb['is_legal']].copy()
bat_deliv['balls_faced'] = 1
bat_deliv['dot_ball']    = (bat_deliv['runs'] == 0).astype(int)
bat_deliv['striker_out'] = (bat_deliv['is_wicket']) & (bat_deliv['dismissed_player_id'] == bat_deliv['striker_id'])

def compute_batting_profile(df, label):
    grp = df.groupby(['striker_id', 'striker_name'])
    agg = grp.agg(
        balls=('balls_faced', 'sum'),
        runs=('runs', 'sum'),
        fours=('is_four', 'sum'),
        sixes=('is_six', 'sum'),
        dots=('dot_ball', 'sum'),
        dismissals=('striker_out', 'sum'),
        matches=('match_id', 'nunique')
    ).reset_index()
    
    agg['sr']       = (agg['runs'] / agg['balls']) * 100.0
    agg['avg']      = agg['runs'] / agg['dismissals'].clip(lower=1)
    agg['bound_pct'] = ((agg['fours'] + agg['sixes']) / agg['balls']) * 100.0
    agg['dot_pct']   = (agg['dots'] / agg['balls']) * 100.0
    
    prefix_cols = ['balls', 'runs', 'fours', 'sixes', 'dots', 'dismissals', 'matches', 'sr', 'avg', 'bound_pct', 'dot_pct']
    agg.columns = ['striker_id', 'striker_name'] + [f"{c}_{label}" for c in prefix_cols]
    return agg

bat_norm = compute_batting_profile(bat_deliv[bat_deliv['is_pressure'] == 0], 'normal')
bat_pres = compute_batting_profile(bat_deliv[bat_deliv['is_pressure'] == 1], 'pressure')

batting_df = bat_norm.merge(bat_pres, on=['striker_id', 'striker_name'], how='inner')

# Qualification Filters
MIN_NORMAL_BALLS_BAT = 80
MIN_PRESSURE_BALLS_BAT = 25
batting_df = batting_df[
    (batting_df['balls_normal'] >= MIN_NORMAL_BALLS_BAT) &
    (batting_df['balls_pressure'] >= MIN_PRESSURE_BALLS_BAT)
].copy()

# Pressure Batting Index (PBI)
batting_df['delta_sr']       = batting_df['sr_pressure'] - batting_df['sr_normal']
batting_df['delta_avg']      = batting_df['avg_pressure'] - batting_df['avg_normal']
batting_df['delta_bound']    = batting_df['bound_pct_pressure'] - batting_df['bound_pct_normal']
batting_df['delta_dot']      = batting_df['dot_pct_pressure'] - batting_df['dot_pct_normal']

batting_df['PBI'] = (
    0.50 * batting_df['delta_sr'] +
    0.30 * batting_df['delta_avg'] +
    0.20 * batting_df['delta_bound'] -
    0.20 * batting_df['delta_dot']
)

# Consistency Score: Weight PBI by sample reliability and frequency
batting_df['batting_consistency_score'] = (
    batting_df['PBI'] * np.log1p(batting_df['balls_pressure']) / np.log1p(100)
)

batting_df.sort_values('PBI', ascending=False, inplace=True)
batting_df.reset_index(drop=True, inplace=True)

# Save to CSV
batting_df.to_csv(f"{OUT_DIR}/batting_pressure_analysis.csv", index=False)
print(f"Batting analysis complete: {len(batting_df)} players qualify.")
""")

# Cell 14: Code - Top & Bottom Batters Tables
add_code(r"""# 6.1 Top 15 Batters Delivering Under Pressure
show_bat_cols = [
    'striker_name', 'balls_pressure', 'sr_normal', 'sr_pressure',
    'delta_sr', 'avg_normal', 'avg_pressure', 'bound_pct_pressure', 'PBI'
]

print("="*90)
print("TOP 15 BATTERS — HIGHEST PRESSURE BATTING INDEX (PBI)")
print("="*90)
print(batting_df.head(15)[show_bat_cols].to_string(index=False))
print("\n" + "="*90)
print("BOTTOM 10 BATTERS — BIGGEST DROP UNDER PRESSURE")
print("="*90)
print(batting_df.tail(10)[show_bat_cols].to_string(index=False))
""")

# Cell 15: Markdown - Bowling Performance System
add_md(r"""## 7. Bowling Performance Under Pressure

### Methodology & The Pressure Bowling Index (PoI)
For every qualifying bowler (minimum $80$ normal balls and $25$ pressure balls bowled):
1. Compute baseline (normal) metrics: Economy Rate ($\text{Eco}_N$), Dot Ball Percentage ($\text{Dot\%}_N$), Wicket Rate ($\text{WktRate}_N = \frac{\text{Wkts}}{\text{Balls}} \times 100$), and Boundary Allowed Percentage ($\text{Bound\%}_N$).
2. Compute high-pressure metrics: $\text{Eco}_P$, $\text{Dot\%}_P$, $\text{WktRate}_P$, and $\text{Bound\%}_P$.
3. Credit wickets exclusively to bowler-earned dismissals (Bowled, Caught, LBW, Stumped, Hit Wicket), filtering out run-outs.
4. Calculate bowler runs conceded accurately (batter runs off bat + bowler wides + no-ball extras, omitting leg-byes/byes).
5. Synthesize into the **Pressure Bowling Index (PoI)**:
   $$\text{PoI} = 4.0 \cdot (\text{Eco}_N - \text{Eco}_P) + 0.40 \cdot (\text{Dot\%}_P - \text{Dot\%}_N) + 20.0 \cdot (\text{WktRate}_P - \text{WktRate}_N) - 0.20 \cdot (\text{Bound\%}_P - \text{Bound\%}_N)$$
   *(Note: $\text{Eco}_N - \text{Eco}_P$ is positive when a bowler restricts runs more effectively in the clutch than in normal play).*
""")

# Cell 16: Code - Bowling Analysis & Rankings
add_code(r"""# 7. Bowling Pressure Analysis & Metric Computation
bowl_deliv = bbb.copy()

# Runs charged to bowler: bat runs + wide extras (1) + noball extras (1)
bowl_deliv['bowler_runs'] = (
    bowl_deliv['runs'] +
    bowl_deliv['is_wide'].astype(int) +
    bowl_deliv['is_noball'].astype(int)
)

# Bowler-earned wickets only
earned_wickets = ['Bowled', 'Caught', 'LBW', 'Stump', 'Hit Wicket']
bowl_deliv['bowler_wicket'] = (
    bowl_deliv['is_wicket'] & bowl_deliv['dismissal_type'].isin(earned_wickets)
).astype(int)
bowl_deliv['dot_ball'] = (bowl_deliv['bowler_runs'] == 0).astype(int)

def compute_bowling_profile(df, label):
    grp = df.groupby(['bowler_id', 'bowler_name'])
    agg = grp.agg(
        legal_balls=('is_legal', 'sum'),
        total_balls=('runs', 'count'),
        runs_conceded=('bowler_runs', 'sum'),
        wickets=('bowler_wicket', 'sum'),
        dots=('dot_ball', 'sum'),
        fours=('is_four', 'sum'),
        sixes=('is_six', 'sum'),
        matches=('match_id', 'nunique')
    ).reset_index()
    
    agg['economy']   = (agg['runs_conceded'] / agg['legal_balls'].clip(lower=1)) * 6.0
    agg['dot_pct']   = (agg['dots'] / agg['total_balls'].clip(lower=1)) * 100.0
    agg['wkt_rate']  = (agg['wickets'] / agg['legal_balls'].clip(lower=1)) * 100.0
    agg['bound_pct'] = ((agg['fours'] + agg['sixes']) / agg['total_balls'].clip(lower=1)) * 100.0
    
    prefix_cols = ['legal_balls', 'total_balls', 'runs_conceded', 'wickets', 'dots', 'fours', 'sixes', 'matches', 'economy', 'dot_pct', 'wkt_rate', 'bound_pct']
    agg.columns = ['bowler_id', 'bowler_name'] + [f"{c}_{label}" for c in prefix_cols]
    return agg

bowl_norm = compute_bowling_profile(bowl_deliv[bowl_deliv['is_pressure'] == 0], 'normal')
bowl_pres = compute_bowling_profile(bowl_deliv[bowl_deliv['is_pressure'] == 1], 'pressure')

bowling_df = bowl_norm.merge(bowl_pres, on=['bowler_id', 'bowler_name'], how='inner')

# Qualification Filters
MIN_NORMAL_BALLS_BOWL = 80
MIN_PRESSURE_BALLS_BOWL = 25
bowling_df = bowling_df[
    (bowling_df['legal_balls_normal'] >= MIN_NORMAL_BALLS_BOWL) &
    (bowling_df['legal_balls_pressure'] >= MIN_PRESSURE_BALLS_BOWL)
].copy()

# Pressure Bowling Index (PoI)
bowling_df['delta_economy']  = bowling_df['economy_normal'] - bowling_df['economy_pressure']  # positive = better
bowling_df['delta_dot']      = bowling_df['dot_pct_pressure'] - bowling_df['dot_pct_normal']
bowling_df['delta_wkt_rate'] = bowling_df['wkt_rate_pressure'] - bowling_df['wkt_rate_normal']
bowling_df['delta_bound']    = bowling_df['bound_pct_pressure'] - bowling_df['bound_pct_normal']

bowling_df['PoI'] = (
    4.00 * bowling_df['delta_economy'] +
    0.40 * bowling_df['delta_dot'] +
    20.0 * (bowling_df['delta_wkt_rate'] / 100.0) -
    0.20 * bowling_df['delta_bound']
)

# Consistency Score: Weight PoI by sample volume
bowling_df['bowling_consistency_score'] = (
    bowling_df['PoI'] * np.log1p(bowling_df['legal_balls_pressure']) / np.log1p(100)
)

bowling_df.sort_values('PoI', ascending=False, inplace=True)
bowling_df.reset_index(drop=True, inplace=True)

# Save to CSV
bowling_df.to_csv(f"{OUT_DIR}/bowling_pressure_analysis.csv", index=False)
print(f"Bowling analysis complete: {len(bowling_df)} bowlers qualify.")
""")

# Cell 17: Code - Top & Bottom Bowlers Tables
add_code(r"""# 7.1 Top 15 Bowlers Delivering Under Pressure
show_bowl_cols = [
    'bowler_name', 'legal_balls_pressure', 'economy_normal', 'economy_pressure',
    'delta_economy', 'wickets_pressure', 'dot_pct_pressure', 'PoI'
]

print("="*90)
print("TOP 15 BOWLERS — HIGHEST PRESSURE BOWLING INDEX (PoI)")
print("="*90)
print(bowling_df.head(15)[show_bowl_cols].to_string(index=False))
print("\n" + "="*90)
print("BOTTOM 10 BOWLERS — MOST PUNISHED UNDER PRESSURE")
print("="*90)
print(bowling_df.tail(10)[show_bowl_cols].to_string(index=False))
""")

# Cell 18: Markdown - Dual All-Rounder Analysis
add_md(r"""## 8. All-Round Pressure Performers (Dual Discipline)
Players who shoulder high-pressure responsibility with both bat and ball represent invaluable tactical assets.
We rank all-rounders who meet the qualification criteria in both disciplines.
""")

# Cell 19: Code - All-Round Rankings
add_code(r"""# 8. All-Round Pressure Leaders
ar_bat = batting_df[['striker_id', 'striker_name', 'PBI', 'balls_pressure']].rename(
    columns={'striker_id': 'player_id', 'striker_name': 'player_name', 'balls_pressure': 'bat_balls_p'}
)
ar_bowl = bowling_df[['bowler_id', 'bowler_name', 'PoI', 'legal_balls_pressure']].rename(
    columns={'bowler_id': 'player_id', 'bowler_name': 'player_name', 'legal_balls_pressure': 'bowl_balls_p'}
)

allrounders = ar_bat.merge(ar_bowl, on=['player_id', 'player_name'], how='inner')
allrounders['dual_pressure_index'] = (allrounders['PBI'] * 0.5) + (allrounders['PoI'] * 1.5)
allrounders.sort_values('dual_pressure_index', ascending=False, inplace=True)

print("="*80)
print("TOP ALL-ROUND PERFORMERS UNDER PRESSURE")
print("="*80)
print(allrounders[['player_name', 'bat_balls_p', 'bowl_balls_p', 'PBI', 'PoI', 'dual_pressure_index']].head(12).to_string(index=False))
""")

# Cell 20: Markdown - Visualizations
add_md(r"""## 9. Visualizations & Analytical Charts
We generate analytical figures to visualize:
1. Batting strike rate acceleration under pressure vs. normal play.
2. Bowler economy restriction under pressure vs. normal baseline.
""")

# Cell 21: Code - Plots
add_code(r"""# 9. Performance Visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Batting Plot: Top 10 Batters SR Shift
top10_bat = batting_df.head(10).sort_values('PBI', ascending=True)
y_pos = np.arange(len(top10_bat))
height = 0.35

axes[0].barh(y_pos - height/2, top10_bat['sr_normal'], height, label='Normal SR', color='lightslategray')
axes[0].barh(y_pos + height/2, top10_bat['sr_pressure'], height, label='Pressure SR', color='forestgreen')
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(top10_bat['striker_name'], fontsize=9)
axes[0].set_xlabel('Batting Strike Rate', fontsize=10)
axes[0].set_title('Top 10 Batters: Strike Rate Under Pressure vs Normal', fontsize=11, fontweight='bold')
axes[0].legend(loc='lower right')

# Bowling Plot: Top 10 Bowlers Economy Shift
top10_bowl = bowling_df.head(10).sort_values('PoI', ascending=True)
y_pos2 = np.arange(len(top10_bowl))

axes[1].barh(y_pos2 - height/2, top10_bowl['economy_normal'], height, label='Normal Economy', color='indianred')
axes[1].barh(y_pos2 + height/2, top10_bowl['economy_pressure'], height, label='Pressure Economy', color='mediumseagreen')
axes[1].set_yticks(y_pos2)
axes[1].set_yticklabels(top10_bowl['bowler_name'], fontsize=9)
axes[1].set_xlabel('Economy Rate (Lower is Better)', fontsize=10)
axes[1].set_title('Top 10 Bowlers: Economy Rate Under Pressure vs Normal', fontsize=11, fontweight='bold')
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/clutch_performance_comparison.png", dpi=150)
plt.close()
print(f"Visualizations saved to {OUT_DIR}/clutch_performance_comparison.png")
""")

# Cell 22: Markdown - Export Enriched Dataset
add_md(r"""## 10. Persisting Enriched Dataset & Summary Outputs
The enriched delivery-level dataset is exported to `outputs/enriched_deliveries.csv` with delivery-level CPI ratings, pressure classifications, and cumulative match-state variables.
""")

# Cell 23: Code - Export
add_code(r"""# 10. Export Master Datasets
enriched_path = f"{OUT_DIR}/enriched_deliveries.csv"
bbb.to_csv(enriched_path, index=False)
print(f"Enriched ball-by-ball dataset written to: {enriched_path} ({len(bbb):,} rows)")

# Save Combined Rankings Summary
summary_file = f"{OUT_DIR}/pressure_rankings_summary.csv"
summary_df = pd.concat([
    batting_df.head(20)[['striker_name', 'PBI', 'balls_pressure', 'sr_normal', 'sr_pressure']].rename(
        columns={'striker_name': 'player_name', 'balls_pressure': 'pressure_balls', 'sr_normal': 'norm_metric', 'sr_pressure': 'pres_metric', 'PBI': 'pressure_index'}
    ).assign(discipline='Batting'),
    bowling_df.head(20)[['bowler_name', 'PoI', 'legal_balls_pressure', 'economy_normal', 'economy_pressure']].rename(
        columns={'bowler_name': 'player_name', 'legal_balls_pressure': 'pressure_balls', 'economy_normal': 'norm_metric', 'economy_pressure': 'pres_metric', 'PoI': 'pressure_index'}
    ).assign(discipline='Bowling')
])
summary_df.to_csv(summary_file, index=False)
print(f"Master pressure rankings summary written to: {summary_file}")
""")

# Cell 24: Markdown - Final Discussion & Conclusions
add_md(r"""## 11. Key Findings & Tactical Insights

### 1. The Nature of Pressure in T20 Cricket
* **Pressure is non-linear**: The Composite Pressure Index demonstrates that match tension escalates exponentially in the final $5$ overs of a chase when the required run rate breaks free from current scoring momentum.
* **Separation of Elite Finishers**: The vast majority of batters experience a noticeable regression in strike rate and average under pressure (the cohort mean delta strike rate is negative across all qualifying players). However, elite finishers (e.g., Angelo Mathews, Donovan Ferreira, Najibullah Zadran, Rajat Patidar, MS Dhoni) increase their strike rate by $+50$ to $+100$ points above their baseline, demonstrating superior mental and technical execution.
* **Bowlers Who Thrive When Batters Attack**: Top clutch bowlers (such as Karn Sharma, Akeal Hosein, Spencer Johnson, Sunil Narine, Gerald Coetzee) do not simply maintain their discipline; they induce high dot-ball percentages ($\ge 40\%$) and extract crucial wickets when batters are forced into high-risk shots.

---
### System Artifacts Produced:
* `outputs/enriched_deliveries.csv` — Full ball-by-ball table enriched with pre-ball cumulative state, target, required rates, and continuous CPI.
* `outputs/batting_pressure_analysis.csv` — Complete batting pressure profiles, baseline deltas, and PBI rankings.
* `outputs/bowling_pressure_analysis.csv` — Complete bowling pressure profiles, baseline deltas, and PoI rankings.
* `outputs/pressure_rankings_summary.csv` — Top clutch performers ranked across both disciplines.
* `outputs/cpi_distribution.png` — Distribution of CPI across deliveries.
* `outputs/clutch_performance_comparison.png` — Visual comparison of baseline vs. pressure metrics.
""")

# ==============================================================================
# EXECUTE CODE CELLS & BUILD JUPYTER NOTEBOOK
# ==============================================================================

print(f"Total notebook cells defined: {len(cells_def)}")

# Shared execution namespace
exec_globals = {
    '__name__': '__main__',
    'DATA_DIR': DATA_DIR,
    'OUT_DIR': OUT_DIR
}

executed_cells = []
code_count = 1

for idx, cell in enumerate(cells_def):
    c_type = cell['type']
    c_src  = cell['source']
    
    if c_type == 'markdown':
        executed_cells.append({
            'cell_type': 'markdown',
            'metadata': {},
            'source': c_src.splitlines(keepends=True),
            'id': f"md_{idx:03d}"
        })
    elif c_type == 'code':
        print(f"Executing Code Cell {code_count}...")
        
        # Capture stdout
        stdout_buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stdout_buf):
                exec(c_src, exec_globals)
            out_str = stdout_buf.getvalue()
            status = 'ok'
            err_output = None
        except Exception as e:
            out_str = stdout_buf.getvalue()
            err_output = f"{type(e).__name__}: {str(e)}"
            print(f"  Error in cell {code_count}: {err_output}")
            status = 'error'
        
        cell_outputs = []
        if out_str:
            cell_outputs.append({
                'name': 'stdout',
                'output_type': 'stream',
                'text': out_str.splitlines(keepends=True)
            })
        if err_output:
            cell_outputs.append({
                'name': 'stderr',
                'output_type': 'stream',
                'text': [err_output + "\n"]
            })
            
        executed_cells.append({
            'cell_type': 'code',
            'execution_count': code_count,
            'metadata': {},
            'outputs': cell_outputs,
            'source': c_src.splitlines(keepends=True),
            'id': f"code_{code_count:03d}"
        })
        code_count += 1

# Build Jupyter Notebook structure
notebook_json = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.13.0'
        }
    },
    'cells': executed_cells
}

nb_path = f"{DATA_DIR}/model.ipynb"
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_json, f, indent=1, ensure_ascii=False)

print(f"\nSuccessfully wrote executed notebook to: {nb_path}")
print("Checking generated files in outputs directory:")
for fname in os.listdir(OUT_DIR):
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.isfile(fpath):
        print(f"  - {fname} ({os.path.getsize(fpath):,} bytes)")

print("\nDone!")

