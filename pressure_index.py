"""
Cricket Pressure Performance Analysis
======================================
Measures how batters and bowlers perform in high-pressure situations
compared to their usual (non-pressure) performance.

Pressure Definition (Composite Pressure Index - CPI):
- Based on: Required Run Rate vs Current Run Rate (2nd innings),
  Wickets in Hand, Balls Remaining, Innings Phase, Match Closeness
- Works symmetrically for batting and bowling analysis
"""

import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

DATA_DIR = "d:/python/RR"

print("Loading data...")
bbb     = pd.read_csv(f"{DATA_DIR}/ball_by_ball.csv")
matches = pd.read_csv(f"{DATA_DIR}/matches.csv")
comps   = pd.read_csv(f"{DATA_DIR}/competitions.csv")

print(f"  Ball-by-ball rows : {len(bbb):,}")
print(f"  Matches           : {len(matches):,}")
print(f"  Competitions      : {len(comps):,}")

# ─────────────────────────────────────────────
# 2. ENRICH WITH MATCH / COMPETITION CONTEXT
# ─────────────────────────────────────────────

bbb = bbb.merge(
    matches[['match_id','comp_id','stage','knock_out','win_team_id',
             'revised_target','second_inn_revised_overs']],
    on='match_id', how='left'
)
bbb = bbb.merge(
    comps[['comp_id','match_format','season']],
    on='comp_id', how='left'
)

# Keep only T20 / T20I (all data is T20 format, but be explicit)
bbb = bbb[bbb['match_format'].isin(['T20','T20I'])].copy()
print(f"\nAfter T20 filter: {len(bbb):,} deliveries across {bbb['match_id'].nunique()} matches")

# ─────────────────────────────────────────────
# 3. DELIVERY-LEVEL FEATURE ENGINEERING
# ─────────────────────────────────────────────

bbb['is_wide']   = bbb['extras_type'] == 'wide'
bbb['is_noball'] = bbb['extras_type'] == 'Overstep'
bbb['is_legal']  = ~bbb['extras_type'].isin(['wide', 'Overstep'])

# Total team runs per delivery (batting runs + all extras)
bbb['legbye_runs'] = (bbb['extras_type'] == 'legbyes').astype(int)
bbb['bye_runs']    = (bbb['extras_type'] == 'byes').astype(int)
bbb['team_runs']   = (bbb['runs']
                      + bbb['is_wide'].astype(int)
                      + bbb['is_noball'].astype(int)
                      + bbb['legbye_runs']
                      + bbb['bye_runs'])

# Sort for cumulative operations
bbb = bbb.sort_values(['match_id', 'innings', 'innings_ball_num']).reset_index(drop=True)

grp = bbb.groupby(['match_id', 'innings'])

# Cumulative stats BEFORE the current ball (shift by 1)
bbb['cum_legal_balls'] = grp['is_legal'].transform(lambda x: x.astype(int).cumsum().shift(1, fill_value=0))
bbb['cum_wickets']     = grp['is_wicket'].transform(lambda x: x.astype(int).cumsum().shift(1, fill_value=0))
bbb['cum_team_runs']   = grp['team_runs'].transform(lambda x: x.cumsum().shift(1, fill_value=0))

# Derived features
bbb['wickets_in_hand'] = 10 - bbb['cum_wickets']
bbb['balls_remaining'] = 120 - bbb['cum_legal_balls']
bbb['overs_elapsed']   = bbb['cum_legal_balls'] / 6

# Innings phase
bbb['phase'] = pd.cut(
    bbb['overid'],
    bins=[0, 6, 15, 20],
    labels=['powerplay', 'middle', 'death']
)

# ─────────────────────────────────────────────
# 4. COMPUTE INNINGS TARGET (for both innings)
# ─────────────────────────────────────────────

# First innings total per match
inn1_totals = (
    bbb[bbb['innings'] == 1]
    .groupby('match_id')['team_runs']
    .sum()
    .reset_index()
    .rename(columns={'team_runs': 'inn1_total'})
)
bbb = bbb.merge(inn1_totals, on='match_id', how='left')

# Target for 2nd innings = inn1_total + 1
bbb['target'] = np.where(
    bbb['innings'] == 2,
    bbb['inn1_total'] + 1,
    np.nan
)

# DL revised target where available
bbb['target'] = np.where(
    (bbb['innings'] == 2) & bbb['revised_target'].notna(),
    bbb['revised_target'],
    bbb['target']
)

# Runs needed at this point
bbb['runs_needed'] = np.where(
    bbb['innings'] == 2,
    (bbb['target'] - bbb['cum_team_runs']).clip(lower=0),
    np.nan
)

# Required Run Rate (balls basis: RRR per ball)
bbb['RRR_per_ball'] = np.where(
    (bbb['innings'] == 2) & (bbb['balls_remaining'] > 0),
    bbb['runs_needed'] / bbb['balls_remaining'],
    np.nan
)

# Current Run Rate (runs per ball scored so far)
bbb['CRR_per_ball'] = np.where(
    bbb['cum_legal_balls'] > 0,
    bbb['cum_team_runs'] / bbb['cum_legal_balls'],
    0.0
)

# RRR - CRR gap (positive = batting team under pressure)
bbb['rrr_crr_gap'] = bbb['RRR_per_ball'] - bbb['CRR_per_ball']

# ─────────────────────────────────────────────
# 5. MATCH CLOSENESS (for 1st innings context)
# ─────────────────────────────────────────────
# We use final margin proxy: matches that ended close get a flag
# "Close" = won by 1-10 runs, or by 1-3 wickets, or tied/super over
close_margins = matches['result'].str.extract(
    r'(?:Won by (\d+) Run|Won by (\d+) Wicket|No result|Super Over|Tied)',
    expand=True
)
matches['runs_margin']    = pd.to_numeric(close_margins[0], errors='coerce')
matches['wickets_margin'] = pd.to_numeric(close_margins[1], errors='coerce')
matches['is_close_match'] = (
    (matches['runs_margin'].between(1, 15, inclusive='both')) |
    (matches['wickets_margin'].between(1, 3, inclusive='both')) |
    (matches['result'].str.contains('Super Over|Tied|No result', na=False))
).astype(int)

bbb = bbb.merge(
    matches[['match_id', 'is_close_match']],
    on='match_id', how='left'
)

# ─────────────────────────────────────────────
# 6. COMPOSITE PRESSURE INDEX (CPI)  0–100 scale
# ─────────────────────────────────────────────
#
# Component                  | Applies to     | Weight
# ----------------------------------------------------------
# A. RRR-CRR gap (2nd inn)   | 2nd inn only   | 35%
# B. Low wickets in hand     | both innings   | 25%
# C. Death overs (16-20)     | both innings   | 20%
# D. Few balls remaining     | both innings   | 10%
# E. Close match             | both innings   | 10%
#
# Each component is normalised 0→1 before weighting.

def sigmoid(x, center=0, scale=1.0):
    return 1 / (1 + np.exp(-(x - center) / scale))


# --- A: RRR-CRR gap (2nd innings) ---
# RRR>CRR means batting team needs to accelerate = pressure
# Clip to avoid extreme outliers; normalise via sigmoid
bbb['comp_A'] = np.where(
    bbb['innings'] == 2,
    sigmoid(bbb['rrr_crr_gap'].clip(-3, 3), center=0, scale=0.5),
    0.5          # neutral for 1st innings
)

# --- B: Low wickets in hand ---
# 10 wickets = no pressure (0), 1 wicket = max pressure (1)
bbb['comp_B'] = (10 - bbb['wickets_in_hand'].clip(0, 10)) / 9

# --- C: Death overs phase ---
bbb['comp_C'] = (bbb['phase'] == 'death').astype(float)

# --- D: Few balls remaining ---
# 120 balls = start (no urgency), 0 balls = maximum urgency
bbb['comp_D'] = (120 - bbb['balls_remaining'].clip(0, 120)) / 120

# --- E: Close match ---
bbb['comp_E'] = bbb['is_close_match'].fillna(0).astype(float)

# --- Composite ---
W_A, W_B, W_C, W_D, W_E = 0.35, 0.25, 0.20, 0.10, 0.10
bbb['CPI'] = (
    W_A * bbb['comp_A']
    + W_B * bbb['comp_B']
    + W_C * bbb['comp_C']
    + W_D * bbb['comp_D']
    + W_E * bbb['comp_E']
) * 100  # scale to 0–100

# Pressure threshold: CPI > 60 = HIGH PRESSURE delivery
PRESSURE_THRESHOLD = 60
bbb['is_pressure'] = (bbb['CPI'] >= PRESSURE_THRESHOLD).astype(int)

print(f"\nPressure delivery breakdown:")
vc = bbb['is_pressure'].value_counts()
print(f"  Normal   : {vc.get(0,0):>7,} deliveries")
print(f"  Pressure : {vc.get(1,0):>7,} deliveries ({vc.get(1,0)/len(bbb)*100:.1f}%)")

print(f"\nCPI distribution:")
print(bbb['CPI'].describe().round(2))

# ─────────────────────────────────────────────
# 7. BATTING PERFORMANCE METRICS
# ─────────────────────────────────────────────
# Only legal deliveries where the batter was at strike
batting = bbb[bbb['is_legal']].copy()

# Balls faced per striker per delivery = 1
batting['balls_faced'] = 1
batting['dot_ball'] = (batting['runs'] == 0).astype(int)

def batting_metrics(df, label):
    g = df.groupby(['striker_id', 'striker_name'])
    agg = g.agg(
        balls=('balls_faced', 'sum'),
        runs_scored=('runs', 'sum'),
        fours=('is_four', 'sum'),
        sixes=('is_six', 'sum'),
        dots=('dot_ball', 'sum'),
        dismissals=('is_wicket', 'sum'),
    ).reset_index()
    agg['strike_rate'] = (agg['runs_scored'] / agg['balls']) * 100
    agg['avg']         = np.where(agg['dismissals'] > 0,
                                  agg['runs_scored'] / agg['dismissals'], np.nan)
    agg['dot_pct']     = agg['dots'] / agg['balls'] * 100
    agg['boundary_pct']= (agg['fours'] + agg['sixes']) / agg['balls'] * 100
    # Batting impact = composite batting score
    agg['bat_impact']  = (agg['strike_rate'] * 0.5 +
                          agg['avg'].fillna(0) * 0.3 +
                          agg['boundary_pct'] * 1.0)
    agg.columns = [f"{c}_{label}" if c not in ['striker_id','striker_name'] else c
                   for c in agg.columns]
    return agg

bat_normal   = batting_metrics(batting[batting['is_pressure']==0], 'normal')
bat_pressure = batting_metrics(batting[batting['is_pressure']==1], 'pressure')

bat_combined = bat_normal.merge(bat_pressure, on=['striker_id','striker_name'], how='inner')

# Minimum thresholds for meaningful stats
MIN_NORMAL_BALLS   = 100
MIN_PRESSURE_BALLS = 30

bat_combined = bat_combined[
    (bat_combined['balls_normal']   >= MIN_NORMAL_BALLS) &
    (bat_combined['balls_pressure'] >= MIN_PRESSURE_BALLS)
].copy()

# Pressure Batting Index (PBI) — positive = performs BETTER under pressure
bat_combined['PBI_sr']  = bat_combined['strike_rate_pressure'] - bat_combined['strike_rate_normal']
bat_combined['PBI_avg'] = bat_combined['avg_pressure'].fillna(0) - bat_combined['avg_normal'].fillna(0)
bat_combined['PBI']     = (bat_combined['PBI_sr'] * 0.6 + bat_combined['PBI_avg'] * 0.4)

bat_combined.sort_values('PBI', ascending=False, inplace=True)

print(f"\nBatting analysis: {len(bat_combined)} players qualify (≥{MIN_NORMAL_BALLS} normal, ≥{MIN_PRESSURE_BALLS} pressure balls)")

# ─────────────────────────────────────────────
# 8. BOWLING PERFORMANCE METRICS
# ─────────────────────────────────────────────
# For bowling: legal deliveries only; wickets exclude run-outs (not bowler's credit)
bowling = bbb[bbb['is_legal']].copy()

bowler_wicket_types = ['Bowled','Caught','LBW','Stump','Hit Wicket']
bowling['bowler_wicket'] = (
    bowling['is_wicket'] &
    bowling['dismissal_type'].isin(bowler_wicket_types)
).astype(int)

bowling['dot_ball']     = (bowling['runs'] == 0).astype(int)
bowling['runs_conceded'] = bowling['runs']

def bowling_metrics(df, label):
    g = df.groupby(['bowler_id', 'bowler_name'])
    agg = g.agg(
        balls=('runs_conceded', 'count'),
        runs_conceded=('runs_conceded', 'sum'),
        wickets=('bowler_wicket', 'sum'),
        dots=('dot_ball', 'sum'),
        fours_allowed=('is_four', 'sum'),
        sixes_allowed=('is_six', 'sum'),
    ).reset_index()
    agg['economy']     = (agg['runs_conceded'] / agg['balls']) * 6
    agg['avg']         = np.where(agg['wickets'] > 0,
                                  agg['runs_conceded'] / agg['wickets'], np.nan)
    agg['dot_pct']     = agg['dots'] / agg['balls'] * 100
    agg['sr']          = np.where(agg['wickets'] > 0,
                                  agg['balls'] / agg['wickets'], np.nan)
    # Lower economy + more wickets = better bowling impact
    agg['bowl_impact'] = agg['dot_pct'] * 0.5 - agg['economy'] * 5 + agg['wickets']
    agg.columns = [f"{c}_{label}" if c not in ['bowler_id','bowler_name'] else c
                   for c in agg.columns]
    return agg

bowl_normal   = bowling_metrics(bowling[bowling['is_pressure']==0], 'normal')
bowl_pressure = bowling_metrics(bowling[bowling['is_pressure']==1], 'pressure')

bowl_combined = bowl_normal.merge(bowl_pressure, on=['bowler_id','bowler_name'], how='inner')

MIN_NORMAL_BOWL_BALLS   = 100
MIN_PRESSURE_BOWL_BALLS = 30

bowl_combined = bowl_combined[
    (bowl_combined['balls_normal']   >= MIN_NORMAL_BOWL_BALLS) &
    (bowl_combined['balls_pressure'] >= MIN_PRESSURE_BOWL_BALLS)
].copy()

# Pressure Bowling Index (PoI) — positive = performs BETTER under pressure
# Better bowling = lower economy, more wickets under pressure
bowl_combined['PoI_economy'] = bowl_combined['economy_normal']  - bowl_combined['economy_pressure']  # positive = tighter
bowl_combined['PoI_sr']      = bowl_combined['sr_normal'].fillna(50) - bowl_combined['sr_pressure'].fillna(50)  # lower SR = quicker wickets
bowl_combined['PoI']         = (bowl_combined['PoI_economy'] * 5.0 +
                                 bowl_combined['PoI_sr'] * 0.5)

bowl_combined.sort_values('PoI', ascending=False, inplace=True)

print(f"Bowling analysis: {len(bowl_combined)} players qualify (≥{MIN_NORMAL_BOWL_BALLS} normal, ≥{MIN_PRESSURE_BOWL_BALLS} pressure balls)")

# ─────────────────────────────────────────────
# 9. SAVE OUTPUTS
# ─────────────────────────────────────────────

os.makedirs(f"{DATA_DIR}/outputs", exist_ok=True)

bbb.to_csv(f"{DATA_DIR}/outputs/enriched_deliveries.csv", index=False)
bat_combined.to_csv(f"{DATA_DIR}/outputs/batting_pressure_analysis.csv", index=False)
bowl_combined.to_csv(f"{DATA_DIR}/outputs/bowling_pressure_analysis.csv", index=False)

# ─────────────────────────────────────────────
# 10. PRINT RANKINGS
# ─────────────────────────────────────────────

def fmt_bat(row):
    return (f"  SR Normal={row['strike_rate_normal']:.1f}  SR Pressure={row['strike_rate_pressure']:.1f}  "
            f"Avg Normal={row['avg_normal']:.1f}  Avg Pressure={row['avg_pressure']:.1f}  "
            f"Balls(P)={int(row['balls_pressure'])}")

def fmt_bowl(row):
    return (f"  Eco Normal={row['economy_normal']:.2f}  Eco Pressure={row['economy_pressure']:.2f}  "
            f"Wkts(P)={int(row['wickets_pressure'])}  Balls(P)={int(row['balls_pressure'])}")

print("\n" + "="*70)
print("TOP 20 BATTERS — Best Under Pressure (Pressure Batting Index, PBI)")
print("="*70)
print("(Positive PBI = performs BETTER under pressure than in normal play)")
for i, (_, row) in enumerate(bat_combined.head(20).iterrows(), 1):
    print(f"\n{i:2}. {row['striker_name']}  PBI={row['PBI']:+.2f}")
    print(fmt_bat(row))

print("\n" + "="*70)
print("BOTTOM 10 BATTERS — Worst Under Pressure (Pressure Batting Index, PBI)")
print("="*70)
for i, (_, row) in enumerate(bat_combined.tail(10).iterrows(), 1):
    print(f"\n{i:2}. {row['striker_name']}  PBI={row['PBI']:+.2f}")
    print(fmt_bat(row))

print("\n" + "="*70)
print("TOP 20 BOWLERS — Best Under Pressure (Pressure Bowling Index, PoI)")
print("="*70)
print("(Positive PoI = more economical AND takes wickets faster under pressure)")
for i, (_, row) in enumerate(bowl_combined.head(20).iterrows(), 1):
    print(f"\n{i:2}. {row['bowler_name']}  PoI={row['PoI']:+.2f}")
    print(fmt_bowl(row))

print("\n" + "="*70)
print("BOTTOM 10 BOWLERS — Worst Under Pressure (Pressure Bowling Index, PoI)")
print("="*70)
for i, (_, row) in enumerate(bowl_combined.tail(10).iterrows(), 1):
    print(f"\n{i:2}. {row['bowler_name']}  PoI={row['PoI']:+.2f}")
    print(fmt_bowl(row))

print("\n\nOutputs saved to:", f"{DATA_DIR}/outputs/")
print("Done!")
