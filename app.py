"""
Cricket Clutch Analytics & Decision Hub
A Streamlit web application to evaluate, visualize, and compare player performances
under high-pressure match situations versus normal levels in T20 cricket.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cricket Clutch Analytics | Decision Hub",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS Styling ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a1a1a;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #28a745;
        font-weight: 500;
    }
    .badge-elite {
        background-color: #d4edda;
        color: #155724;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-solid {
        background-color: #cce5ff;
        color: #004085;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-risk {
        background-color: #f8d7da;
        color: #721c24;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
        border-radius: 4px 4px 0 0;
    }
</style>
""", unsafe_allow_html=True)

DATA_DIR = r"d:/python/RR"

# ── Data Ingestion & Caching ──────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading cricket ball-by-ball match data...")
def load_and_prepare_data():
    enriched_file = os.path.join(DATA_DIR, "outputs", "enriched_deliveries.csv")
    if os.path.exists(enriched_file):
        df = pd.read_csv(enriched_file, low_memory=False)
    else:
        # Fallback to rebuilding if not present
        bbb = pd.read_csv(os.path.join(DATA_DIR, "ball_by_ball.csv"))
        matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))
        comps = pd.read_csv(os.path.join(DATA_DIR, "competitions.csv"))
        df = bbb.merge(matches, on='match_id', how='left').merge(comps, on='comp_id', how='left')

    # Teams mapping
    teams = pd.read_csv(os.path.join(DATA_DIR, "teams.csv"))
    matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))
    t_dict = dict(zip(teams['team_id'], teams['team_name']))
    t_dict.update(dict(zip(matches['home_team_id'], matches['home_team'])))
    t_dict.update(dict(zip(matches['away_team_id'], matches['away_team'])))

    comps = pd.read_csv(os.path.join(DATA_DIR, "competitions.csv"))
    c_dict = dict(zip(comps['comp_id'], comps['comp_name']))

    df['batting_team_name'] = df['batting_team_id'].map(t_dict).fillna("Unknown Team")
    df['bowling_team_name'] = df['bowling_team_id'].map(t_dict).fillna("Unknown Team")
    df['comp_name'] = df['comp_id'].map(c_dict).fillna("Other T20")

    # Competition category: IPL vs International vs Other
    df['is_ipl'] = df['comp_name'].str.contains("IPL|Indian Premier League", case=False, na=False)
    df['comp_category'] = np.where(df['is_ipl'], 'IPL / Franchise', 'T20 International (T20I)')

    # Batting order categorization
    df['role_category'] = pd.cut(
        df['batting_order'],
        bins=[0, 3, 5, 20],
        labels=['Top Order (1-3)', 'Middle Order (4-5)', 'Lower / Finisher (6+)']
    )

    # Bowler earned wickets filter
    earned_wkt_types = ['Bowled', 'Caught', 'LBW', 'Stump', 'Hit Wicket']
    df['bowler_wicket'] = (
        df['is_wicket'] & df['dismissal_type'].isin(earned_wkt_types)
    ).astype(int)

    # Striker dismissal
    df['striker_out'] = (
        df['is_wicket'] & (df['dismissed_player_id'] == df['striker_id'])
    ).astype(int)

    # Runs conceded by bowler
    df['bowler_runs_conceded'] = (
        df['runs'] +
        (df['extras_type'] == 'wide').astype(int) +
        (df['extras_type'] == 'Overstep').astype(int)
    )
    df['bowler_dot'] = (df['bowler_runs_conceded'] == 0).astype(int)
    df['batter_dot'] = (df['runs'] == 0).astype(int)

    return df

df_full = load_and_prepare_data()

# All player names (batters + bowlers) used by the Duel, Scout Card and Lookup tabs
all_players = sorted(
    list(set(df_full['striker_name'].dropna().unique()) | set(df_full['bowler_name'].dropna().unique()))
)

# ── Sidebar Filter Controls ───────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/cricket.png", width=54)
st.sidebar.title("Decision Control Panel")
st.sidebar.caption("Filter match contexts to evaluate player clutch performance.")

# 1. Quick Scenario Preset
quick_preset = st.sidebar.radio(
    "⚡ Quick Scenario Focus",
    ["All Historical Data", "🎯 Latest Season (2026) Only", "🔥 Recent Form (2025-2026)", "🏆 IPL / League Only", "⚡ Knockouts & Finals Only"],
    index=0
)

# Extract available seasons
available_seasons = sorted(df_full['season'].dropna().unique().tolist(), reverse=True)

# 2. Season Selection
if quick_preset == "🎯 Latest Season (2026) Only":
    selected_seasons = ['2026'] if '2026' in available_seasons else available_seasons[:1]
elif quick_preset == "🔥 Recent Form (2025-2026)":
    selected_seasons = [s for s in ['2026', '2025', '2024/25'] if s in available_seasons]
else:
    selected_seasons = st.sidebar.multiselect(
        "📅 Tournament Seasons",
        options=available_seasons,
        default=available_seasons,
        help="Select one or more seasons. For recent form evaluation, select 2026."
    )

if not selected_seasons:
    selected_seasons = available_seasons

# 3. Competition Category
comp_options = ["All Tournaments", "IPL / Franchise", "T20 International (T20I)"]
if quick_preset == "🏆 IPL / League Only":
    selected_comp = "IPL / Franchise"
else:
    selected_comp = st.sidebar.selectbox("🏟️ Tournament Level", comp_options, index=0)

# 4. Match Stage
stage_options = ["All Stages", "Knockouts / Playoffs Only", "League Stage Only"]
if quick_preset == "⚡ Knockouts & Finals Only":
    selected_stage = "Knockouts / Playoffs Only"
else:
    selected_stage = st.sidebar.selectbox("🎖️ Match Stage", stage_options, index=0)

# 5. Innings / Chase Context
selected_innings = st.sidebar.selectbox(
    "🔄 Innings Context",
    ["Both Innings", "2nd Innings Only (Chasing Target)", "1st Innings Only (Setting Target)"],
    index=0
)

# 6. Match Closeness
close_match_only = st.sidebar.checkbox(
    "🎯 Tight Contests Only",
    value=False,
    help="Restricts to matches decided by <=15 runs, <=3 wickets, or Super Overs."
)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Model Parameters")

# 7. Pressure Threshold
cpi_cutoff = st.sidebar.slider(
    "CPI Pressure Threshold",
    min_value=40.0,
    max_value=80.0,
    value=60.0,
    step=5.0,
    help="Deliveries with CPI >= threshold are classified as High Pressure (Default: 60)."
)

# 8. Minimum Sample Sliders
min_pres_balls = st.sidebar.slider("Min Pressure Balls Faced/Bowled", min_value=5, max_value=60, value=15, step=5)
min_norm_balls = st.sidebar.slider("Min Normal Balls Faced/Bowled", min_value=10, max_value=120, value=30, step=10)

# 9. Role Filters
st.sidebar.divider()
st.sidebar.subheader("👤 Role & Player Specialization")
selected_bat_role = st.sidebar.selectbox("Batting Order", ["All Orders", "Top Order (1-3)", "Middle Order (4-5)", "Finishers (6+)"])
selected_bat_hand = st.sidebar.selectbox("Batting Hand", ["All", "RHB", "LHB"])
selected_bowl_style = st.sidebar.selectbox("Bowling Style", ["All", "Pace", "Spin"])

# ── Dynamic Filtering Logic ───────────────────────────────────────────────────
mask = df_full['season'].isin(selected_seasons)

if selected_comp == "IPL / Franchise":
    mask = mask & df_full['is_ipl']
elif selected_comp == "T20 International (T20I)":
    mask = mask & (~df_full['is_ipl'])

if selected_stage == "Knockouts / Playoffs Only":
    mask = mask & (df_full['stage'].isin(['Final', 'Qualifier', 'Eliminator', 'Semi Final', 'Super 4']) | (df_full['knock_out'] == 1))
elif selected_stage == "League Stage Only":
    mask = mask & (df_full['stage'] == 'League')

if selected_innings == "2nd Innings Only (Chasing Target)":
    mask = mask & (df_full['innings'] == 2)
elif selected_innings == "1st Innings Only (Setting Target)":
    mask = mask & (df_full['innings'] == 1)

if close_match_only:
    mask = mask & (df_full['is_close_match'] == 1)

df_active = df_full[mask].copy()

# Recalculate dynamic pressure flag based on slider
df_active['is_pressure_dynamic'] = (df_active['CPI'] >= cpi_cutoff).astype(int)

# ── Metric Aggregation Functions ──────────────────────────────────────────────
@st.cache_data
def get_batting_summary(df_sub, min_n, min_p, role_filter, hand_filter):
    bat = df_sub[df_sub['is_legal']].copy()

    if role_filter != "All Orders":
        bat = bat[bat['role_category'] == role_filter]
    if hand_filter != "All":
        bat = bat[bat['batting_type'] == hand_filter]

    def agg_bat(sub_df, label):
        g = sub_df.groupby(['striker_id', 'striker_name', 'batting_type'])
        res = g.agg(
            balls=('runs', 'count'),
            runs=('runs', 'sum'),
            fours=('is_four', 'sum'),
            sixes=('is_six', 'sum'),
            dots=('batter_dot', 'sum'),
            dismissals=('striker_out', 'sum'),
            matches=('match_id', 'nunique'),
            last_order=('batting_order', 'median')
        ).reset_index()
        res[f'sr_{label}'] = (res['runs'] / res['balls'].clip(lower=1)) * 100.0
        res[f'avg_{label}'] = res['runs'] / res['dismissals'].clip(lower=1)
        res[f'bound_{label}'] = ((res['fours'] + res['sixes']) / res['balls'].clip(lower=1)) * 100.0
        res[f'dot_{label}'] = (res['dots'] / res['balls'].clip(lower=1)) * 100.0
        res.rename(columns={
            'balls': f'balls_{label}', 'runs': f'runs_{label}',
            'fours': f'fours_{label}', 'sixes': f'sixes_{label}',
            'dismissals': f'dismissals_{label}', 'matches': f'matches_{label}'
        }, inplace=True)
        return res

    norm = agg_bat(bat[bat['is_pressure_dynamic'] == 0], 'normal')
    pres = agg_bat(bat[bat['is_pressure_dynamic'] == 1], 'pressure')

    merged = norm.merge(pres, on=['striker_id', 'striker_name', 'batting_type'], how='inner')
    merged = merged[(merged['balls_normal'] >= min_n) & (merged['balls_pressure'] >= min_p)].copy()

    merged['delta_sr'] = merged['sr_pressure'] - merged['sr_normal']
    merged['delta_avg'] = merged['avg_pressure'] - merged['avg_normal']
    merged['delta_bound'] = merged['bound_pressure'] - merged['bound_normal']
    merged['delta_dot'] = merged['dot_pressure'] - merged['dot_normal']

    # Composite Pressure Batting Index (PBI)
    merged['PBI'] = (
        0.50 * merged['delta_sr'] +
        0.30 * merged['delta_avg'] +
        0.20 * merged['delta_bound'] -
        0.20 * merged['delta_dot']
    )

    # Clutch Classification Badge
    conditions = [
        (merged['PBI'] >= 45.0),
        (merged['PBI'] >= 20.0),
        (merged['PBI'] >= 0.0),
    ]
    choices = ['👑 Elite Clutch', '🔥 High Impact', '⚖️ Balanced']
    merged['clutch_rating'] = np.select(conditions, choices, default='⚠️ Choke Risk')

    return merged.sort_values('PBI', ascending=False).reset_index(drop=True)

@st.cache_data
def get_bowling_summary(df_sub, min_n, min_p, style_filter):
    bowl = df_sub.copy()

    if style_filter != "All":
        bowl = bowl[bowl['pace_or_spin'] == style_filter]

    def agg_bowl(sub_df, label):
        g = sub_df.groupby(['bowler_id', 'bowler_name', 'pace_or_spin', 'bowling_type'])
        res = g.agg(
            legal_balls=('is_legal', 'sum'),
            total_balls=('runs', 'count'),
            runs_conceded=('bowler_runs_conceded', 'sum'),
            wickets=('bowler_wicket', 'sum'),
            dots=('bowler_dot', 'sum'),
            fours=('is_four', 'sum'),
            sixes=('is_six', 'sum'),
            matches=('match_id', 'nunique')
        ).reset_index()
        res[f'economy_{label}'] = (res['runs_conceded'] / res['legal_balls'].clip(lower=1)) * 6.0
        res[f'dot_{label}'] = (res['dots'] / res['total_balls'].clip(lower=1)) * 100.0
        res[f'wkt_rate_{label}'] = (res['wickets'] / res['legal_balls'].clip(lower=1)) * 100.0
        res[f'bound_{label}'] = ((res['fours'] + res['sixes']) / res['total_balls'].clip(lower=1)) * 100.0
        res.rename(columns={
            'legal_balls': f'balls_{label}', 'runs_conceded': f'runs_{label}',
            'wickets': f'wickets_{label}', 'dots': f'dots_{label}',
            'matches': f'matches_{label}'
        }, inplace=True)
        return res

    norm = agg_bowl(bowl[bowl['is_pressure_dynamic'] == 0], 'normal')
    pres = agg_bowl(bowl[bowl['is_pressure_dynamic'] == 1], 'pressure')

    merged = norm.merge(pres, on=['bowler_id', 'bowler_name', 'pace_or_spin', 'bowling_type'], how='inner')
    merged = merged[(merged['balls_normal'] >= min_n) & (merged['balls_pressure'] >= min_p)].copy()

    merged['delta_economy'] = merged['economy_normal'] - merged['economy_pressure'] # Positive = more economical under pressure
    merged['delta_dot'] = merged['dot_pressure'] - merged['dot_normal']
    merged['delta_wkt_rate'] = merged['wkt_rate_pressure'] - merged['wkt_rate_normal']
    merged['delta_bound'] = merged['bound_pressure'] - merged['bound_normal']

    # Composite Pressure Bowling Index (PoI)
    merged['PoI'] = (
        4.00 * merged['delta_economy'] +
        0.40 * merged['delta_dot'] +
        20.0 * (merged['delta_wkt_rate'] / 100.0) -
        0.20 * merged['delta_bound']
    )

    conditions = [
        (merged['PoI'] >= 15.0),
        (merged['PoI'] >= 7.0),
        (merged['PoI'] >= 0.0),
    ]
    choices = ['❄️ Ice in Veins', '🎯 Lockdown Specialist', '⚖️ Steady']
    merged['clutch_rating'] = np.select(conditions, choices, default='⚠️ High Pressure Leaker')

    return merged.sort_values('PoI', ascending=False).reset_index(drop=True)

# ── Player Lookup Helpers (used by the new "Player Pressure Lookup" tab) ──────
def _bat_stats(x):
    """Batting stats for a slice of legal deliveries faced."""
    balls = len(x)
    if balls == 0:
        return {'Balls': 0, 'Runs': 0, 'Dismissals': 0, 'SR': np.nan,
                'Avg': np.nan, 'Bound %': np.nan, 'Dot %': np.nan}
    runs = x['runs'].sum()
    outs = x['striker_out'].sum()
    return {
        'Balls': balls,
        'Runs': runs,
        'Dismissals': outs,
        'SR': runs / balls * 100.0,
        'Avg': runs / outs if outs > 0 else np.nan,   # NaN = not out / never dismissed
        'Bound %': (x['is_four'].sum() + x['is_six'].sum()) / balls * 100.0,
        'Dot %': x['batter_dot'].sum() / balls * 100.0,
    }

def _bowl_stats(x):
    """Bowling stats for a slice of deliveries bowled."""
    total = len(x)
    legal = int(x['is_legal'].sum()) if total else 0
    if legal == 0:
        return {'Balls': 0, 'Runs': 0, 'Wickets': 0, 'Economy': np.nan,
                'Dot %': np.nan, 'Bound %': np.nan, 'Wkt Rate %': np.nan}
    runs = x['bowler_runs_conceded'].sum()
    wkts = x['bowler_wicket'].sum()
    return {
        'Balls': legal,
        'Runs': runs,
        'Wickets': wkts,
        'Economy': runs / legal * 6.0,
        'Dot %': x['bowler_dot'].sum() / total * 100.0,
        'Bound %': (x['is_four'].sum() + x['is_six'].sum()) / total * 100.0,
        'Wkt Rate %': wkts / legal * 100.0,
    }

def _avg_clip(x):
    """Batting average with not-outs counted as 1 dismissal (matches main PBI calc)."""
    if len(x) == 0:
        return np.nan
    return x['runs'].sum() / max(x['striker_out'].sum(), 1)

def pbi_tier(v):
    if pd.isna(v): return 'N/A'
    if v >= 45.0: return '👑 Elite Clutch'
    if v >= 20.0: return '🔥 High Impact'
    if v >= 0.0: return '⚖️ Balanced'
    return '⚠️ Choke Risk'

def poi_tier(v):
    if pd.isna(v): return 'N/A'
    if v >= 15.0: return '❄️ Ice in Veins'
    if v >= 7.0: return '🎯 Lockdown Specialist'
    if v >= 0.0: return '⚖️ Steady'
    return '⚠️ High Pressure Leaker'

def build_player_batting_tables(src, players):
    long_rows, delta_rows = [], []
    bat = src[src['is_legal'] & src['striker_name'].isin(players)]
    for p in players:
        sub = bat[bat['striker_name'] == p]
        n = _bat_stats(sub[sub['is_pressure_dynamic'] == 0])
        pr = _bat_stats(sub[sub['is_pressure_dynamic'] == 1])
        long_rows.append({'Player': p, 'Condition': 'Normal', **n})
        long_rows.append({'Player': p, 'Condition': 'High Pressure', **pr})
        d_sr = pr['SR'] - n['SR']
        d_avg = pr['Avg'] - n['Avg']
        d_bound = pr['Bound %'] - n['Bound %']
        d_dot = pr['Dot %'] - n['Dot %']
        # Same PBI formula as the main Batters tab (average uses clip(lower=1) for not-outs)
        avg_n = _avg_clip(sub[sub['is_pressure_dynamic'] == 0])
        avg_p = _avg_clip(sub[sub['is_pressure_dynamic'] == 1])
        pbi = 0.50 * d_sr + 0.30 * (avg_p - avg_n) + 0.20 * d_bound - 0.20 * d_dot
        delta_rows.append({
            'Player': p,
            'Pressure Balls': pr['Balls'],
            'Δ SR': d_sr,
            'Δ Avg': d_avg,
            'Δ Bound %': d_bound,
            'Δ Dot %': d_dot,
            'PBI Index': pbi,
            'Tier': pbi_tier(pbi),
        })
    return pd.DataFrame(long_rows), pd.DataFrame(delta_rows)

def build_player_bowling_tables(src, players):
    long_rows, delta_rows = [], []
    bowl = src[src['bowler_name'].isin(players)]
    for p in players:
        sub = bowl[bowl['bowler_name'] == p]
        n = _bowl_stats(sub[sub['is_pressure_dynamic'] == 0])
        pr = _bowl_stats(sub[sub['is_pressure_dynamic'] == 1])
        long_rows.append({'Player': p, 'Condition': 'Normal', **n})
        long_rows.append({'Player': p, 'Condition': 'High Pressure', **pr})
        d_eco = n['Economy'] - pr['Economy']
        d_dot = pr['Dot %'] - n['Dot %']
        d_wkt = pr['Wkt Rate %'] - n['Wkt Rate %']
        d_bound = pr['Bound %'] - n['Bound %']
        # Same PoI formula as the main Bowlers tab
        poi = 4.00 * d_eco + 0.40 * d_dot + 20.0 * (d_wkt / 100.0) - 0.20 * d_bound
        delta_rows.append({
            'Player': p,
            'Pressure Balls': pr['Balls'],
            'Δ Eco (Tighter +)': d_eco,
            'Δ Dot %': d_dot,
            'Δ Wkt Rate %': d_wkt,
            'Δ Bound %': d_bound,
            'PoI Index': poi,
            'Tier': poi_tier(poi),
        })
    return pd.DataFrame(long_rows), pd.DataFrame(delta_rows)

def highlight_pressure_rows(row):
    color = 'background-color: #fff3cd' if row['Condition'] == 'High Pressure' else ''
    return [color] * len(row)

batting_data = get_batting_summary(df_active, min_norm_balls, min_pres_balls, selected_bat_role, selected_bat_hand)
bowling_data = get_bowling_summary(df_active, min_norm_balls, min_pres_balls, selected_bowl_style)

# ── Main Header ───────────────────────────────────────────────────────────────
st.title("🏏 Cricket Clutch Performance & Decision Matrix")
st.markdown(
    f"**Active Filters**: **{len(selected_seasons)} Seasons** ({', '.join(selected_seasons[:3])}{'...' if len(selected_seasons)>3 else ''}) | "
    f"**Tournament**: {selected_comp} | **Stage**: {selected_stage} | **Innings**: {selected_innings} | "
    f"**Pressure Cutoff**: CPI ≥ {cpi_cutoff:.0f} | **Active Deliveries**: {len(df_active):,} "
    f"({(df_active['is_pressure_dynamic']==1).sum():,} in Clutch Situations)"
)

# ── Top-Level KPI Summary Row ─────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.markdown("""<div class="metric-card"><div class="metric-title">Qualifying Batters</div>"""
                f"""<div class="metric-value">{len(batting_data)}</div>"""
                f"""<div class="metric-sub">≥{min_pres_balls} pressure balls</div></div>""", unsafe_allow_html=True)

with kpi2:
    top_bat = batting_data.iloc[0]['striker_name'] if len(batting_data) > 0 else "N/A"
    top_bat_pbi = f"+{batting_data.iloc[0]['PBI']:.1f}" if len(batting_data) > 0 else "-"
    st.markdown(f"""<div class="metric-card"><div class="metric-title">Top Clutch Batter</div>"""
                f"""<div class="metric-value" style="font-size:1.25rem;">{top_bat}</div>"""
                f"""<div class="metric-sub">PBI: {top_bat_pbi}</div></div>""", unsafe_allow_html=True)

with kpi3:
    st.markdown("""<div class="metric-card"><div class="metric-title">Qualifying Bowlers</div>"""
                f"""<div class="metric-value">{len(bowling_data)}</div>"""
                f"""<div class="metric-sub">≥{min_pres_balls} pressure balls</div></div>""", unsafe_allow_html=True)

with kpi4:
    top_bowl = bowling_data.iloc[0]['bowler_name'] if len(bowling_data) > 0 else "N/A"
    top_bowl_poi = f"+{bowling_data.iloc[0]['PoI']:.1f}" if len(bowling_data) > 0 else "-"
    st.markdown(f"""<div class="metric-card"><div class="metric-title">Top Clutch Bowler</div>"""
                f"""<div class="metric-value" style="font-size:1.25rem;">{top_bowl}</div>"""
                f"""<div class="metric-sub">PoI: {top_bowl_poi}</div></div>""", unsafe_allow_html=True)

with kpi5:
    pct_pressure = (df_active['is_pressure_dynamic'] == 1).mean() * 100
    st.markdown(f"""<div class="metric-card"><div class="metric-title">High-Pressure Balls</div>"""
                f"""<div class="metric-value">{pct_pressure:.1f}%</div>"""
                f"""<div class="metric-sub">CPI ≥ {cpi_cutoff:.0f}</div></div>""", unsafe_allow_html=True)

st.write("")

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab_bat, tab_bowl, tab_allround, tab_duel, tab_player, tab_lookup = st.tabs([
    "🏏 Batters Decision Hub",
    "🎯 Bowlers Lockdown Hub",
    "⚖️ All-Rounders Matrix",
    "⚔️ Head-to-Head Duel (Decision Tool)",
    "🔍 Player Deep-Dive Scout Card",
    "👤 Player Pressure Lookup"
])

# ==============================================================================
# TAB 1: BATTING CLUTCH DECISION HUB
# ==============================================================================
with tab_bat:
    st.subheader("Batting Clutch Performance: Accelerators vs Regressors")
    st.caption("Compares each batter's Strike Rate in Normal conditions vs High-Pressure situations. Upper-left quadrant indicates elite clutch accelerators.")

    if len(batting_data) > 0:
        col_chart, col_side = st.columns([3, 1])

        with col_chart:
            # Interactive Altair Quadrant Chart
            scatter = alt.Chart(batting_data).mark_circle(opacity=0.85, stroke='#333', strokeWidth=0.5).encode(
                x=alt.X('sr_normal:Q', title='Normal Strike Rate (CPI < 60)', scale=alt.Scale(domain=[50, 220])),
                y=alt.Y('sr_pressure:Q', title='High-Pressure Strike Rate (CPI >= 60)', scale=alt.Scale(domain=[50, 260])),
                size=alt.Size('balls_pressure:Q', title='Pressure Balls Faced', scale=alt.Scale(range=[60, 450])),
                color=alt.Color('PBI:Q', title='Pressure Batting Index (PBI)', scale=alt.Scale(scheme='redyellowgreen', domain=[-30, 70])),
                tooltip=[
                    alt.Tooltip('striker_name:N', title='Player'),
                    alt.Tooltip('batting_type:N', title='Hand'),
                    alt.Tooltip('balls_pressure:Q', title='Pressure Balls'),
                    alt.Tooltip('sr_normal:Q', title='Normal SR', format='.1f'),
                    alt.Tooltip('sr_pressure:Q', title='Pressure SR', format='.1f'),
                    alt.Tooltip('delta_sr:Q', title='SR Delta', format='+.1f'),
                    alt.Tooltip('PBI:Q', title='PBI Score', format='+.1f'),
                    alt.Tooltip('clutch_rating:N', title='Rating')
                ]
            ).properties(
                height=420
            ).interactive()

            # Diagonal break-even line (y = x)
            line_data = pd.DataFrame({'x': [50, 240], 'y': [50, 240]})
            ref_line = alt.Chart(line_data).mark_line(color='gray', strokeDash=[4, 4], strokeWidth=1.5).encode(
                x='x:Q', y='y:Q'
            )

            st.altair_chart(scatter + ref_line, use_container_width=True)

        with col_side:
            st.markdown("#### 🎯 Scouting Insights")
            st.info(
                "**Above the dotted line**: Batters who strike faster under pressure than normal play.\n\n"
                "**Finisher Specialists**: Look for high Delta SR (+40+) with at least 30 pressure balls."
            )

            # Quick Top 5 Finishers in Death Overs
            death_bat = df_active[(df_active['overid'] >= 16) & (df_active['is_pressure_dynamic'] == 1)]
            top_death = death_bat.groupby('striker_name').agg(
                runs=('runs', 'sum'), balls=('runs', 'count')
            ).reset_index()
            top_death = top_death[top_death['balls'] >= 15]
            top_death['death_sr'] = (top_death['runs'] / top_death['balls']) * 100
            top_death = top_death.sort_values('death_sr', ascending=False).head(5)

            st.markdown("**Top Death Over Finishers (Overs 16-20):**")
            for _, r in top_death.iterrows():
                st.write(f"• **{r['striker_name']}**: SR **{r['death_sr']:.1f}** ({r['runs']}r/{r['balls']}b)")

        # Full Interactive Data Table
        st.write("### 📋 Complete Batting Decision Table")

        display_bat = batting_data[[
            'striker_name', 'batting_type', 'clutch_rating', 'balls_pressure',
            'sr_normal', 'sr_pressure', 'delta_sr',
            'avg_normal', 'avg_pressure', 'bound_pressure', 'dot_pressure', 'PBI'
        ]].rename(columns={
            'striker_name': 'Player', 'batting_type': 'Hand', 'clutch_rating': 'Tier',
            'balls_pressure': 'Pres Balls', 'sr_normal': 'Normal SR', 'sr_pressure': 'Pressure SR',
            'delta_sr': 'Δ SR', 'avg_normal': 'Norm Avg', 'avg_pressure': 'Pres Avg',
            'bound_pressure': 'Bound %', 'dot_pressure': 'Dot %', 'PBI': 'PBI Index'
        })

        st.dataframe(
            display_bat.style.format({
                'Normal SR': '{:.1f}', 'Pressure SR': '{:.1f}', 'Δ SR': '{:+.1f}',
                'Norm Avg': '{:.1f}', 'Pres Avg': '{:.1f}', 'Bound %': '{:.1f}%',
                'Dot %': '{:.1f}%', 'PBI Index': '{:+.2f}'
            }).background_gradient(subset=['PBI Index'], cmap='RdYlGn', vmin=-25, vmax=60),
            use_container_width=True,
            height=380
        )

        csv_bat = display_bat.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Batting Rankings to CSV", csv_bat, "batting_clutch_rankings.csv", "text/csv")
    else:
        st.warning("No batters meet the current filter criteria. Try lowering the Minimum Balls sliders in the sidebar.")

# ==============================================================================
# TAB 2: BOWLING CLUTCH DECISION HUB
# ==============================================================================
with tab_bowl:
    st.subheader("Bowling Lockdown Performance: Chokers vs Leakers")
    st.caption("Compares each bowler's Economy Rate in Normal play vs High-Pressure situations. Points below the dashed line indicate bowlers who tighten up and concede fewer runs under pressure.")

    if len(bowling_data) > 0:
        col_bchart, col_bside = st.columns([3, 1])

        with col_bchart:
            # Interactive Altair Quadrant Chart for Bowling
            scatter_bowl = alt.Chart(bowling_data).mark_circle(opacity=0.85, stroke='#333', strokeWidth=0.5).encode(
                x=alt.X('economy_normal:Q', title='Normal Economy Rate', scale=alt.Scale(domain=[4, 13])),
                y=alt.Y('economy_pressure:Q', title='High-Pressure Economy Rate', scale=alt.Scale(domain=[4, 15])),
                size=alt.Size('balls_pressure:Q', title='Pressure Balls Bowled', scale=alt.Scale(range=[60, 450])),
                color=alt.Color('PoI:Q', title='Pressure Bowling Index (PoI)', scale=alt.Scale(scheme='redyellowgreen', domain=[-15, 25])),
                tooltip=[
                    alt.Tooltip('bowler_name:N', title='Bowler'),
                    alt.Tooltip('pace_or_spin:N', title='Style'),
                    alt.Tooltip('bowling_type:N', title='Type'),
                    alt.Tooltip('balls_pressure:Q', title='Pressure Balls'),
                    alt.Tooltip('economy_normal:Q', title='Normal Eco', format='.2f'),
                    alt.Tooltip('economy_pressure:Q', title='Pressure Eco', format='.2f'),
                    alt.Tooltip('delta_economy:Q', title='Eco Gain', format='+.2f'),
                    alt.Tooltip('wickets_pressure:Q', title='Pressure Wkts'),
                    alt.Tooltip('PoI:Q', title='PoI Score', format='+.2f'),
                    alt.Tooltip('clutch_rating:N', title='Rating')
                ]
            ).properties(
                height=420
            ).interactive()

            b_line_data = pd.DataFrame({'x': [4, 14], 'y': [4, 14]})
            b_ref_line = alt.Chart(b_line_data).mark_line(color='gray', strokeDash=[4, 4], strokeWidth=1.5).encode(
                x='x:Q', y='y:Q'
            )

            st.altair_chart(scatter_bowl + b_ref_line, use_container_width=True)

        with col_bside:
            st.markdown("#### 🎯 Scouting Insights")
            st.info(
                "**Below the line**: Bowlers whose economy rate drops under pressure.\n\n"
                "**Death Over Enforcers**: Look for high dot ball % (>40%) and low pressure economy (<8.0)."
            )

            # Death overs bowler rankings
            death_bowl = df_active[(df_active['overid'] >= 16) & (df_active['is_pressure_dynamic'] == 1)]
            top_death_b = death_bowl.groupby('bowler_name').agg(
                runs=('bowler_runs_conceded', 'sum'),
                balls=('is_legal', 'sum'),
                wkts=('bowler_wicket', 'sum')
            ).reset_index()
            top_death_b = top_death_b[top_death_b['balls'] >= 15]
            top_death_b['death_eco'] = (top_death_b['runs'] / top_death_b['balls'].clip(lower=1)) * 6.0
            top_death_b = top_death_b.sort_values('death_eco', ascending=True).head(5)

            st.markdown("**Top Death Over Bowlers (Overs 16-20):**")
            for _, r in top_death_b.iterrows():
                st.write(f"• **{r['bowler_name']}**: Eco **{r['death_eco']:.2f}** ({r['wkts']}w/{r['balls']}b)")

        st.write("### 📋 Complete Bowling Decision Table")

        display_bowl = bowling_data[[
            'bowler_name', 'pace_or_spin', 'bowling_type', 'clutch_rating', 'balls_pressure',
            'economy_normal', 'economy_pressure', 'delta_economy',
            'wickets_pressure', 'dot_pressure', 'bound_pressure', 'PoI'
        ]].rename(columns={
            'bowler_name': 'Bowler', 'pace_or_spin': 'Class', 'bowling_type': 'Action',
            'clutch_rating': 'Tier', 'balls_pressure': 'Pres Balls',
            'economy_normal': 'Norm Eco', 'economy_pressure': 'Pres Eco',
            'delta_economy': 'Δ Eco (Tighter)', 'wickets_pressure': 'Pres Wkts',
            'dot_pressure': 'Dot %', 'bound_pressure': 'Bound %', 'PoI': 'PoI Index'
        })

        st.dataframe(
            display_bowl.style.format({
                'Norm Eco': '{:.2f}', 'Pres Eco': '{:.2f}', 'Δ Eco (Tighter)': '{:+.2f}',
                'Dot %': '{:.1f}%', 'Bound %': '{:.1f}%', 'PoI Index': '{:+.2f}'
            }).background_gradient(subset=['PoI Index'], cmap='RdYlGn', vmin=-15, vmax=25),
            use_container_width=True,
            height=380
        )

        csv_bowl = display_bowl.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Bowling Rankings to CSV", csv_bowl, "bowling_clutch_rankings.csv", "text/csv")
    else:
        st.warning("No bowlers meet the current filter criteria. Try lowering the Minimum Balls sliders in the sidebar.")

# ==============================================================================
# TAB 3: ALL-ROUNDERS MATRIX
# ==============================================================================
with tab_allround:
    st.subheader("All-Rounders & Dual Impact Matrix")
    st.caption("Identifies dual-threat cricketers who withstand and excel under pressure in both batting and bowling disciplines.")

    if len(batting_data) > 0 and len(bowling_data) > 0:
        ar_bat = batting_data[['striker_name', 'PBI', 'balls_pressure', 'sr_pressure']].rename(
            columns={'striker_name': 'player_name', 'balls_pressure': 'bat_balls_p', 'sr_pressure': 'bat_sr_p'}
        )
        ar_bowl = bowling_data[['bowler_name', 'PoI', 'balls_pressure', 'economy_pressure']].rename(
            columns={'bowler_name': 'player_name', 'balls_pressure': 'bowl_balls_p', 'economy_pressure': 'bowl_eco_p'}
        )

        allrounders = ar_bat.merge(ar_bowl, on='player_name', how='inner')
        allrounders['dual_impact_score'] = (allrounders['PBI'] * 0.5) + (allrounders['PoI'] * 1.5)
        allrounders.sort_values('dual_impact_score', ascending=False, inplace=True)
        allrounders.reset_index(drop=True, inplace=True)

        if len(allrounders) > 0:
            col_ar1, col_ar2 = st.columns([2, 1])

            with col_ar1:
                ar_chart = alt.Chart(allrounders).mark_circle(size=180, opacity=0.85).encode(
                    x=alt.X('PBI:Q', title='Batting Clutch Index (PBI)'),
                    y=alt.Y('PoI:Q', title='Bowling Clutch Index (PoI)'),
                    color=alt.Color('dual_impact_score:Q', scale=alt.Scale(scheme='viridis'), title='Dual Score'),
                    tooltip=[
                        alt.Tooltip('player_name:N', title='Player'),
                        alt.Tooltip('PBI:Q', title='Batting PBI', format='+.1f'),
                        alt.Tooltip('PoI:Q', title='Bowling PoI', format='+.1f'),
                        alt.Tooltip('bat_sr_p:Q', title='Pres SR', format='.1f'),
                        alt.Tooltip('bowl_eco_p:Q', title='Pres Eco', format='.2f'),
                        alt.Tooltip('dual_impact_score:Q', title='Dual Impact', format='+.1f')
                    ]
                ).properties(height=380).interactive()

                st.altair_chart(ar_chart, use_container_width=True)

            with col_ar2:
                st.markdown("#### 💎 Top 5 Dual-Impact Match Winners")
                for i, r in allrounders.head(5).iterrows():
                    st.markdown(f"**{i+1}. {r['player_name']}**")
                    st.write(f"• Batting: PBI **{r['PBI']:+.1f}** (SR: {r['bat_sr_p']:.1f})")
                    st.write(f"• Bowling: PoI **{r['PoI']:+.1f}** (Eco: {r['bowl_eco_p']:.2f})")
                    st.write(f"• Overall Dual Index: **{r['dual_impact_score']:.1f}**")
                    st.write("---")

            st.write("### 📋 All-Rounder Clutch Rankings")
            st.dataframe(
                allrounders.rename(columns={
                    'player_name': 'Player', 'bat_balls_p': 'Bat Balls (Pres)',
                    'bat_sr_p': 'Bat SR (Pres)', 'bowl_balls_p': 'Bowl Balls (Pres)',
                    'bowl_eco_p': 'Bowl Eco (Pres)', 'PBI': 'Batting PBI',
                    'PoI': 'Bowling PoI', 'dual_impact_score': 'Dual Score'
                }).style.format({
                    'Bat SR (Pres)': '{:.1f}', 'Bowl Eco (Pres)': '{:.2f}',
                    'Batting PBI': '{:+.2f}', 'Bowling PoI': '{:+.2f}', 'Dual Score': '{:+.2f}'
                }),
                use_container_width=True
            )
        else:
            st.info("No players meet the minimum pressure qualification criteria in both batting and bowling simultaneously. Try lowering the ball thresholds.")
    else:
        st.info("Insufficient qualifying data for all-rounders with active filters.")

# ==============================================================================
# TAB 4: HEAD-TO-HEAD DUEL (DECISION TOOL)
# ==============================================================================
with tab_duel:
    st.subheader("⚔️ Head-to-Head Clutch Decision Matrix")
    st.caption("Select two players to run a direct statistical comparison for team selection, auction bids, or role assignments.")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        p1 = st.selectbox("Player A (Baseline)", options=all_players, index=all_players.index("Rohit Sharma") if "Rohit Sharma" in all_players else 0)
    with col_p2:
        p2 = st.selectbox("Player B (Challenger)", options=all_players, index=all_players.index("MS Dhoni") if "MS Dhoni" in all_players else 1)

    if p1 and p2 and p1 != p2:
        # Compute player profiles across full history vs active filter
        def get_profile(name):
            bat_sub = df_full[(df_full['striker_name'] == name) & (df_full['is_legal'])]
            bowl_sub = df_full[(df_full['bowler_name'] == name)]

            # Batting
            bat_norm = bat_sub[bat_sub['CPI'] < cpi_cutoff]
            bat_pres = bat_sub[bat_sub['CPI'] >= cpi_cutoff]

            sr_n = (bat_norm['runs'].sum() / max(len(bat_norm), 1)) * 100
            sr_p = (bat_pres['runs'].sum() / max(len(bat_pres), 1)) * 100
            dots_p = (bat_pres['batter_dot'].sum() / max(len(bat_pres), 1)) * 100
            bounds_p = ((bat_pres['is_four'].sum() + bat_pres['is_six'].sum()) / max(len(bat_pres), 1)) * 100

            # Bowling
            bowl_norm = bowl_sub[bowl_sub['CPI'] < cpi_cutoff]
            bowl_pres = bowl_sub[bowl_sub['CPI'] >= cpi_cutoff]

            eco_n = (bowl_norm['bowler_runs_conceded'].sum() / max(bowl_norm['is_legal'].sum(), 1)) * 6.0
            eco_p = (bowl_pres['bowler_runs_conceded'].sum() / max(bowl_pres['is_legal'].sum(), 1)) * 6.0
            wkts_p = bowl_pres['bowler_wicket'].sum()

            # Recent Season (2026) Form
            bat_2026 = bat_sub[(bat_sub['season'] == '2026') & (bat_sub['CPI'] >= cpi_cutoff)]
            sr_2026 = (bat_2026['runs'].sum() / max(len(bat_2026), 1)) * 100 if len(bat_2026) > 0 else np.nan

            bowl_2026 = bowl_sub[(bowl_sub['season'] == '2026') & (bowl_sub['CPI'] >= cpi_cutoff)]
            eco_2026 = (bowl_2026['bowler_runs_conceded'].sum() / max(bowl_2026['is_legal'].sum(), 1)) * 6.0 if len(bowl_2026) > 0 else np.nan

            return {
                'name': name,
                'bat_balls_p': len(bat_pres),
                'sr_n': sr_n,
                'sr_p': sr_p,
                'delta_sr': sr_p - sr_n,
                'dots_p': dots_p,
                'bounds_p': bounds_p,
                'sr_2026': sr_2026,
                'bat_balls_2026': len(bat_2026),
                'bowl_balls_p': bowl_pres['is_legal'].sum(),
                'eco_n': eco_n,
                'eco_p': eco_p,
                'delta_eco': eco_n - eco_p,
                'wkts_p': wkts_p,
                'eco_2026': eco_2026,
                'bowl_balls_2026': bowl_2026['is_legal'].sum()
            }

        prof1 = get_profile(p1)
        prof2 = get_profile(p2)

        st.write("### 📊 Direct Metrics Comparison")

        comp_df = pd.DataFrame([
            {"Metric": "Pressure Balls Faced (Batting)", p1: f"{prof1['bat_balls_p']}", p2: f"{prof2['bat_balls_p']}", "Advantage": p1 if prof1['bat_balls_p'] > prof2['bat_balls_p'] else p2},
            {"Metric": "Normal Strike Rate", p1: f"{prof1['sr_n']:.1f}", p2: f"{prof2['sr_n']:.1f}", "Advantage": p1 if prof1['sr_n'] > prof2['sr_n'] else p2},
            {"Metric": "High-Pressure Strike Rate", p1: f"{prof1['sr_p']:.1f}", p2: f"{prof2['sr_p']:.1f}", "Advantage": p1 if prof1['sr_p'] > prof2['sr_p'] else p2},
            {"Metric": "Strike Rate Clutch Delta (Δ SR)", p1: f"{prof1['delta_sr']:+.1f}", p2: f"{prof2['delta_sr']:+.1f}", "Advantage": p1 if prof1['delta_sr'] > prof2['delta_sr'] else p2},
            {"Metric": "Boundary % Under Pressure", p1: f"{prof1['bounds_p']:.1f}%", p2: f"{prof2['bounds_p']:.1f}%", "Advantage": p1 if prof1['bounds_p'] > prof2['bounds_p'] else p2},
            {"Metric": "Dot Ball % Under Pressure (Lower is better)", p1: f"{prof1['dots_p']:.1f}%", p2: f"{prof2['dots_p']:.1f}%", "Advantage": p1 if prof1['dots_p'] < prof2['dots_p'] else p2},
            {"Metric": "Latest Season (2026) Pressure SR", p1: f"{prof1['sr_2026']:.1f} ({prof1['bat_balls_2026']}b)" if pd.notna(prof1['sr_2026']) else "N/A", p2: f"{prof2['sr_2026']:.1f} ({prof2['bat_balls_2026']}b)" if pd.notna(prof2['sr_2026']) else "N/A", "Advantage": "N/A"},
            {"Metric": "Pressure Balls Bowled", p1: f"{prof1['bowl_balls_p']}", p2: f"{prof2['bowl_balls_p']}", "Advantage": p1 if prof1['bowl_balls_p'] > prof2['bowl_balls_p'] else p2},
            {"Metric": "High-Pressure Economy (Lower is better)", p1: f"{prof1['eco_p']:.2f}" if prof1['bowl_balls_p']>0 else "N/A", p2: f"{prof2['eco_p']:.2f}" if prof2['bowl_balls_p']>0 else "N/A", "Advantage": p1 if (prof1['bowl_balls_p']>0 and prof1['eco_p'] < prof2['eco_p']) else p2},
        ])

        st.table(comp_df.set_index("Metric"))

        # Tactical Verdict
        st.markdown("#### 💡 Tactical Recommendation & Scouting Verdict")
        if prof1['bat_balls_p'] >= 20 and prof2['bat_balls_p'] >= 20:
            winner = p1 if prof1['delta_sr'] > prof2['delta_sr'] else p2
            margin = abs(prof1['delta_sr'] - prof2['delta_sr'])
            st.success(
                f"**Batting Clutch Pick**: **{winner}** is the statistically superior high-pressure batter with a **{margin:.1f} point advantage** in clutch Strike Rate acceleration. "
                f"Under high leverage (CPI ≥ {cpi_cutoff:.0f}), {winner} elevates their scoring tempo while minimizing dot balls."
            )
        else:
            st.info("One or both players have low pressure sample size in batting. Check bowling metrics or lower the threshold.")

# ==============================================================================
# TAB 5: PLAYER DEEP-DIVE SCOUT CARD
# ==============================================================================
with tab_player:
    st.subheader("🔍 Individual Player Pressure Scout Card")
    st.caption("Deep-dive into any player's career clutch trajectory, phase splits, and season-by-season progression.")

    selected_scout = st.selectbox("Select Player to Inspect", options=all_players, index=all_players.index("MS Dhoni") if "MS Dhoni" in all_players else 0)

    if selected_scout:
        p_bat = df_full[df_full['striker_name'] == selected_scout].copy()
        p_bowl = df_full[df_full['bowler_name'] == selected_scout].copy()

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Total Deliveries Faced", f"{len(p_bat):,}")
        with col_s2:
            pres_faced = (p_bat['CPI'] >= cpi_cutoff).sum()
            st.metric("Clutch Deliveries Faced", f"{pres_faced:,}", f"{(pres_faced/max(len(p_bat),1)*100):.1f}% of total")
        with col_s3:
            st.metric("Total Deliveries Bowled", f"{len(p_bowl):,}")
        with col_s4:
            pres_bowled = (p_bowl['CPI'] >= cpi_cutoff).sum()
            st.metric("Clutch Deliveries Bowled", f"{pres_bowled:,}", f"{(pres_bowled/max(len(p_bowl),1)*100):.1f}% of total")

        st.write("---")

        # 1. Season-by-Season Progression
        col_prog1, col_prog2 = st.columns(2)

        with col_prog1:
            st.markdown("#### 📈 Batting Strike Rate Progression Under Pressure")
            if pres_faced > 0:
                p_bat['is_p'] = p_bat['CPI'] >= cpi_cutoff
                season_bat = p_bat.groupby(['season', 'is_p']).agg(
                    runs=('runs', 'sum'), balls=('runs', 'count')
                ).reset_index()
                season_bat['sr'] = (season_bat['runs'] / season_bat['balls'].clip(lower=1)) * 100
                season_bat['Status'] = np.where(season_bat['is_p'], 'Pressure (CPI ≥ 60)', 'Normal')

                chart_prog = alt.Chart(season_bat).mark_bar().encode(
                    x=alt.X('season:N', title='Season'),
                    y=alt.Y('sr:Q', title='Strike Rate'),
                    color=alt.Color('Status:N', scale=alt.Scale(domain=['Normal', 'Pressure (CPI ≥ 60)'], range=['#708090', '#2E8B57'])),
                    xOffset='Status:N',
                    tooltip=['season', 'Status', alt.Tooltip('sr:Q', format='.1f'), 'balls']
                ).properties(height=280)

                st.altair_chart(chart_prog, use_container_width=True)
            else:
                st.info("No high-pressure batting deliveries recorded for this player.")

        with col_prog2:
            st.markdown("#### 🎯 Phase-by-Phase Performance Splits")
            if len(p_bat) > 0:
                p_bat['is_p'] = p_bat['CPI'] >= cpi_cutoff
                phase_bat = p_bat.groupby(['phase', 'is_p'], observed=True).agg(
                    runs=('runs', 'sum'), balls=('runs', 'count')
                ).reset_index()
                phase_bat['sr'] = (phase_bat['runs'] / phase_bat['balls'].clip(lower=1)) * 100
                phase_bat['Condition'] = np.where(phase_bat['is_p'], 'Pressure', 'Normal')

                chart_phase = alt.Chart(phase_bat).mark_bar().encode(
                    x=alt.X('phase:N', title='Innings Phase'),
                    y=alt.Y('sr:Q', title='Strike Rate'),
                    color=alt.Color('Condition:N', scale=alt.Scale(domain=['Normal', 'Pressure'], range=['#4682B4', '#DC143C'])),
                    xOffset='Condition:N',
                    tooltip=['phase', 'Condition', alt.Tooltip('sr:Q', format='.1f'), 'balls']
                ).properties(height=280)

                st.altair_chart(chart_phase, use_container_width=True)
            else:
                st.info("No batting deliveries found.")

        # Top High-Pressure Moments Log
        st.write("#### ⚡ Top 10 Highest Leverage Match Moments Experienced")
        sample_moments = p_bat.sort_values('CPI', ascending=False).head(10)[[
            'match_id', 'season', 'comp_name', 'innings', 'overid', 'over_ball_num',
            'runs', 'is_four', 'is_six', 'is_wicket', 'CPI'
        ]].rename(columns={
            'season': 'Season', 'comp_name': 'Tournament', 'innings': 'Inn',
            'overid': 'Over', 'over_ball_num': 'Ball', 'runs': 'Runs',
            'is_four': '4', 'is_six': '6', 'is_wicket': 'Wkt', 'CPI': 'CPI Score'
        })
        st.dataframe(sample_moments.style.format({'CPI Score': '{:.1f}'}), use_container_width=True)

# ==============================================================================
# TAB 6: PLAYER PRESSURE LOOKUP  (NEW)
# Pick one or more players from the dropdown -> only their data is shown
# ==============================================================================
with tab_lookup:
    st.subheader("👤 Player Pressure Lookup")
    st.caption("Choose player name(s) from the dropdown (you can also type to search). Only the selected players' normal vs high-pressure data is shown.")

    col_input, col_opts = st.columns([3, 1])

    with col_input:
        chosen_players = st.multiselect(
            "🔎 Player Name(s)",
            options=all_players,
            default=[],
            placeholder="Click here and pick or type a player name...",
            key="lookup_players"
        )

    with col_opts:
        use_full_history = st.checkbox(
            "Ignore sidebar filters (full history)",
            value=False,
            help="Off = uses the sidebar filters (seasons, tournament, stage, innings...). On = uses all available data.",
            key="lookup_full"
        )
        view_mode = st.radio("Show", ["Batting + Bowling", "Batting only", "Bowling only"], index=0, key="lookup_view")

    if not chosen_players:
        st.info("👆 Select at least one player from the dropdown to see their pressure data.")
    else:
        base_df = df_full if use_full_history else df_active
        src = base_df[
            base_df['striker_name'].isin(chosen_players) | base_df['bowler_name'].isin(chosen_players)
        ].copy()
        # Always classify pressure using the current CPI slider value
        src['is_pressure_dynamic'] = (src['CPI'] >= cpi_cutoff).astype(int)

        scope = "full history" if use_full_history else "sidebar-filtered data"
        st.markdown(
            f"**Showing**: {', '.join(chosen_players)} · **Scope**: {scope} · **Pressure cutoff**: CPI ≥ {cpi_cutoff:.0f}"
        )

        export_frames = []

        # ── Batting ──
        if view_mode in ("Batting + Bowling", "Batting only"):
            st.markdown("### 🏏 Batting – Normal vs High Pressure")
            bat_long, bat_delta = build_player_batting_tables(src, chosen_players)

            if bat_long['Balls'].sum() == 0:
                st.warning("No batting deliveries found for the selected player(s) under the current filters.")
            else:
                st.dataframe(
                    bat_long.style.apply(highlight_pressure_rows, axis=1).format({
                        'Balls': '{:,.0f}', 'Runs': '{:,.0f}', 'Dismissals': '{:,.0f}',
                        'SR': '{:.1f}', 'Avg': '{:.1f}', 'Bound %': '{:.1f}%', 'Dot %': '{:.1f}%'
                    }, na_rep="-"),
                    use_container_width=True,
                    hide_index=True
                )
                st.caption("Avg shows “-” when the player was not dismissed in that sample. Highlighted rows = high-pressure deliveries.")

                st.markdown("**Clutch Delta & Pressure Batting Index (PBI)**")
                st.dataframe(
                    bat_delta.style.format({
                        'Pressure Balls': '{:,.0f}', 'Δ SR': '{:+.1f}', 'Δ Avg': '{:+.1f}',
                        'Δ Bound %': '{:+.1f}', 'Δ Dot %': '{:+.1f}', 'PBI Index': '{:+.2f}'
                    }, na_rep="-"),
                    use_container_width=True,
                    hide_index=True
                )
                export_frames.append(bat_long.merge(bat_delta[['Player', 'PBI Index', 'Tier']], on='Player', how='left').assign(Discipline='Batting'))

        # ── Bowling ──
        if view_mode in ("Batting + Bowling", "Bowling only"):
            st.markdown("### 🎯 Bowling – Normal vs High Pressure")
            bowl_long, bowl_delta = build_player_bowling_tables(src, chosen_players)

            if bowl_long['Balls'].sum() == 0:
                st.warning("No bowling deliveries found for the selected player(s) under the current filters.")
            else:
                st.dataframe(
                    bowl_long.style.apply(highlight_pressure_rows, axis=1).format({
                        'Balls': '{:,.0f}', 'Runs': '{:,.0f}', 'Wickets': '{:,.0f}',
                        'Economy': '{:.2f}', 'Dot %': '{:.1f}%', 'Bound %': '{:.1f}%', 'Wkt Rate %': '{:.2f}%'
                    }, na_rep="-"),
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("**Clutch Delta & Pressure Bowling Index (PoI)** · positive Δ Eco = tighter under pressure")
                st.dataframe(
                    bowl_delta.style.format({
                        'Pressure Balls': '{:,.0f}', 'Δ Eco (Tighter +)': '{:+.2f}',
                        'Δ Dot %': '{:+.1f}', 'Δ Wkt Rate %': '{:+.2f}', 'Δ Bound %': '{:+.1f}', 'PoI Index': '{:+.2f}'
                    }, na_rep="-"),
                    use_container_width=True,
                    hide_index=True
                )
                export_frames.append(bowl_long.merge(bowl_delta[['Player', 'PoI Index', 'Tier']], on='Player', how='left').assign(Discipline='Bowling'))

        if export_frames:
            export_df = pd.concat(export_frames, ignore_index=True)
            st.download_button(
                "📥 Export Selected Player Data to CSV",
                export_df.to_csv(index=False).encode('utf-8'),
                "selected_player_pressure_data.csv",
                "text/csv",
                key="lookup_download"
            )

# ── Footer / Methodology Accordion ────────────────────────────────────────────
st.divider()
with st.expander("📖 Methodology & CPI Mathematical Formulation"):
    st.markdown(r"""
    ### What is the Composite Pressure Index (CPI)?
    Every delivery in the database is evaluated on a $0$ to $100$ scale:
    $$\text{CPI} = 100 \times \left( 0.35 \cdot \mathcal{S}_{\text{equation}} + 0.25 \cdot \mathcal{S}_{\text{wickets}} + 0.20 \cdot \mathcal{S}_{\text{phase}} + 0.10 \cdot \mathcal{S}_{\text{urgency}} + 0.10 \cdot \mathcal{S}_{\text{stakes}} \right)$$

    1. **$\mathcal{S}_{\text{equation}}$ (35%)**: Sigmoidal distance between Required Run Rate and Current Run Rate in 2nd innings; par scoring demand in 1st innings.
    2. **$\mathcal{S}_{\text{wickets}}$ (25%)**: $(10 - \text{wickets in hand}) / 9$. Captures catastrophic risk when only tailenders remain.
    3. **$\mathcal{S}_{\text{phase}}$ (20%)**: Death overs ($1.00$), Middle ($0.40$), Powerplay ($0.15$).
    4. **$\mathcal{S}_{\text{urgency}}$ (10%)**: Depletion of legal balls in the innings.
    5. **$\mathcal{S}_{\text{stakes}}$ (10%)**: Close contest indicator (margin $\le 15$ runs or $\le 3$ wkts) + Knockout tournament status.

    ### Pressure Batting Index (PBI)
    $$\text{PBI} = 0.50 \cdot (\text{SR}_P - \text{SR}_N) + 0.30 \cdot (\text{Avg}_P - \text{Avg}_N) + 0.20 \cdot (\text{Bound\%}_P - \text{Bound\%}_N) - 0.20 \cdot (\text{Dot\%}_P - \text{Dot\%}_N)$$

    ### Pressure Bowling Index (PoI)
    $$\text{PoI} = 4.00 \cdot (\text{Eco}_N - \text{Eco}_P) + 0.40 \cdot (\text{Dot\%}_P - \text{Dot\%}_N) + 20.0 \cdot (\text{WktRate}_P - \text{WktRate}_N) - 0.20 \cdot (\text{Bound\%}_P - \text{Bound\%}_N)$$
    """)