import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import umap  # kept for internal embedding if you want later

# ============================================================
# 1. Load data
# ============================================================

@st.cache_data
def load_full_data():
    # Adjust path if needed
    df = pd.read_csv("data/master data/nba_fyp_final_dataset.csv")
    return df

full = load_full_data()

# Ensure minutes_played column exists
if "minutes_played" not in full.columns:
    if "pass__minutes" in full.columns:
        full["minutes_played"] = np.where(
            full["pass__minutes"] == 0,
            np.nan,
            full["pass__minutes"]
        )
    else:
        full["minutes_played"] = np.nan

# Create prefixed usage columns if present
if "usg_pct" in full.columns:
    full["usage__usg_pct"] = full["usg_pct"]
if "offensive_load" in full.columns:
    full["usage__offensive_load"] = full["offensive_load"]
if "box_creation" in full.columns:
    full["usage__box_creation"] = full["box_creation"]

cols = full.columns.tolist()

# ============================================================
# 2. Category definitions
# ============================================================

categories = {
    "stature": [
        c for c in cols
        if c.startswith("stature__") and not c.endswith("_pct")
    ],
    "athleticism": ["ath__athleticism_score"],
    # ONLY the five shot frequency columns for shot selection
    "shot_selection": [
        c for c in [
            "shot__freq_rim",
            "shot__freq_paint",
            "shot__freq_mid",
            "shot__freq_corner3",
            "shot__freq_atb3",
        ] if c in cols
    ],
    "shooting": ["shoot__quality"],
    "usage": [
        c for c in [
            "usage__usg_pct",
            "usage__offensive_load",
            "usage__box_creation",
        ] if c in cols
    ],
    "passing": ["pass__passer_rating"],
    "defense": ["defense__crafted_dpm"],
    "playstyle": [c for c in cols if c.startswith("playstyle__")],
    "portability": ["portability__score"],
    "dominance": ["dominance__cpm"],
}

# Flatten feature list (all raw features used for scaling)
all_feature_cols = sorted({c for cols_list in categories.values() for c in cols_list})
feature_df = full[all_feature_cols].astype(float)

# Percentile scaling (for CLUSTERING & LEAGUE COMPARISON)
scaled = feature_df.rank(pct=True)

# Separate scaling for SIMILARITY engine (min–max, NOT percentiles)
sim_feature_df = feature_df.astype(float)
sim_min = sim_feature_df.min()
sim_max = sim_feature_df.max()
sim_range = (sim_max - sim_min).replace(0, np.nan)
sim_scaled = (sim_feature_df - sim_min) / sim_range
sim_scaled = sim_scaled.fillna(0.0)

# ============================================================
# 3. Shot selection – average distance for clustering/graph
# ============================================================

shot_dist_weights = {
    "shot__freq_rim":      1.0,
    "shot__freq_paint":    2.0,
    "shot__freq_mid":      3.0,
    "shot__freq_corner3":  4.0,
    "shot__freq_atb3":     5.0,
}

shot_dist_cols = [c for c in shot_dist_weights if c in full.columns]

if shot_dist_cols:
    freqs = full[shot_dist_cols].astype(float)
    freq_sum = freqs.sum(axis=1)
    freq_norm = freqs.div(freq_sum.replace(0, np.nan), axis=0)

    w = pd.Series({c: shot_dist_weights[c] for c in shot_dist_cols})
    full["shot_avg_distance_raw"] = (freq_norm * w).sum(axis=1)
    full["shot_avg_distance_pct"] = full["shot_avg_distance_raw"].rank(pct=True)
else:
    full["shot_avg_distance_pct"] = np.nan

# For similarity engine
SHOT_FREQ_COLS = [
    c for c in [
        "shot__freq_rim",
        "shot__freq_paint",
        "shot__freq_mid",
        "shot__freq_corner3",
        "shot__freq_atb3",
    ] if c in full.columns
]

# ============================================================
# 3b. Playstyle – creation index for clustering/graph
# ============================================================

playstyle_weights = {
    "playstyle__iso_freq":        5.0,
    "playstyle__pr_handler_freq": 5.0,
    "playstyle__handoff_freq":    4.0,
    "playstyle__pr_roll_freq":    3.0,
    "playstyle__post_freq":       2.0,
    "playstyle__spotup_freq":     1.0,
    "playstyle__offscreen_freq":  1.0,
    "playstyle__cut_freq":        0.5,
}

playstyle_cols_weighted = [c for c in playstyle_weights if c in full.columns]

if playstyle_cols_weighted:
    ps_vals = full[playstyle_cols_weighted].astype(float).clip(lower=0)
    ps_sum = ps_vals.sum(axis=1)
    ps_norm = ps_vals.div(ps_sum.replace(0, np.nan), axis=0)

    w_ps = pd.Series({c: playstyle_weights[c] for c in playstyle_cols_weighted})
    full["playstyle_creation_index_raw"] = (ps_norm * w_ps).sum(axis=1)
    full["playstyle_creation_index_pct"] = full["playstyle_creation_index_raw"].rank(pct=True)
else:
    full["playstyle_creation_index_pct"] = np.nan

# ============================================================
# 4. Stature subfeatures for radar
# ============================================================

STATURE_COLS_GRAPH = []
for c in full.columns:
    lc = c.lower()
    if c.startswith("stature__"):
        if "height" in lc or "wingspan" in lc or "weight" in lc:
            STATURE_COLS_GRAPH.append(c)
STATURE_COLS_GRAPH = [c for c in sorted(set(STATURE_COLS_GRAPH)) if c in scaled.columns]

# 10 categories (including playstyle) for radar + clustering
RADAR_CATEGORIES = list(categories.keys())

# ============================================================
# 5. Category profiles (player + league)
# ============================================================

def get_player_category_profile_for_graph(player_name: str, df=full) -> pd.Series:
    mask = df["player_name"].str.lower() == player_name.lower()
    if not mask.any():
        raise ValueError(f"Player '{player_name}' not found.")
    idx = df.index[mask][0]

    scores = {}

    for cat in RADAR_CATEGORIES:

        # STATURE
        if cat == "stature":
            if not STATURE_COLS_GRAPH:
                scores[cat] = np.nan
                continue
            vals = scaled.loc[idx, STATURE_COLS_GRAPH].astype(float).values
            vals = vals[~np.isnan(vals)]
            scores[cat] = float(vals.mean()) if vals.size > 0 else np.nan
            continue

        # SHOT SELECTION – average distance percentile
        if cat == "shot_selection":
            scores[cat] = float(df.loc[idx, "shot_avg_distance_pct"])
            continue

        # PLAYSTYLE – creation index percentile
        if cat == "playstyle":
            scores[cat] = float(df.loc[idx, "playstyle_creation_index_pct"])
            continue

        # OTHER CATEGORIES – mean percentile across that category’s stats
        cat_cols = [c for c in categories[cat] if c in scaled.columns]
        if not cat_cols:
            scores[cat] = np.nan
            continue
        vals = scaled.loc[idx, cat_cols].astype(float).values
        vals = vals[~np.isnan(vals)]
        scores[cat] = float(vals.mean()) if vals.size > 0 else np.nan

    return pd.Series(scores, name=player_name)


def get_league_category_profile_for_graph(df=full) -> pd.Series:
    scores = {}

    # Stature league avg
    if STATURE_COLS_GRAPH:
        scores["stature"] = float(scaled[STATURE_COLS_GRAPH].mean(axis=1).mean())
    else:
        scores["stature"] = np.nan

    # Shot selection league avg (avg distance percentile)
    scores["shot_selection"] = float(df["shot_avg_distance_pct"].mean())

    # Playstyle league avg (creation index percentile)
    scores["playstyle"] = float(df["playstyle_creation_index_pct"].mean())

    # Other categories
    for cat in RADAR_CATEGORIES:
        if cat in ["stature", "shot_selection", "playstyle"]:
            continue
        cat_cols = [c for c in categories[cat] if c in scaled.columns]
        if not cat_cols:
            scores[cat] = np.nan
            continue
        scores[cat] = float(scaled[cat_cols].mean(axis=1).mean())

    return pd.Series(scores, name="League Avg")


LEAGUE_CAT_PROFILE = get_league_category_profile_for_graph()

# build per-player category matrix (percentile-based) for clustering / league radar
rows = []
for _, r in full.iterrows():
    name = r["player_name"]
    prof = get_player_category_profile_for_graph(name)
    prof["player_name"] = name
    rows.append(prof)

CAT_PROFILE_DF = pd.DataFrame(rows).set_index("player_name")

# fill any NaNs here with league avg
for cat in RADAR_CATEGORIES:
    CAT_PROFILE_DF[cat] = CAT_PROFILE_DF[cat].fillna(LEAGUE_CAT_PROFILE.get(cat, 0.0))

# ============================================================
# 6. Clustering (using percentiles) + UMAP (internal only)
# ============================================================

X = CAT_PROFILE_DF[RADAR_CATEGORIES].values.astype(float)

N_CLUSTERS = 12
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
cluster_labels = kmeans.fit_predict(X)
CAT_PROFILE_DF["cluster_roles_k"] = cluster_labels

# attach cluster back to full
full = full.merge(
    CAT_PROFILE_DF[["cluster_roles_k"]],
    left_on="player_name",
    right_index=True,
    how="left",
    suffixes=("", "_from_cat")
)

# UMAP embedding (kept internally, unused in UI)
reducer = umap.UMAP(
    n_neighbors=25,
    min_dist=0.1,
    n_components=2,
    random_state=42,
    metric="euclidean",
)
embedding = reducer.fit_transform(X)
CAT_PROFILE_DF["umap_x"] = embedding[:, 0]
CAT_PROFILE_DF["umap_y"] = embedding[:, 1]

full = full.merge(
    CAT_PROFILE_DF[["umap_x", "umap_y"]],
    left_on="player_name",
    right_index=True,
    how="left"
)

# ============================================================
# 7. Similarity engine (NOT percentile-based)
# ============================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    cos = float(np.dot(a, b) / denom)  # [-1,1]
    return max(0.0, min(1.0, 0.5 * (cos + 1.0)))  # [0,1]


def _shot_selection_similarity(base_idx: int, other_idx: int) -> float:
    """
    Shot selection similarity based on the five frequency stats.
    Each location gets its own similarity, and we weight:
      - mid: 2/6
      - rim/paint/corner/atb: 1/6 each
    """
    if not SHOT_FREQ_COLS:
        return np.nan

    p = full.loc[base_idx, SHOT_FREQ_COLS].astype(float).values
    q = full.loc[other_idx, SHOT_FREQ_COLS].astype(float).values

    sp, sq = p.sum(), q.sum()
    if sp <= 0 or sq <= 0:
        return np.nan

    p = p / sp
    q = q / sq

    # order is exactly SHOT_FREQ_COLS: rim, paint, mid, corner, atb
    weights = np.array([1, 1, 2, 1, 1], dtype=float)
    weights = weights / weights.sum()

    per_dim_sim = 1.0 - np.abs(p - q)
    per_dim_sim = np.clip(per_dim_sim, 0.0, 1.0)

    return float(np.sum(weights * per_dim_sim))


def _playstyle_similarity(base_idx: int, other_idx: int) -> float:
    play_cols = [c for c in categories.get("playstyle", []) if c in full.columns]
    if not play_cols:
        return np.nan

    p = full.loc[base_idx, play_cols].astype(float).clip(lower=0).values
    q = full.loc[other_idx, play_cols].astype(float).clip(lower=0).values

    sp, sq = p.sum(), q.sum()
    if sp <= 0 or sq <= 0:
        return np.nan

    p = p / sp
    q = q / sq

    l1 = float(np.abs(p - q).sum())  # [0,2]
    sim = 1.0 - 0.5 * l1            # [0,1]
    return max(0.0, min(1.0, sim))


def _generic_category_similarity(cat: str, base_idx: int, other_idx: int) -> float:
    """
    Similarity for all non-shot-selection, non-playstyle categories.

    Uses 1 - mean(|p - q|) on min-max scaled stats (in [0,1]),
    so the result is also in [0,1]. Works well even if the
    category only has a single stat.
    """
    cat_cols = [c for c in categories.get(cat, []) if c in sim_scaled.columns]
    if not cat_cols:
        return np.nan

    p = sim_scaled.loc[base_idx, cat_cols].astype(float).values
    q = sim_scaled.loc[other_idx, cat_cols].astype(float).values

    diff = np.abs(p - q)
    per_dim_sim = 1.0 - diff
    per_dim_sim = np.clip(per_dim_sim, 0.0, 1.0)

    sim = float(per_dim_sim.mean())
    return sim


def compute_category_similarities(base_idx: int, other_idx: int) -> dict:
    sims = {}
    for cat in RADAR_CATEGORIES:
        if cat == "shot_selection":
            sims[cat] = _shot_selection_similarity(base_idx, other_idx)
        elif cat == "playstyle":
            sims[cat] = _playstyle_similarity(base_idx, other_idx)
        else:
            sims[cat] = _generic_category_similarity(cat, base_idx, other_idx)
    return sims


def find_similar_players_app(
    player_name: str,
    top_n: int = 10,
    min_minutes: float = 300.0,
) -> pd.DataFrame:
    """
    Similarity engine.

    - Uses *non-percentile* min-max scaled stats for all categories
      except shot_selection and playstyle.
    - shot_selection similarity is computed from the five shot frequency
      stats with weights (mid = 2/6, others = 1/6 each).
    - playstyle similarity uses distribution similarity over playtype
      frequencies.
    - Overall similarity is the mean of available category similarities.
    """
    # locate base
    mask = full["player_name"].str.lower() == player_name.lower()
    if not mask.any():
        raise ValueError(f"Player '{player_name}' not found in dataset.")
    idx_base = full.index[mask][0]

    # eligible comparison players
    df = full.copy()
    if "minutes_played" in df.columns:
        df = df[df["minutes_played"] >= min_minutes]

    df = df[df.index != idx_base]

    overall_sims = []
    cat_sims = {cat: [] for cat in RADAR_CATEGORIES}

    for idx, row in df.iterrows():
        sims = compute_category_similarities(idx_base, idx)
        # mean of categories with non-NaN similarity
        vals = [v for v in sims.values() if not np.isnan(v)]
        overall = float(np.mean(vals)) if vals else 0.0
        overall_sims.append(overall)
        for cat in RADAR_CATEGORIES:
            cat_sims[cat].append(sims.get(cat, np.nan))

    df = df.copy()
    df["similarity_overall"] = overall_sims
    for cat in RADAR_CATEGORIES:
        df[f"sim_{cat}"] = cat_sims[cat]

    df = df.sort_values("similarity_overall", ascending=False)
    return df.head(top_n)

# ============================================================
# 8. Plotting helpers (radar, shot selection, playstyle)
# ============================================================

def _get_player_index(player_name: str) -> int:
    mask = full["player_name"].str.lower() == player_name.lower()
    if not mask.any():
        raise ValueError(f"Player '{player_name}' not found.")
    return full.index[mask][0]


def plot_category_radar_streamlit(player_name: str):
    if player_name not in CAT_PROFILE_DF.index:
        st.warning(f"Player '{player_name}' not found in CAT_PROFILE_DF.")
        return

    player_vals = CAT_PROFILE_DF.loc[player_name, RADAR_CATEGORIES].values.astype(float)
    league_vals = np.array([LEAGUE_CAT_PROFILE[cat] for cat in RADAR_CATEGORIES], dtype=float)

    # convert 0–1 percentiles to 0–100
    player_vals_pct = player_vals * 100.0
    league_vals_pct = league_vals * 100.0

    N = len(RADAR_CATEGORIES)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])  # close loop

    player_plot = np.concatenate([player_vals_pct, player_vals_pct[:1]])
    league_plot = np.concatenate([league_vals_pct, league_vals_pct[:1]])

    labels = [cat.replace("_", " ").title() for cat in RADAR_CATEGORIES]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(angles, league_plot, linewidth=1, linestyle="--", label="League Avg")
    ax.fill(angles, league_plot, alpha=0.1)

    ax.plot(angles, player_plot, linewidth=2, label=player_name)
    ax.fill(angles, player_plot, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"])
    ax.set_ylim(0, 100)
    ax.set_title(f"Category Radar – {player_name} vs League (Percentiles)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    st.pyplot(fig)


def plot_shot_selection_streamlit(player_name: str):
    idx = _get_player_index(player_name)

    if not SHOT_FREQ_COLS:
        st.info("No shot selection frequency columns available.")
        return

    vals = full.loc[idx, SHOT_FREQ_COLS].astype(float).values
    total = vals.sum()
    if total <= 0:
        st.info("No shot attempts recorded for this player (sum of freqs <= 0).")
        return

    freqs_pct = (vals / total) * 100.0
    labels = [
        col.replace("shot__freq_", "").replace("atb3", "atb_3").title()
        for col in SHOT_FREQ_COLS
    ]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(labels, freqs_pct)
    ax.set_ylabel("Shot Frequency (%)")
    ax.set_title(f"Shot Selection Profile – {player_name}")
    ax.set_ylim(0, max(50, freqs_pct.max() + 5))
    plt.xticks(rotation=20)
    plt.tight_layout()
    st.pyplot(fig)


def plot_playstyle_streamlit(player_name: str):
    idx = _get_player_index(player_name)
    play_cols = [c for c in categories.get("playstyle", []) if c in full.columns]

    if not play_cols:
        st.info("No playstyle columns found.")
        return

    vals = full.loc[idx, play_cols].astype(float).clip(lower=0).values
    total = vals.sum()
    if total <= 0:
        st.info("No playstyle data for this player (sum <= 0).")
        return

    vals_pct = (vals / total) * 100.0
    labels = [
        col.replace("playstyle__", "")
           .replace("_freq", "")
           .replace("pr_", "PnR_")
           .title()
        for col in play_cols
    ]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(labels, vals_pct)
    ax.set_ylabel("Playtype Frequency (%)")
    ax.set_title(f"Playstyle Profile – {player_name}")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================
# 9. Role labels (basic) and Streamlit UI
# ============================================================

basic_role_labels = {
    0: "Heliocentric Shot-Creating Guards",
    1: "Secondary Slashing / Connector Guards",
    2: "Low-Usage Defensive Wings",
    3: "Mid-Usage Scoring Wings",
    4: "Athletic Defensive Forwards / Finishers",
    5: "High-IQ Connector Guards",
    6: "Off-Ball Shooting Wings",
    7: "Defensive Anchor Bigs",
    8: "Two-Way Point-Forward Engines",
    9: "Elite Offensive Engine Guards/Wings",
    10: "Developing Scoring Wings & Combo Guards",
    11: "Two-Way Athletic Scoring Forwards",
}

CLUSTER_ROLE_SUMMARY = {
    0: (
        "Ball-dominant pick-and-roll and isolation guards with very high creation load. "
        "They live off pull-ups and perimeter shotmaking, carry a heavy offensive burden, "
        "and usually need defensive size and spacing around them."
    ),
    1: (
        "Secondary slashing guards who can attack off the catch or second-side, run some actions, "
        "and keep the offense functional without being the main engine. Passing is solid, "
        "but shooting and defense are more mixed."
    ),
    2: (
        "Defense-first, low-usage wings who provide size, mobility, and strong on-ball/off-ball defense. "
        "Offensively they mostly space the floor, cut, and take simple shots rather than create."
    ),
    3: (
        "On-ball scoring wings with mid-tier creation load. They can get their own shot from drives and jumpers, "
        "but offer limited passing and usually below-average defense. Best as secondary scorers."
    ),
    4: (
        "Highly athletic forwards or big forwards who guard multiple positions, finish plays, and play with low usage. "
        "They bring energy, rim pressure, and defensive impact but limited shooting and self-creation."
    ),
    5: (
        "High-IQ connector guards who move the ball, space the floor, and play solid team defense. "
        "Creation load is moderate, but decision-making, shooting, and off-ball impact are their main value."
    ),
    6: (
        "Off-ball shooting wings who provide spacing and quick trigger three-point volume. "
        "They rarely initiate offense, contribute little in playmaking, and often need protection on defense."
    ),
    7: (
        "Classic rim-protecting bigs who anchor the paint on defense. Offensively they finish around the rim, "
        "set screens, and rarely shoot from outside. Creation load is almost entirely off-ball."
    ),
    8: (
        "Big point-forward engines with high creation load and strong two-way impact. "
        "They initiate offense through passing, drives, and post touches while also defending multiple positions. "
        "Shooting is often average, but versatility is elite."
    ),
    9: (
        "Elite offensive engines on the perimeter — guards or wings with top-tier shooting, playmaking, "
        "and usage. They bend defenses with pull-ups and passing, and usually need defensive cover behind them."
    ),
    10: (
        "Developing scoring wings and combo guards with flashes of creation but low established usage. "
        "They space the floor, attack closeouts, and handle simple reads while they grow into bigger roles."
    ),
    11: (
        "Athletic two-way forwards who contribute real defense, solid shooting, and efficient scoring "
        "without demanding high usage. They fit into many lineups as scalable, plug-and-play forwards."
    ),
}

if "role_label" not in full.columns:
    full["role_label"] = full["cluster_roles_k"].map(basic_role_labels)
else:
    full["role_label"] = full["role_label"].fillna(
        full["cluster_roles_k"].map(basic_role_labels)
    )

# ============================================================
# 10. Streamlit App Layout
# ============================================================

st.set_page_config(page_title="NBA Player Archetypes – FYP", layout="wide")

st.title("NBA Player Archetypes – Clustering & Similarity Explorer")

with st.sidebar:
    st.header("Player Selection")

    player_names = sorted(full["player_name"].unique().tolist())
    default_player = "Nikola Jokic" if "Nikola Jokic" in player_names else player_names[0]
    selected_player = st.selectbox("Select player", player_names, index=player_names.index(default_player))

    min_minutes = st.slider("Minimum minutes for similar players", 0, 3000, 300, step=100)
    top_n_similar = st.slider("Top N similar players", 5, 30, 10, step=1)

tab_player, tab_clusters, tab_data = st.tabs(["Player Explorer", "Cluster View", "Raw Data"])

# ------------------------------------------------------------
# Player Explorer Tab
# ------------------------------------------------------------
with tab_player:
    st.subheader(f"Player Report: {selected_player}")

    idx_sel = _get_player_index(selected_player)
    row_sel = full.loc[idx_sel]

    col_info, col_radar = st.columns([1, 2])

    with col_info:
        st.markdown("#### Basic Info")
        st.write(f"**Minutes played**: {row_sel.get('minutes_played', np.nan)}")
        st.write(f"**Cluster ID**: {row_sel.get('cluster_roles_k', np.nan)}")
        st.write(f"**Role label**: {row_sel.get('role_label', 'Unknown')}")

        # League comparison table (percentiles)
        st.markdown("#### Category Percentiles vs League")
        player_prof = CAT_PROFILE_DF.loc[selected_player, RADAR_CATEGORIES]
        league_prof = pd.Series({cat: LEAGUE_CAT_PROFILE[cat] for cat in RADAR_CATEGORIES})

        comp_df = pd.DataFrame({
            "Category": [c.replace("_", " ").title() for c in RADAR_CATEGORIES],
            f"{selected_player} (Pct)": (player_prof.values * 100).round(1),
            "League Avg (Pct)": (league_prof.values * 100).round(1),
        })

        st.dataframe(
            comp_df,
            use_container_width=True,
            column_config={
                "Category": st.column_config.Column(
                    label="Category",
                    width=220,
                ),
                f"{selected_player} (Pct)": st.column_config.NumberColumn(
                    label=f"{selected_player} (Pct)",
                    format="%.1f"
                ),
                "League Avg (Pct)": st.column_config.NumberColumn(
                    label="League Avg (Pct)",
                    format="%.1f"
                ),
            }
        )

    with col_radar:
        st.markdown("#### Category Radar")
        plot_category_radar_streamlit(selected_player)

    col_shot, col_play = st.columns(2)
    with col_shot:
        st.markdown("#### Shot Selection Profile")
        plot_shot_selection_streamlit(selected_player)
    with col_play:
        st.markdown("#### Playstyle Profile")
        plot_playstyle_streamlit(selected_player)

    # Similar players
    st.markdown("### Similar Players (Category Similarity – % Match)")
    try:
        sims = find_similar_players_app(
            selected_player,
            top_n=top_n_similar,
            min_minutes=min_minutes,
        )

        base_cols = ["player_name", "position", "minutes_played", "role_label", "similarity_overall"]
        cat_cols = [f"sim_{cat}" for cat in RADAR_CATEGORIES]
        display_cols = [c for c in base_cols + cat_cols if c in sims.columns]

        sims_display = sims[display_cols].copy()

        # Convert similarity values to percentages
        pct_cols = [c for c in display_cols if c.startswith("sim_") or c == "similarity_overall"]
        for c in pct_cols:
            sims_display[c] = (sims_display[c] * 100.0).round(1)

        # Rename columns for clarity
        rename_map = {"similarity_overall": "similarity_overall_pct"}
        for cat in RADAR_CATEGORIES:
            col = f"sim_{cat}"
            if col in sims_display.columns:
                pretty = cat.replace("_", " ").title()
                rename_map[col] = f"{pretty} Sim (%)"
        sims_display = sims_display.rename(columns=rename_map)

        st.dataframe(sims_display.reset_index(drop=True), use_container_width=True)
    except ValueError as e:
        st.error(str(e))

# ------------------------------------------------------------
# Cluster View Tab
# ------------------------------------------------------------
with tab_clusters:
    st.subheader("Cluster Browser")

    # Cluster counts
    st.markdown("### Cluster Counts")

    counts = full.groupby("cluster_roles_k")["player_name"].count().rename("count")
    counts = counts.to_frame()
    counts["role_label"] = counts.index.map(basic_role_labels)
    st.dataframe(counts, use_container_width=True)

    st.markdown("---")
    st.markdown("### Inspect Cluster")

    cluster_options = sorted(full["cluster_roles_k"].dropna().unique().astype(int).tolist())

    # Default to selected player's cluster
    sel_cluster = int(row_sel.get("cluster_roles_k", cluster_options[0]))
    if sel_cluster not in cluster_options:
        sel_cluster = cluster_options[0]

    selected_cluster = st.selectbox(
        "Select a cluster:",
        options=cluster_options,
        index=cluster_options.index(sel_cluster),
        format_func=lambda c: f"{c} – {basic_role_labels.get(c, 'Cluster')}"
    )

    cluster_players = full[full["cluster_roles_k"] == selected_cluster].copy()

    # Players in cluster
    st.markdown(f"### Players in Cluster {selected_cluster} – {basic_role_labels.get(selected_cluster, '')}")

    display_cols = ["player_name", "position", "minutes_played", "role_label"]
    display_cols = [c for c in display_cols if c in cluster_players.columns]

    st.dataframe(
        cluster_players[display_cols].sort_values("minutes_played", ascending=False),
        use_container_width=True
    )

    st.markdown("---")
    st.markdown(f"### Cluster Summary Stats")

    # Percentile-based category profile for players in this cluster
    cluster_profile = CAT_PROFILE_DF.loc[
        cluster_players["player_name"].tolist(),
        RADAR_CATEGORIES
    ].mean().sort_values(ascending=False)

    summary_df = pd.DataFrame({
        "Category": [c.replace("_", " ").title() for c in cluster_profile.index],
        "Average Percentile (0–100)": (cluster_profile.values * 100).round(1),
    })

    st.dataframe(summary_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### Cluster Archetype Description")

    def generate_cluster_description(cat_profile: pd.Series) -> str:
        sorted_cats = cat_profile.sort_values(ascending=False)
        top_strengths = sorted_cats.head(3)
        bottom_weaknesses = sorted_cats.tail(2)

        strength_lines = [
            f"- **{cat.replace('_', ' ').title()}** ({val*100:.1f} percentile)"
            for cat, val in top_strengths.items()
        ]
        weakness_lines = [
            f"- **{cat.replace('_', ' ').title()}** ({val*100:.1f} percentile)"
            for cat, val in bottom_weaknesses.items()
        ]

        role = basic_role_labels.get(selected_cluster, f"Cluster {selected_cluster}")
        static_summary = CLUSTER_ROLE_SUMMARY.get(
            selected_cluster,
            "Players in this cluster share similar tendencies in size, creation load, defense and shooting."
        )

        text = f"""
**Role Label:** *{role}*

**Conceptual Archetype:**  
{static_summary}

**Statistical Strengths (relative to league):**
{chr(10).join(strength_lines)}

**Statistical Weaknesses (relative to league):**
{chr(10).join(weakness_lines)}
"""
        return text

    st.markdown(generate_cluster_description(cluster_profile))


# ------------------------------------------------------------
# Raw Data Tab
# ------------------------------------------------------------
with tab_data:
    st.subheader("Raw Dataset Preview")
    st.dataframe(full.head(100), use_container_width=True)
