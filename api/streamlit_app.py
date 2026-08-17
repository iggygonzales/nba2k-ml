"""
NBA 2K ML — Streamlit Dashboard
--------------------------------
Run from project root:
    streamlit run api/streamlit_app.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from streamlit_searchbox import st_searchbox

API = st.secrets.get("API_URL", "http://localhost:8000")
# st.sidebar.write(f"API: {API}")
# old API = "http://18.222.150.138:8000"
# API = "http://localhost:8000"

st.set_page_config(
    page_title="NBA2K27 Rating Predictor",
    page_icon="🏀",
    layout="wide"
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.card {
    background: #13151c;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.ovr-superstar { color: #f0c040; font-size: 3rem; font-weight: 800; line-height: 1; }
.ovr-star      { color: #c084fc; font-size: 3rem; font-weight: 800; line-height: 1; }
.ovr-elite     { color: #40a0f0; font-size: 3rem; font-weight: 800; line-height: 1; }
.ovr-starter   { color: #34d399; font-size: 3rem; font-weight: 800; line-height: 1; }
.ovr-rotation  { color: #f08040; font-size: 3rem; font-weight: 800; line-height: 1; }
.ovr-bench     { color: #606080; font-size: 3rem; font-weight: 800; line-height: 1; }
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    margin-top: 4px;
}
.delta-pos { color: #40d080; font-weight: 700; font-size: 1rem; }
.delta-neg { color: #f04060; font-weight: 700; font-size: 1rem; }
.delta-neu { color: #888;    font-weight: 700; font-size: 1rem; }
div[data-testid="stHorizontalBlock"] button {
    border: 1px solid #2a2d3e !important;
    background: #13151c !important;
    color: #c0c4d8 !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    transition: all 0.15s !important;
}
div[data-testid="stHorizontalBlock"] button:hover {
    border-color: #4a6fa5 !important;
    color: #ffffff !important;
    background: #1a1e2e !important;
}
.legend-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1rem; }
.legend-item { display: flex; align-items: center; gap: 5px; font-size: 0.75rem; color: #aaa; }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
</style>
""", unsafe_allow_html=True)


# ── Season / game mappings ────────────────────────────────────────────────────

SEASON_OPTIONS = [
    ("2025-26", "2K27 Prediction"),
    ("2024-25", "2K26"),
    ("2023-24", "2K25"),
    ("2022-23", "2K24"),
    ("2021-22", "2K23"),
    ("2020-21", "2K22"),
    ("2019-20", "2K21"),
    ("2018-19", "2K20"),
]

SEASON_DISPLAY = {s: f"{s}  ({g})" for s, g in SEASON_OPTIONS}
DISPLAY_TO_KEY = {v: k for k, v in SEASON_DISPLAY.items()}
SEASON_TO_GAME = {s: g for s, g in SEASON_OPTIONS}


# ── Helpers ───────────────────────────────────────────────────────────────────

TIERS = [
    (95, "SUPERSTAR", "#f0c040", "ovr-superstar"),
    (90, "STAR",      "#c084fc", "ovr-star"),
    (85, "ELITE",     "#40a0f0", "ovr-elite"),
    (80, "STARTER",   "#34d399", "ovr-starter"),
    (75, "ROTATION",  "#f08040", "ovr-rotation"),
    (0,  "BENCH",     "#606080", "ovr-bench"),
]

def get_tier(ovr):
    if ovr is None:
        return "BENCH", "#606080", "ovr-bench"
    for threshold, label, color, cls in TIERS:
        if ovr >= threshold:
            return label, color, cls
    return "BENCH", "#606080", "ovr-bench"

def ovr_html(ovr, label="OVR"):
    if ovr is None:
        return (
            f'<div class="card">'
            f'<div style="font-size:0.75rem;color:#888;margin-bottom:4px">{label}</div>'
            f'<span style="color:#555;font-size:2rem">N/A</span>'
            f'</div>'
        )
    tier_label, color, cls = get_tier(ovr)
    badge_html = (
        f'<span class="badge" style="background:{color}22;color:{color}">'
        f'{tier_label}</span>'
    )
    return (
        f'<div class="card">'
        f'<div style="font-size:0.75rem;color:#888;margin-bottom:4px">{label}</div>'
        f'<span class="{cls}">{ovr}</span>'
        f'<div>{badge_html}</div>'
        f'</div>'
    )

def delta_html(val):
    if val is None or val == 0:
        return '<span class="delta-neu">→ No change from 2K26</span>'
    arrow = "▲" if val > 0 else "▼"
    cls   = "delta-pos" if val > 0 else "delta-neg"
    return f'<span class="{cls}">{arrow} {val:+.1f} from 2K26</span>'

def format_game_version(gv):
    return gv.replace("nba-", "").upper()

def color_delta_style(val):
    try:
        v = float(val)
    except (ValueError, TypeError):
        return "color: #888"
    if v == 0: return "color: #888"
    return "color: #40d080" if v > 0 else "color: #f04060"

def num_col(fmt="%.1f"):
    return st.column_config.NumberColumn(format=fmt)

def tier_legend_html():
    items = "".join([
        f'<div class="legend-item">'
        f'<span class="legend-dot" style="background:{color}"></span>'
        f'{label} ({threshold}+)'
        f'</div>'
        for threshold, label, color, _ in TIERS[:-1]
    ] + [
        '<div class="legend-item">'
        '<span class="legend-dot" style="background:#606080"></span>'
        'BENCH (&lt;75)'
        '</div>'
    ])
    return f'<div class="legend-row">{items}</div>'


# ── Cached API calls ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3601)
def get_leaderboard():
    try:
        top10    = requests.get(f"{API}/ask", params={"q": "Top 10 highest rated players in nba-2k26 show player_name ovr_rating pts reb ast"}, timeout=15)
        improved = requests.get(f"{API}/ask", params={"q": "Top 10 players with biggest rating increase in nba-2k26 show player_name ovr_rating ovr_prev ovr_delta"}, timeout=15)
        declined = requests.get(f"{API}/ask", params={"q": "Top 10 players with biggest rating decrease in nba-2k26 show player_name ovr_rating ovr_prev ovr_delta"}, timeout=15)
        return top10.json(), improved.json(), declined.json()
    except Exception:
        return None, None, None


@st.cache_data(ttl=3601)
def get_2k27_predictions():
    try:
        # Fetch a wider pool so the true 2K27 top 10 can surface
        res   = requests.get(f"{API}/ask", params={"q": "Top 30 players by ovr_rating in nba-2k26 show player_name ovr_rating"}, timeout=15).json()
        names = [r.get("player_name") for r in res.get("data", []) if r.get("player_name")]
        rows  = []
        for name in names:
            p = requests.get(f"{API}/predict/2k27/{name}", timeout=10).json()
            if "detail" not in p:
                last = p.get("last_known_ovr") or 0
                pred = p.get("rounded_ovr") or 0
                rows.append({
                    "Player":         name,
                    "2K26":           last,
                    "Predicted 2K27": pred,
                    "Δ":              round(pred - last, 1),
                    "PTS": round(float(str(p.get("current_stats", {}).get("pts") or 0).strip()), 1),
                    "AGE": int(float(p.get("current_stats", {}).get("age") or 0)),
                     })
        rows.sort(key=lambda r: r["Predicted 2K27"], reverse=True)
        return rows[:10] if rows else None
    except Exception as e:
        st.write("Error:", e)  # debug
        return None

@st.cache_data(ttl=300)
def get_2k27_movers():
    try:
        res = requests.get(f"{API}/leaderboard/2k27-movers", timeout=15).json()
        return res.get("risers"), res.get("decliners")
    except Exception:
        return None, None

@st.cache_data(ttl=300)
def get_player_history(name):
    return requests.get(f"{API}/player/{name}/history", timeout=10).json()

@st.cache_data(ttl=300)
def get_player_rating(name, season):
    return requests.get(f"{API}/player/{name}", params={"season": season}, timeout=10).json()

@st.cache_data(ttl=300)
def get_2k27_prediction(name):
    return requests.get(f"{API}/predict/2k27/{name}", timeout=10).json()

@st.cache_data(ttl=300)
def search_players(query):
    return requests.get(f"{API}/search/{query}", timeout=10).json()


# ── Page header ───────────────────────────────────────────────────────────────

st.title("🏀🎮 NBA 2K27 Rating Predictor")
st.caption("Predict 2K27 ratings using real NBA stats + Machine Learning · Model MAE: ±1.23 OVR · R² = 0.933")

tab1, tab2, tab3 = st.tabs(["Player Lookup", "Leaderboard", "Ask Anything"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Player Lookup
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:

    featured_default = None
    if "featured_player" in st.session_state:
        featured_default = st.session_state.pop("featured_player")
        st.session_state["searchbox_key"] = featured_default

    if "searchbox_key" not in st.session_state:
        st.session_state["searchbox_key"] = "default"

    search_col, season_col = st.columns([4, 1])

    def search_players_live(query: str):
        if not query or len(query) < 2:
            return []
        try:
            res = requests.get(f"{API}/search/{query}", timeout=5).json()
            return res.get("results", [])
        except Exception:
            return []

    with search_col:
        selected = st_searchbox(
            search_players_live,
            placeholder="🔍  Search any NBA player — e.g. LeBron James, Nikola Jokic...",
            key=f"player_searchbox_{st.session_state['searchbox_key']}",
            default_use_searchterm=False,
            default=featured_default,
        )

    with season_col:
        season_display = st.selectbox(
            "Season",
            [SEASON_DISPLAY[s] for s, _ in SEASON_OPTIONS],
            label_visibility="collapsed"
        )
        season_key = DISPLAY_TO_KEY[season_display]
        game_label = SEASON_TO_GAME[season_key]
        is_predict_season = season_key == "2025-26"

    # Track selected player across season changes
    if selected:
        st.session_state["selected_player"] = selected
        st.session_state["selected_season"] = season_key
    elif not selected and "selected_player" in st.session_state:
        # Restore last player when only the season dropdown changes
        selected = st.session_state["selected_player"]
        st.divider()

    if selected:
        try:
            pred_res    = get_2k27_prediction(selected)
            history_res = get_player_history(selected)
            if not is_predict_season:
                player_res = get_player_rating(selected, season_key)
            else:
                player_res = None
        except Exception:
            st.error("Could not load player data — API issue")
            pred_res    = {"detail": "error"}
            history_res = {}
            player_res  = None

        col1, col2 = st.columns(2)

        # ── Left column: current season rating (or info if 2025-26) ──────────
        with col1:
            if is_predict_season:
                st.markdown(
                    '<div class="card" style="border-color:#40a0f030">'
                    '<div style="font-size:0.75rem;color:#40a0f0;margin-bottom:8px">2025-26 SEASON</div>'
                    '<p style="color:#aaa;font-size:0.9rem;margin:0">The 2025-26 season is the basis '
                    'for 2K27 predictions. No 2K rating exists yet — see this player\'s predicted rating for this year or for previous seasons</p>'
                    '</div>',
                    unsafe_allow_html=True
                )
            elif player_res and "detail" not in player_res:
                actual    = player_res.get("actual_ovr")
                predicted = player_res.get("predicted_ovr")

                st.markdown(
                    ovr_html(int(actual) if actual else None,
                             f"{game_label} Rating — {season_key}"),
                    unsafe_allow_html=True
                )

                if predicted and actual:
                    diff = round(actual - predicted, 1)
                    if abs(diff) < 1.5:
                        verdict, vc = "Fairly rated by stats", "#40d080"
                    elif diff > 0:
                        verdict, vc = f"Overrated ~{abs(diff)} OVR vs stats", "#f04060"
                    else:
                        verdict, vc = f"Underrated ~{abs(diff)} OVR vs stats", "#40a0f0"
                    st.markdown(
                        f'<div style="background:{vc}11;border-left:3px solid {vc};'
                        f'padding:0.6rem 1rem;border-radius:0 8px 8px 0;margin:0.5rem 0 1rem">'
                        f'<strong style="color:{vc}">{verdict}</strong>'
                        f'<span style="font-size:0.8rem;color:#aaa;margin-left:8px">'
                        f'For this season the model predicted a rating of {predicted}</span></div>',
                        unsafe_allow_html=True
                    )

                stats = player_res.get("stats", {})
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.caption(f"Season stats — {season_key}")
                st.dataframe(
                    pd.DataFrame([stats]).rename(columns={
                        "pts": "PTS", "reb": "REB", "ast": "AST",
                        "stl": "STL", "blk": "BLK", "usg_pct": "USG%"
                    }),
                    column_config={
                        "PTS": num_col(), "REB": num_col(), "AST": num_col(),
                        "STL": num_col(), "BLK": num_col(), "USG%": num_col("%.3f"),
                    },
                    hide_index=True, use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning(f"No {game_label} rating found for {selected}")

        # ── Right column: 2K27 prediction ────────────────────────────────────
        with col2:
            if "detail" not in pred_res:
                last_ovr  = pred_res.get("last_known_ovr")
                pred_ovr  = pred_res.get("predicted_2k27_ovr")
                rounded   = pred_res.get("rounded_ovr")
                delta_val = round(pred_ovr - last_ovr, 1) if pred_ovr and last_ovr else None

                st.markdown(ovr_html(rounded, "Predicted 2K27 Rating"), unsafe_allow_html=True)
                st.markdown(delta_html(delta_val), unsafe_allow_html=True)

                cs = pred_res.get("current_stats", {})
                st.markdown('<div class="card" style="margin-top:0.75rem">', unsafe_allow_html=True)
                st.caption("2025-26 stats (basis for prediction)")
                st.dataframe(
                    pd.DataFrame([cs]).rename(columns={
                        "pts": "PTS", "reb": "REB", "ast": "AST",
                        "age": "AGE", "gp": "GP"
                    }),
                    column_config={
                        "PTS": num_col(), "REB": num_col(),
                        "AST": num_col(), "AGE": num_col(),
                        "GP":  num_col("%d"),
                    },
                    hide_index=True, use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
                st.caption(pred_res.get("note", ""))
            else:
                st.info(f"No 2025-26 stats found for {selected}")

        # ── Career trajectory chart ───────────────────────────────────────────
        if "history" in history_res:
            df = pd.DataFrame(history_res["history"])
            df["game_label"] = df["game_version"].apply(format_game_version)

            actual_df = df[df["ovr_rating"].notna()]
            fig = go.Figure()

            # Shade predicted region
            if len(actual_df) > 0 and "detail" not in pred_res:
                last_label = actual_df.iloc[-1]["game_label"]
                fig.add_vrect(
                    x0=last_label, x1="2K27",
                    fillcolor="#40a0f0", opacity=0.05,
                    layer="below", line_width=0,
                    annotation_text="Predicted",
                    annotation_position="top left",
                    annotation_font=dict(color="#40a0f0", size=11)
                )

            # Actual OVR line
            fig.add_trace(go.Scatter(
                x=actual_df["game_label"],
                y=actual_df["ovr_rating"],
                mode="lines+markers+text",
                name="Actual OVR",
                line=dict(color="#f0c040", width=3),
                marker=dict(size=9, color="#f0c040"),
                text=actual_df["ovr_rating"].astype(int).astype(str),
                textposition="top center",
                textfont=dict(color="#f0c040", size=11),
                hovertemplate="<b>%{x}</b><br>OVR: %{y}<extra></extra>"
            ))

            # Predicted 2K27 line
            if "detail" not in pred_res and len(actual_df) > 0:
                last_row     = actual_df.iloc[-1]
                pred_ovr_val = pred_res.get("rounded_ovr")
                fig.add_trace(go.Scatter(
                    x=[last_row["game_label"], "2K27"],
                    y=[last_row["ovr_rating"], pred_ovr_val],
                    mode="lines+markers+text",
                    name="Predicted 2K27",
                    line=dict(color="#40a0f0", width=2, dash="dash"),
                    marker=dict(size=12, color="#40a0f0", symbol="star"),
                    text=["", str(pred_ovr_val)],
                    textposition="top center",
                    textfont=dict(color="#40a0f0", size=12),
                    hovertemplate="<b>%{x}</b><br>Predicted OVR: %{y}<extra></extra>"
                ))

            fig.update_layout(
                title=dict(
                    text=f"{selected} — Career Rating Trajectory",
                    font=dict(size=16, color="#e8e8f0")
                ),
                xaxis_title="Game Version",
                yaxis_title="OVR Rating",
                yaxis=dict(range=[65, 102], gridcolor="#1e2130"),
                xaxis=dict(gridcolor="#1e2130"),
                plot_bgcolor="#0d0f14",
                paper_bgcolor="#0d0f14",
                font=dict(color="#e8e8f0"),
                legend=dict(
                    bgcolor="#13151c", bordercolor="#2a2a38",
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1
                ),
                height=420,
                margin=dict(t=60, b=40),
            )

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Full career history"):
                display_cols = ["game_label", "season", "ovr_rating", "pts",
                            "reb", "ast", "age", "gp", "ovr_delta", "split"]
                display_df = df[display_cols].copy()
                
                for col in ["ovr_rating", "pts", "reb", "ast", "age", "ovr_delta"]:
                    display_df[col] = pd.to_numeric(display_df[col], errors="coerce")
                
                # Format as strings so Streamlit can't add trailing zeros
                display_df["ovr_rating"] = display_df["ovr_rating"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "None")
                display_df["pts"]        = display_df["pts"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "None")
                display_df["reb"]        = display_df["reb"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "None")
                display_df["ast"]        = display_df["ast"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "None")
                display_df["age"]        = display_df["age"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "None")
                display_df["ovr_delta"]  = display_df["ovr_delta"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "None")

                display_df = display_df.rename(columns={
                    "game_label": "Game", "season": "Season",
                    "ovr_rating": "OVR", "pts": "PTS", "reb": "REB",
                    "ast": "AST", "age": "AGE", "gp": "GP",
                    "ovr_delta": "OVR Δ", "split": "Split"
                })
                st.dataframe(
                    display_df.style.map(color_delta_style, subset=["OVR Δ"]),
                    column_config={
                        "GP": st.column_config.NumberColumn(format="%d"),
                    },
                    hide_index=True, use_container_width=True
                )

    else:
        st.markdown("#### Featured players — click to explore")
        featured = [
            "LeBron James", "Nikola Jokic", "Stephen Curry",
            "Giannis Antetokounmpo", "Luka Doncic", "Shai Gilgeous-Alexander"
        ]
        fcols = st.columns(3)
        for i, name in enumerate(featured):
            if fcols[i % 3].button(f"🏀 {name}", key=f"f{i}", use_container_width=True):
                st.session_state["featured_player"] = name
                st.rerun()

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown("**Tier guide:**")
        st.markdown(tier_legend_html(), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Leaderboard
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:

    top10, improved, declined = get_leaderboard()
    pred_rows = get_2k27_predictions()
    up_pred, dn_pred = get_2k27_movers()

    st.markdown("**Tier guide:**")
    st.markdown(tier_legend_html(), unsafe_allow_html=True)

    if top10 is None:
        st.warning("⚠️ Leaderboard temporarily unavailable — API issue.")
    else:
        # ── Section 1: Current ratings vs predicted ───────────────────────────
        lcol1, lcol2 = st.columns(2)

        with lcol1:
            st.markdown("#### 🏆 2K26 Top 10")
            if top10.get("data"):
                df_top = pd.DataFrame(top10["data"])
                df_top.columns = [c.replace("_", " ").title() for c in df_top.columns]
                for c in ["Ovr Rating", "Pts", "Reb", "Ast"]:
                    if c in df_top.columns:
                        df_top[c] = pd.to_numeric(df_top[c], errors="coerce").round(1)
                df_top.index = range(1, len(df_top) + 1)
                st.dataframe(
                    df_top,
                    column_config={
                        "Ovr Rating": num_col(), "Pts": num_col(),
                        "Reb": num_col(), "Ast": num_col(),
                    },
                    use_container_width=True
                )

        with lcol2:
            st.markdown("#### 🔮 Predicted 2K27 Top 10")
            if pred_rows:
                df_p = pd.DataFrame(pred_rows)
                for c in ["2K26", "Predicted 2K27", "Δ", "PTS", "AGE"]:
                    if c in df_p.columns:
                        df_p[c] = pd.to_numeric(df_p[c], errors="coerce").round(1)

                df_p.index = range(1, len(df_p) + 1)
                df_p["PTS"] = df_p["PTS"].apply(lambda x: f"{float(x):.1f}")
                df_p["AGE"] = df_p["AGE"].astype(int)
                df_p["2K26"] = df_p["2K26"].astype(int)
                df_p["Predicted 2K27"] = df_p["Predicted 2K27"].astype(int)

                st.dataframe(
                    df_p.style.map(color_delta_style, subset=["Δ"]),
                    column_config={
                        "2K26":           st.column_config.NumberColumn(format="%d"),
                        "Predicted 2K27": st.column_config.NumberColumn(format="%d"),
                        "Δ":              st.column_config.NumberColumn(format="%.1f"),
                        "PTS":            st.column_config.NumberColumn(format="%.1f"),
                        "AGE":            st.column_config.NumberColumn(format="%d"),
                    },
                    use_container_width=True
                )
            else:
                st.info("Loading 2K27 predictions...")

        # ── Section 2: Biggest movers in 2K26 ────────────────────────────────
        st.divider()
        st.subheader("Biggest movers in 2K26")
        mcol1, mcol2 = st.columns(2)

        with mcol1:
            st.markdown("#### 📈 Most improved")
            if improved and improved.get("data"):
                df_up = pd.DataFrame(improved["data"])
                df_up.columns = [c.replace("_", " ").title() for c in df_up.columns]
                for c in ["Ovr Rating", "Ovr Prev", "Ovr Delta"]:
                    if c in df_up.columns:
                        df_up[c] = pd.to_numeric(df_up[c], errors="coerce").round(1)
                st.dataframe(
                    df_up.style.map(color_delta_style, subset=["Ovr Delta"]),
                    column_config={
                        "Ovr Rating": num_col(), "Ovr Prev": num_col(), "Ovr Delta": num_col(),
                    },
                    hide_index=True, use_container_width=True
                )

        with mcol2:
            st.markdown("#### 📉 Biggest declines")
            if declined and declined.get("data"):
                df_dn = pd.DataFrame(declined["data"])
                df_dn.columns = [c.replace("_", " ").title() for c in df_dn.columns]
                for c in ["Ovr Rating", "Ovr Prev", "Ovr Delta"]:
                    if c in df_dn.columns:
                        df_dn[c] = pd.to_numeric(df_dn[c], errors="coerce").round(1)
                st.dataframe(
                    df_dn.style.map(color_delta_style, subset=["Ovr Delta"]),
                    column_config={
                        "Ovr Rating": num_col(), "Ovr Prev": num_col(), "Ovr Delta": num_col(),
                    },
                    hide_index=True, use_container_width=True
                )

        # ── Section 3: Predicted 2K27 movers ─────────────────────────────────
        st.divider()
        st.subheader("Predicted 2K27 Movers")
        st.caption("Based on 2025-26 season stats — who will be rising and who keeps falling?")

        pcol1, pcol2 = st.columns(2)

        with pcol1:
            st.markdown("#### 🚀 Predicted Biggest Risers for 2K27")
            if up_pred:
                df_up_pred = pd.DataFrame(up_pred)
                for c in ["2K26", "Predicted 2K27", "Δ", "PTS", "AGE"]:
                    if c in df_up_pred.columns:
                        df_up_pred[c] = pd.to_numeric(df_up_pred[c], errors="coerce").round(1)
                df_up_pred = df_up_pred[df_up_pred["Δ"] > 0].sort_values("Δ", ascending=False)
                df_up_pred = df_up_pred.rename(columns={
                    "Player": "Player Name",
                    "2K26": "Ovr Prev",
                    "Predicted 2K27": "Ovr Rating",
                    "Δ": "Ovr Delta"
                })

                df_up_pred["Ovr Delta"] = df_up_pred["Ovr Delta"].apply(lambda x: f"{float(x):.1f}")
                st.dataframe(
                    df_up_pred[["Player Name", "Ovr Rating", "Ovr Prev", "Ovr Delta"]].style.map(
                        color_delta_style, subset=["Ovr Delta"]
                    ),
                    column_config={
                        "Ovr Rating": num_col(), "Ovr Prev": num_col(), "Ovr Delta": num_col(),
                    },
                    hide_index=True, use_container_width=True
                )
            else:
                st.info("Loading predictions...")

        with pcol2:
            st.markdown("#### 📉 Predicted Biggest Decliners for 2K27")
            if dn_pred:
                df_dn_pred = pd.DataFrame(dn_pred)
                for c in ["2K26", "Predicted 2K27", "Δ", "PTS", "AGE"]:
                    if c in df_dn_pred.columns:
                        df_dn_pred[c] = pd.to_numeric(df_dn_pred[c], errors="coerce").round(1)
                df_dn_pred = df_dn_pred[df_dn_pred["Δ"] < 0].sort_values("Δ", ascending=True)
                df_dn_pred = df_dn_pred.rename(columns={
                    "Player": "Player Name",
                    "2K26": "Ovr Prev",
                    "Predicted 2K27": "Ovr Rating",
                    "Δ": "Ovr Delta"
                })
                df_dn_pred["Ovr Delta"] = df_dn_pred["Ovr Delta"].apply(lambda x: f"{float(x):.1f}")
                st.dataframe(
                    df_dn_pred[["Player Name", "Ovr Rating", "Ovr Prev", "Ovr Delta"]].style.map(
                        color_delta_style, subset=["Ovr Delta"]
                    ),
                    column_config={
                        "Ovr Rating": num_col(), "Ovr Prev": num_col(), "Ovr Delta": num_col(),
                    },
                    hide_index=True, use_container_width=True
                )
            else:
                st.info("Loading predictions...")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Ask Anything
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Ask anything about the database")
    st.caption("Powered by Claude — converts your question to SQL and queries the live database")

    QUICK = [
        "Who had the biggest rating jump in 2K26?",
        "What team has the highest average rating in 2024-25?",
        "What are the predicted 2K27 ratings for the top 10 players?",
        "Who is the most underrated player in 2K26?",
        "Which players declined the most from 2K25 to 2K26?",
        "Who has the highest career average OVR rating?",
    ]

    st.markdown("**Try a quick question:**")
    qcols = st.columns(3)
    for i, q in enumerate(QUICK):
        if qcols[i % 3].button(q, key=f"q{i}", use_container_width=True):
            st.session_state["ask_question"] = q
            st.rerun()

    st.markdown('<br>', unsafe_allow_html=True)

    question = st.text_input(
        "Ask a question",
        value=st.session_state.get("ask_question", ""),
        placeholder="e.g. Who is the most underrated player in 2K26?",
        label_visibility="collapsed"
    )

    if st.button("Ask →", type="primary") and question:
        with st.spinner("Generating SQL and querying database..."):
            try:
                res = requests.get(f"{API}/ask", params={"q": question}, timeout=30).json()

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**Answer:**")
                st.markdown(res.get("answer", "No answer"))
                st.markdown('</div>', unsafe_allow_html=True)

                with st.expander("Generated SQL"):
                    st.code(res.get("sql", ""), language="sql")

                if res.get("data"):
                    with st.expander("Raw data", expanded=True):
                        raw_df = pd.DataFrame(res["data"])
                        numeric_cols = raw_df.select_dtypes(include="number").columns.tolist()
                        for c in numeric_cols:
                            raw_df[c] = pd.to_numeric(raw_df[c], errors="coerce").round(1)
                        col_config = {c: num_col() for c in numeric_cols}
                        st.dataframe(
                            raw_df,
                            column_config=col_config,
                            hide_index=True,
                            use_container_width=True
                        )
            except Exception as e:
                st.error(f"Something went wrong: {e}")