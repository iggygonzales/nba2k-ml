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

API = "http://localhost:8000"

st.set_page_config(
    page_title="NBA 2K ML",
    page_icon="🏀",
    layout="wide"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def rating_tier(ovr):
    if ovr is None:
        return "", ""
    if ovr >= 95:   return "SUPERSTAR", "#f0c040"
    if ovr >= 90:   return "STAR",      "#40d080"
    if ovr >= 85:   return "ELITE",     "#40a0f0"
    if ovr >= 80:   return "STARTER",   "#a060f0"
    if ovr >= 75:   return "ROTATION",  "#f08040"
    return "BENCH", "#606080"

def delta_color(val):
    if val is None or val == 0: return "grey"
    return "normal" if val > 0 else "inverse"

def format_game_version(gv):
    return gv.replace("nba-", "").upper()

def color_delta_style(val):
    if pd.isna(val) or val == 0: return "color: grey"
    return "color: #40d080" if val > 0 else "color: #f04060"

def tier_badge(ovr):
    tier, color = rating_tier(ovr)
    if not tier:
        return ""
    return (
        f'<span style="background:{color}22;color:{color};'
        f'padding:3px 10px;border-radius:4px;font-size:0.75rem;'
        f'font-weight:700;letter-spacing:0.1em">{tier}</span>'
    )

# Standard column configs for reuse
ONE_DP  = {"format": "%.1f"}
INT_FMT = {"format": "%d"}

def num_col(fmt="%.1f"):
    return st.column_config.NumberColumn(format=fmt)

# ── Cached API calls ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_leaderboard():
    top10 = requests.get(f"{API}/ask", params={
        "q": "Top 10 highest rated players in nba-2k26 show player_name ovr_rating pts reb ast"
    }).json()
    improved = requests.get(f"{API}/ask", params={
        "q": "Top 10 players with biggest rating increase in nba-2k26 show player_name ovr_rating ovr_prev ovr_delta"
    }).json()
    declined = requests.get(f"{API}/ask", params={
        "q": "Top 10 players with biggest rating decrease in nba-2k26 show player_name ovr_rating ovr_prev ovr_delta"
    }).json()
    return top10, improved, declined


@st.cache_data(ttl=3600)
def get_2k27_predictions():
    res = requests.get(f"{API}/ask", params={
        "q": "Top 10 predicted 2K27 ratings based on 2025-26 stats show player_name ovr_prev pts age"
    }).json()
    names = [r.get("player_name") for r in res.get("data", []) if r.get("player_name")]
    pred_rows = []
    for name in names:
        p = requests.get(f"{API}/predict/2k27/{name}").json()
        if "detail" not in p:
            pred_rows.append({
                "Player":         name,
                "2K26":           p.get("last_known_ovr"),
                "Predicted 2K27": p.get("rounded_ovr"),
                "Δ":              round((p.get("rounded_ovr") or 0) - (p.get("last_known_ovr") or 0), 1),
                "PTS":            round(float(p.get("current_stats", {}).get("pts") or 0), 1),
                "AGE":            round(float(p.get("current_stats", {}).get("age") or 0), 1),
            })
    return pred_rows


@st.cache_data(ttl=300)
def get_player_history(name):
    return requests.get(f"{API}/player/{name}/history").json()


@st.cache_data(ttl=300)
def get_player_rating(name, season):
    return requests.get(f"{API}/player/{name}", params={"season": season}).json()


@st.cache_data(ttl=300)
def get_2k27_prediction(name):
    return requests.get(f"{API}/predict/2k27/{name}").json()


@st.cache_data(ttl=300)
def search_players(query):
    return requests.get(f"{API}/search/{query}").json()


# ── Page header ───────────────────────────────────────────────────────────────

st.title("🏀 NBA 2K Rating Predictor")
st.caption("Predict 2K27 ratings using real NBA stats + Machine Learning · Model MAE: ±1.16 OVR · R² = 0.942")

tab1, tab2, tab3 = st.tabs(["Player Lookup", "Leaderboard", "Ask Anything"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Player Lookup
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:

    if "featured_player" in st.session_state:
        initial_value = st.session_state.pop("featured_player")
    else:
        initial_value = ""

    search_col, season_col = st.columns([3, 1])
    with search_col:
        player_input = st.text_input(
            "Search player",
            value=initial_value,
            placeholder="e.g. LeBron James, Nikola Jokic, Stephen Curry...",
            label_visibility="collapsed",
        )
    with season_col:
        season = st.selectbox(
            "Season",
            ["2024-25", "2023-24", "2022-23", "2021-22", "2020-21", "2019-20", "2018-19"],
            label_visibility="collapsed"
        )

    selected = None
    if player_input and len(player_input) >= 2:
        search_res = search_players(player_input)
        suggestions = search_res.get("results", [])
        if suggestions:
            selected = st.selectbox(
                "Select player",
                suggestions,
                label_visibility="collapsed"
            )
        else:
            st.warning("No players found — try a different name")

    st.divider()

    if selected:
        col1, col2 = st.columns(2)

        with col1:
            player_res = get_player_rating(selected, season)

            if "detail" not in player_res:
                actual    = player_res.get("actual_ovr")
                predicted = player_res.get("predicted_ovr")

                st.metric(
                    label=f"2K Rating — {season}",
                    value=int(actual) if actual else "N/A",
                    delta=f"Model predicts {predicted} (±1.16 avg error)" if predicted else None
                )
                st.markdown(tier_badge(actual), unsafe_allow_html=True)
                st.caption("Stats")
                stats = player_res.get("stats", {})
                st.dataframe(
                    pd.DataFrame([stats]).rename(columns={
                        "pts": "PTS", "reb": "REB", "ast": "AST",
                        "stl": "STL", "blk": "BLK", "usg_pct": "USG%"
                    }),
                    column_config={
                        "PTS": num_col(), "REB": num_col(), "AST": num_col(),
                        "STL": num_col(), "BLK": num_col(), "USG%": num_col("%.3f"),
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.warning(player_res["detail"])

        with col2:
            pred_res = get_2k27_prediction(selected)

            if "detail" not in pred_res:
                last_ovr  = pred_res.get("last_known_ovr")
                pred_ovr  = pred_res.get("predicted_2k27_ovr")
                rounded   = pred_res.get("rounded_ovr")
                delta_val = round(pred_ovr - last_ovr, 1) if pred_ovr and last_ovr else None

                st.metric(
                    label="Predicted 2K27 Rating",
                    value=rounded if rounded else "N/A",
                    delta=f"{delta_val:+.1f} from 2K26" if delta_val is not None else None,
                    delta_color=delta_color(delta_val)
                )
                st.markdown(tier_badge(rounded), unsafe_allow_html=True)
                st.caption("2025-26 stats (basis for prediction)")
                cs = pred_res.get("current_stats", {})
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
                    hide_index=True,
                    use_container_width=True
                )
                st.caption(pred_res.get("note", ""))
            else:
                st.warning(pred_res["detail"])

        st.divider()

        # ── Overrated / underrated indicator ─────────────────────────────────
        if "detail" not in player_res and player_res.get("actual_ovr") and player_res.get("predicted_ovr"):
            actual_ovr = player_res["actual_ovr"]
            model_pred = player_res["predicted_ovr"]
            diff       = round(actual_ovr - model_pred, 1)

            if abs(diff) < 1.5:
                label = "Fairly rated"
                color = "#40d080"
                desc  = f"2K rating ({actual_ovr}) closely matches what the model expects ({model_pred}) based on stats alone."
            elif diff > 0:
                label = f"Overrated by ~{abs(diff)} OVR"
                color = "#f04060"
                desc  = f"2K rating ({actual_ovr}) is higher than stats justify ({model_pred}). Likely benefiting from reputation."
            else:
                label = f"Underrated by ~{abs(diff)} OVR"
                color = "#40a0f0"
                desc  = f"2K rating ({actual_ovr}) is lower than stats justify ({model_pred}). Deserves a higher rating."

            st.markdown(
                f'<div style="background:{color}11;border-left:3px solid {color};'
                f'padding:0.75rem 1rem;border-radius:0 8px 8px 0;margin-bottom:1rem">'
                f'<strong style="color:{color}">{label}</strong><br>'
                f'<span style="font-size:0.85rem;color:#aaa">{desc}</span></div>',
                unsafe_allow_html=True
            )

        # ── Career trajectory chart ───────────────────────────────────────────
        history_res = get_player_history(selected)

        if "history" in history_res:
            df = pd.DataFrame(history_res["history"])
            df["game_label"] = df["game_version"].apply(format_game_version)

            fig = go.Figure()

            actual_df = df[df["ovr_rating"].notna()]
            fig.add_trace(go.Scatter(
                x=actual_df["game_label"],
                y=actual_df["ovr_rating"],
                mode="lines+markers",
                name="Actual OVR",
                line=dict(color="#f0c040", width=3),
                marker=dict(size=8),
            ))

            if "detail" not in pred_res and len(actual_df) > 0:
                last_row = actual_df.iloc[-1]
                fig.add_trace(go.Scatter(
                    x=[last_row["game_label"], "2K27"],
                    y=[last_row["ovr_rating"], pred_res.get("rounded_ovr")],
                    mode="lines+markers",
                    name="Predicted 2K27",
                    line=dict(color="#40a0f0", width=2, dash="dash"),
                    marker=dict(size=10, symbol="star"),
                ))

            fig.update_layout(
                title=f"{selected} — Career Rating Trajectory",
                xaxis_title="Game Version",
                yaxis_title="OVR Rating",
                yaxis=dict(range=[65, 100]),
                plot_bgcolor="#0a0a0f",
                paper_bgcolor="#0a0a0f",
                font=dict(color="#e8e8f0"),
                legend=dict(bgcolor="#111118", bordercolor="#2a2a38"),
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Full career history"):
                display_cols = ["game_label", "season", "ovr_rating", "pts",
                               "reb", "ast", "age", "gp", "ovr_delta", "split"]
                display_df = df[display_cols].rename(columns={
                    "game_label": "Game", "season": "Season",
                    "ovr_rating": "OVR", "pts": "PTS", "reb": "REB",
                    "ast": "AST", "age": "AGE", "gp": "GP",
                    "ovr_delta": "OVR Δ", "split": "Split"
                })
                st.dataframe(
                    display_df.style.map(color_delta_style, subset=["OVR Δ"]),
                    column_config={
                        "OVR":  num_col(), "PTS": num_col(),
                        "REB":  num_col(), "AST": num_col(),
                        "AGE":  num_col(), "GP":  num_col("%d"),
                        "OVR Δ": num_col(),
                    },
                    hide_index=True,
                    use_container_width=True
                )

    else:
        st.markdown("#### Featured players — click to load")
        featured = [
            "LeBron James", "Nikola Jokic", "Stephen Curry",
            "Giannis Antetokounmpo", "Luka Doncic", "Kevin Durant"
        ]
        fcols = st.columns(3)
        for i, name in enumerate(featured):
            if fcols[i % 3].button(f"🏀 {name}", key=f"f{i}", use_container_width=True):
                st.session_state["featured_player"] = name
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Leaderboard
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Top rated players")

    top10, improved, declined = get_leaderboard()
    pred_rows = get_2k27_predictions()

    lcol1, lcol2 = st.columns(2)

    with lcol1:
        st.markdown("#### 2K26 Top 10")
        if top10.get("data"):
            df_top = pd.DataFrame(top10["data"])
            df_top.index = range(1, len(df_top) + 1)
            st.dataframe(
                df_top,
                column_config={
                    "ovr_rating": num_col(), "pts": num_col(),
                    "reb": num_col(), "ast": num_col(),
                },
                use_container_width=True
            )

    with lcol2:
        st.markdown("#### Predicted 2K27 Top 10")
        if pred_rows:
            df_p = pd.DataFrame(pred_rows)
            st.dataframe(
                df_p.style.map(color_delta_style, subset=["Δ"]),
                column_config={
                    "2K26": num_col(), "Predicted 2K27": num_col(),
                    "Δ": num_col(), "PTS": num_col(), "AGE": num_col(),
                },
                hide_index=True,
                use_container_width=True
            )

    st.divider()
    st.subheader("Biggest movers in 2K26")

    mcol1, mcol2 = st.columns(2)

    with mcol1:
        st.markdown("#### Most improved")
        if improved.get("data"):
            df_up = pd.DataFrame(improved["data"])
            st.dataframe(
                df_up,
                column_config={
                    "ovr_rating": num_col(), "ovr_prev": num_col(), "ovr_delta": num_col(),
                },
                hide_index=True,
                use_container_width=True
            )

    with mcol2:
        st.markdown("#### Biggest declines")
        if declined.get("data"):
            df_dn = pd.DataFrame(declined["data"])
            st.dataframe(
                df_dn,
                column_config={
                    "ovr_rating": num_col(), "ovr_prev": num_col(), "ovr_delta": num_col(),
                },
                hide_index=True,
                use_container_width=True
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Ask Anything
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Ask anything about the database")
    st.caption("Powered by Claude — converts your question to SQL and runs it live")

    QUICK = [
        "Who had the biggest rating jump in 2K26?",
        "What team has the highest average rating in 2024-25?",
        "What are the predicted 2K27 ratings for the top 10 players?",
        "Who is the most underrated player in 2K26?",
        "Which players declined the most from 2K25 to 2K26?",
        "Who has the highest career average OVR rating?",
    ]

    st.markdown("**Quick questions:**")
    qcols = st.columns(3)
    for i, q in enumerate(QUICK):
        if qcols[i % 3].button(q, key=f"q{i}", use_container_width=True):
            st.session_state["ask_question"] = q

    question = st.text_input(
        "Ask a question",
        value=st.session_state.get("ask_question", ""),
        placeholder="e.g. Who is the most underrated player in 2K26?",
        label_visibility="collapsed"
    )

    if st.button("Ask", type="primary") and question:
        with st.spinner("Generating SQL and querying database..."):
            res = requests.get(f"{API}/ask", params={"q": question}).json()

        st.markdown("**Answer:**")
        st.markdown(res.get("answer", "No answer"))

        with st.expander("Generated SQL"):
            st.code(res.get("sql", ""), language="sql")

        if res.get("data"):
            with st.expander("Raw data", expanded=True):
                raw_df = pd.DataFrame(res["data"])
                numeric_cols = raw_df.select_dtypes(include="number").columns.tolist()
                col_config = {c: num_col() for c in numeric_cols}
                st.dataframe(
                    raw_df,
                    column_config=col_config,
                    hide_index=True,
                    use_container_width=True
                )