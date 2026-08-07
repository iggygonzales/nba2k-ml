import os
import time
import requests
import pandas as pd
from io import StringIO

# ── Config ────────────────────────────────────────────────────────────────────

SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23",
           "2023-24", "2024-25", "2025-26"]
CURRENT_SEASON = SEASONS[-1]

STATS_DIR = "data/raw/stats"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
}

PG_COLS = [
    "Player", "Tm", "Age", "G", "MP",
    "FG", "FGA", "FT", "FTA",
    "ORB", "DRB", "TRB",
    "AST", "STL", "BLK", "TOV", "PF",
    "PTS", "FG%", "3P%", "FT%",
]

ADV_COLS = ["Player", "Tm", "PER", "USG%", "AST%", "BPM", "TS%"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def season_to_year(season):
    """'2025-26' -> 2026"""
    return int(season.split("-")[0]) + 1


def season_file_path(season):
    return os.path.join(STATS_DIR, f"nba_player_stats_{season}.csv")


def fetch_table(url, candidate_ids):
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    html = resp.text
    for tid in candidate_ids:
        try:
            df = pd.read_html(StringIO(html), attrs={"id": tid})[0]
            return df
        except Exception:
            continue
    tables = pd.read_html(StringIO(html))
    return max(tables, key=len)


def clean_player_col(df):
    player_col = next((c for c in df.columns if str(c).lower() in ("player", "name")), None)
    if player_col:
        # Drop header repeat rows
        df = df[df[player_col].astype(str) != player_col].copy()
        df[player_col] = df[player_col].str.replace("*", "", regex=False).str.strip()
        # FIX: Drop the "League Average" row that Basketball Reference appends
        df = df[df[player_col].astype(str).str.strip() != "League Average"].copy()
    return df.reset_index(drop=True)


def fetch_per_game(year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
    df  = fetch_table(url, ["per_game_stats", "per_game", "players"])
    df  = clean_player_col(df)
    keep = [c for c in PG_COLS if c in df.columns]
    return df[keep].copy()


def fetch_advanced(year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
    df  = fetch_table(url, ["advanced_stats", "advanced", "players_advanced"])
    df  = clean_player_col(df)
    keep = [c for c in ADV_COLS if c in df.columns]
    return df[keep].copy()


def fetch_league_averages(year):
    url  = f"https://www.basketball-reference.com/leagues/NBA_{year}.html"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    tables = pd.read_html(StringIO(resp.text))
    for table in tables:
        if "PTS" not in table.columns:
            continue
        mask = table.apply(
            lambda col: col.astype(str).str.contains("League Average", case=False, na=False)
        ).any(axis=1)
        if mask.any():
            return table[mask].iloc[0]
    return None


# NOTE: PIE has been commented out in favor of using PER instead (see fetch_season below).
# def calculate_pie(df, league_row):
#     def player_num(d):
#         def g(k):
#             return pd.to_numeric(d[k], errors="coerce").fillna(0) if k in d.columns else 0
#         return (
#             g("PTS") + g("FG") + g("FT")
#           - g("FGA") - g("FTA")
#           + g("DRB") + g("ORB") * 0.5
#           + g("AST") + g("STL") + g("BLK") * 0.5
#           - g("PF")  - g("TOV")
#         )
#
#     def league_num(r):
#         def g(k):
#             return float(pd.to_numeric(r.get(k, 0), errors="coerce") or 0)
#         return (
#             g("PTS") + g("FG") + g("FT")
#           - g("FGA") - g("FTA")
#           + g("DRB") + g("ORB") * 0.5
#           + g("AST") + g("STL") + g("BLK") * 0.5
#           - g("PF")  - g("TOV")
#         )
#
#     p_num = player_num(df)
#
#     if league_row is not None:
#         l_num = league_num(league_row)
#         if l_num != 0:
#             # FIX: divide by sum, not just league numerator
#             return p_num / (p_num + l_num)
#
#     # Fallback
#     max_val = p_num.abs().max()
#     return p_num / max_val if max_val != 0 else p_num


# ── Per-season fetch ──────────────────────────────────────────────────────────

def fetch_season(season):
    year = season_to_year(season)

    print(f"  Fetching per-game stats ({year})...")
    pg = fetch_per_game(year)
    time.sleep(4)

    print(f"  Fetching advanced stats ({year})...")
    adv = fetch_advanced(year)
    time.sleep(4)

    print(f"  Fetching league averages ({year})...")
    league_row = fetch_league_averages(year)
    time.sleep(4)

    # Players traded mid-season appear multiple times — keep TOT (combined) row
    if "Tm" in pg.columns:
        pg = pd.concat([pg[pg["Tm"] != "TOT"], pg[pg["Tm"] == "TOT"]]) \
               .drop_duplicates("Player", keep="last")
    if "Tm" in adv.columns:
        adv = pd.concat([adv[adv["Tm"] != "TOT"], adv[adv["Tm"] == "TOT"]]) \
                .drop_duplicates("Player", keep="last")

    # Final safety dedup — catches any remaining duplicates with blank Tm
    pg  = pg.drop_duplicates("Player", keep="first")
    adv = adv.drop_duplicates("Player", keep="first")

    merged = pg.merge(adv, on="Player", how="left", suffixes=("", "_adv"))

    # Drop Tm columns — no longer needed after dedup
    merged = merged.drop(columns=[c for c in merged.columns if c in ("Tm", "Tm_adv")], errors="ignore")

    # PIE calculation disabled — using PER (from bref advanced stats) instead
    # merged["PIE"] = calculate_pie(merged, league_row)

    # Calculate AST_TO ratio (bref doesn't have it as a column)
    ast = pd.to_numeric(merged.get("AST"), errors="coerce").fillna(0)
    tov = pd.to_numeric(merged.get("TOV"), errors="coerce").fillna(0)
    merged["AST_TO"] = (ast / tov.replace(0, float("nan"))).round(2)

    # Rename bref columns to match existing schema
    merged = merged.rename(columns={
        "Player": "PLAYER_NAME",
        "Age":    "AGE",
        "G":      "GP",
        "MP":     "MIN",
        "TRB":    "REB",
        "FG%":    "FG_PCT",
        "3P%":    "FG3_PCT",
        "FT%":    "FT_PCT",
        "USG%":   "USG_PCT",
        "AST%":   "AST_PCT",
        "BPM":    "NET_RATING",
        "PER":    "PER",
    })

    # Add columns bref doesn't have
    merged["PLAYER_ID"]         = None
    merged["TEAM_ABBREVIATION"] = None
    merged["SEASON"]            = season
    merged["season_year"]       = season_to_year(season)

    # Cast numerics
    num_cols = [
        "AGE", "GP", "MIN", "PTS", "REB", "AST", "STL", "BLK",
        "TOV", "FG_PCT", "FG3_PCT", "FT_PCT", "USG_PCT",
        "AST_PCT", "NET_RATING", "PER", "AST_TO", "TS%",  # PIE removed
    ]
    for col in num_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Match exact column order from original NBA API CSV
    final_cols = [
        "SEASON", "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION",
        "AGE", "GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV",
        "FG_PCT", "FG3_PCT", "FT_PCT", "NET_RATING", "USG_PCT",
        "AST_PCT", "AST_TO", "PER", "season_year", "TS%",  # PIE removed
    ]
    available = [c for c in final_cols if c in merged.columns]
    return merged[available].copy()


# ── Main ──────────────────────────────────────────────────────────────────────

os.makedirs(STATS_DIR, exist_ok=True)

# Check which seasons are already saved as individual files
have_seasons = {s for s in SEASONS if os.path.exists(season_file_path(s))}
print(f"Seasons already on disk: {sorted(have_seasons)}")

# Always refetch the current season — stats update as games are played
seasons_to_fetch = [
    s for s in SEASONS if s == CURRENT_SEASON or s not in have_seasons
]

if not seasons_to_fetch:
    print("Nothing new to fetch — all seasons already on disk.")
else:
    print(f"Fetching: {seasons_to_fetch}\n")

    for season in seasons_to_fetch:
        print(f"\n[{season}]")
        df   = fetch_season(season)
        path = season_file_path(season)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  -> {len(df)} players saved to {path}")

print("\nDone!")
print(f"\nSample (top 5 by PTS in {CURRENT_SEASON}):")
current_path = season_file_path(CURRENT_SEASON)
if os.path.exists(current_path):
    sample_df = pd.read_csv(current_path, encoding="utf-8-sig")
    print(
        sample_df.sort_values("PTS", ascending=False)
        .head()[["PLAYER_NAME", "PTS", "REB", "AST", "PER", "NET_RATING", "TS%"]]  # PIE removed
        .to_string(index=False)
    )