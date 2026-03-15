"""
Build Database Pipeline
-----------------------
1. Loads all ratings CSVs from data/raw/ratings/
2. Loads stats CSV from data/raw/stats/
3. Fuzzy matches player names
4. Joins on PLAYER_NAME + SEASON
5. Bulk inserts everything into Postgres using SQLAlchemy

Run from project root:
    python pipeline/build_database.py
"""

import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from thefuzz import process

load_dotenv()

RATINGS_DIR    = os.path.join("data", "raw", "ratings")
STATS_PATH     = os.path.join("data", "raw", "stats", "nba_player_stats_2018_2026.csv")
PROCESSED_PATH = os.path.join("data", "processed", "joined_dataset.csv")
FUZZY_THRESHOLD = 85

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://nba2k_user:nba2k_pass@localhost:5432/nba2k_db"
)

MANUAL_MAP = {
    "Kristaps Porzingis": "Kristaps Porziņģis",
    "Luka Doncic":        "Luka Dončić",
    "TJ Leaf":            "T.J. Leaf",
    "Moe Harkless":       "Maurice Harkless",
    "Jonas Valanciunas":  "Jonas Valančiūnas",
    "Nikola Vucevic":     "Nikola Vučević",
    "Vlatko Cancar":      "Vlatko Čančar",
    "Cameron Reynolds":   "Cam Reynolds",
    "Anzejs Pasecniks":   "Anžejs Pasečņiks",
    "Vit Krejci":         "Vít Krejčí",
    "Alexandre Sarr":     "Alex Sarr",
}


# ── Database ──────────────────────────────────────────────────────────────────

def get_db():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB", "nba2k_db"),
        user=os.getenv("POSTGRES_USER", "nba2k_user"),
        password=os.getenv("POSTGRES_PASSWORD", "nba2k_pass"),
    )


def get_engine():
    return create_engine(DATABASE_URL)


# ── Load data ─────────────────────────────────────────────────────────────────

def load_ratings():
    dfs = []
    for f in sorted(os.listdir(RATINGS_DIR)):
        if f.endswith(".csv"):
            dfs.append(pd.read_csv(os.path.join(RATINGS_DIR, f)))
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} ratings rows across {df['SEASON'].nunique()} seasons")
    return df


def load_stats():
    df = pd.read_csv(STATS_PATH)
    if "season_year" not in df.columns:
        df["season_year"] = df["SEASON"].apply(lambda x: int(x.split("-")[0]) + 1)
    print(f"Loaded {len(df)} stats rows across {df['SEASON'].nunique()} seasons")
    return df


# ── Name matching ─────────────────────────────────────────────────────────────

def build_name_map(ratings_names, stats_names):
    print("\nFuzzy matching player names...")
    name_map  = {}
    unmatched = []
    stats_list = list(stats_names)

    for name in ratings_names:
        if name in MANUAL_MAP:
            name_map[name] = MANUAL_MAP[name]
            continue
        result = process.extractOne(name, stats_list)
        if result and result[1] >= FUZZY_THRESHOLD:
            name_map[name] = result[0]
        else:
            unmatched.append((name, result))

    print(f"  Matched:   {len(name_map)}")
    print(f"  Unmatched: {len(unmatched)} (will be flagged as rookies)")
    if unmatched:
        print("\n  Unmatched players:")
        for name, result in unmatched:
            print(f"    '{name}' -> best guess: {result}")

    return name_map


# ── Join ──────────────────────────────────────────────────────────────────────

def join_datasets(ratings, stats, name_map):
    ratings["PLAYER_NAME_MATCHED"] = ratings["PLAYER_NAME"].map(name_map)

    merged = ratings.merge(
        stats,
        left_on=["PLAYER_NAME_MATCHED", "SEASON"],
        right_on=["PLAYER_NAME", "SEASON"],
        how="left",
        suffixes=("_2k", "_nba")
    )

    merged["IS_ROOKIE"] = merged["PTS"].isna()

    print(f"\nJoin results:")
    print(f"  Matched rows:   {len(merged[~merged['IS_ROOKIE']])}")
    print(f"  Unmatched rows: {len(merged[merged['IS_ROOKIE']])} (rookies / missing stats)")

    return merged


# ── Bulk write to Postgres ────────────────────────────────────────────────────

def write_to_postgres_bulk(merged, stats):
    engine = get_engine()

    print("\nClearing existing data...")
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE ratings, stats, features, players RESTART IDENTITY CASCADE"))
        conn.commit()

    # ── Players ──────────────────────────────────────────────────────────────
    players_df = stats[["PLAYER_ID", "PLAYER_NAME"]].drop_duplicates("PLAYER_ID").copy()
    players_df["first_name"] = players_df["PLAYER_NAME"].apply(
        lambda x: str(x).split(" ")[0]
    )
    players_df["last_name"] = players_df["PLAYER_NAME"].apply(
        lambda x: " ".join(str(x).split(" ")[1:])
    )
    players_df.rename(columns={
        "PLAYER_ID":   "player_id",
        "PLAYER_NAME": "full_name",
    }, inplace=True)
    players_df.to_sql("players", engine, if_exists="append", index=False, method="multi")
    print(f"  Players inserted: {len(players_df)}")

    # ── Ratings ───────────────────────────────────────────────────────────────
    valid_ids = set(players_df["player_id"].tolist())
    rated = merged[merged["PLAYER_ID"].notna()].copy()
    rated = rated[rated["PLAYER_ID"].astype(int).isin(valid_ids)]
    rated["player_id"]    = rated["PLAYER_ID"].astype(int)
    rated["season_year"]  = rated["SEASON"].apply(lambda x: int(x.split("-")[0]) + 1)
    rated["ovr_rating"]   = rated["RATING"].where(pd.notna(rated["RATING"]), None)
    rated["team_in_game"] = rated.get("TEAM_2k", rated.get("TEAM", ""))
    rated["rank_in_game"] = rated["RANK"]
    rated["is_rookie"]    = rated["PTS"].isna()

    ratings_df = rated[[
        "player_id", "season_year", "ovr_rating",
        "team_in_game", "rank_in_game", "GAME_VERSION", "is_rookie"
    ]].copy()
    ratings_df.rename(columns={"GAME_VERSION": "game_version"}, inplace=True)
    ratings_df = ratings_df.drop_duplicates(["player_id", "season_year"])
    ratings_df.to_sql("ratings", engine, if_exists="append", index=False, method="multi")
    print(f"  Ratings inserted: {len(ratings_df)}")

    # ── Stats (all seasons including 2025-26) ─────────────────────────────────
    all_stats = stats[stats["PLAYER_ID"].isin(valid_ids)].copy()
    all_stats.rename(columns={
        "PLAYER_ID":        "player_id",
        "season_year":      "season_year",
        "TEAM_ABBREVIATION":"team_abbr",
        "AGE":              "age",
        "GP":               "gp",
        "MIN":              "min",
        "PTS":              "pts",
        "REB":              "reb",
        "AST":              "ast",
        "STL":              "stl",
        "BLK":              "blk",
        "TOV":              "tov",
        "FG_PCT":           "fg_pct",
        "FG3_PCT":          "fg3_pct",
        "FT_PCT":           "ft_pct",
        "NET_RATING":       "net_rating",
        "USG_PCT":          "usg_pct",
        "AST_PCT":          "ast_pct",
        "AST_TO":           "ast_to",
        "PIE":              "pie",
    }, inplace=True)

    stats_cols = [
        "player_id", "season_year", "team_abbr", "age",
        "gp", "min", "pts", "reb", "ast", "stl", "blk", "tov",
        "fg_pct", "fg3_pct", "ft_pct",
        "net_rating", "usg_pct", "ast_pct", "ast_to", "pie",
    ]
    all_stats = all_stats[stats_cols].drop_duplicates(["player_id", "season_year"])
    all_stats.to_sql("stats", engine, if_exists="append", index=False, method="multi")
    print(f"  Stats inserted:   {len(all_stats)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Build Database Pipeline")
    print("=" * 50)

    ratings = load_ratings()
    stats   = load_stats()

    name_map = build_name_map(
        ratings["PLAYER_NAME"].unique(),
        stats["PLAYER_NAME"].unique()
    )

    merged = join_datasets(ratings, stats, name_map)

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    merged.to_csv(PROCESSED_PATH, index=False)
    print(f"\nSaved joined dataset to {PROCESSED_PATH}")

    write_to_postgres_bulk(merged, stats)

    print("\nDone! Verifying row counts...")
    conn = get_db()
    cur  = conn.cursor()
    for table in ["players", "ratings", "stats"]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        print(f"  {table}: {cur.fetchone()[0]} rows")
    cur.execute("SELECT split, COUNT(*) FROM ml_dataset GROUP BY split ORDER BY split")
    print("\nml_dataset split distribution:")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]} rows")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()