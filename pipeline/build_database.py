"""
Build Database Pipeline
-----------------------
1. Loads all ratings CSVs from data/raw/ratings/
2. Loads all per-season stats CSVs from data/raw/stats/
3. Fuzzy matches player names
4. Joins on PLAYER_NAME + SEASON
5. Bulk inserts everything into Postgres using SQLAlchemy
6. Builds features table (career year, deltas)

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
STATS_DIR      = os.path.join("data", "raw", "stats")
PROCESSED_PATH = os.path.join("data", "processed", "joined_dataset.csv")
FUZZY_THRESHOLD = 85

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://nba2k_user:nba2k_pass@localhost:5432/nba2k_db"
)

MANUAL_MAP = {
    # Existing correct entries
    "Kristaps Porzingis":       "Kristaps Porziņģis",
    "Luka Doncic":              "Luka Dončić",
    "TJ Leaf":                  "T.J. Leaf",
    "Moe Harkless":             "Maurice Harkless",
    "Jonas Valanciunas":        "Jonas Valančiūnas",
    "Nikola Vucevic":           "Nikola Vučević",
    "Vlatko Cancar":            "Vlatko Čančar",
    "Anzejs Pasecniks":         "Anžejs Pasečņiks",
    "Vit Krejci":               "Vít Krejčí",
    "Alexandre Sarr":           "Alex Sarr",

    # Diacritics (ratings has ASCII, bref has accented)
    "Dzanan Musa":              "Džanan Musa",
    "Bojan Bogdanovic":         "Bojan Bogdanović",
    "Bogdan Bogdanovic":        "Bogdan Bogdanović",
    "Dairis Bertans":           "Dairis Bertāns",
    "Milos Teodosic":           "Miloš Teodosić",
    "Skal Labissiere":          "Skal Labissière",
    "Timothe Luwawu-Cabarrot":  "Timothé Luwawu-Cabarrot",
    "Donatas Motiejunas":       "Donatas Motiejūnas",
    "Angel Delgado":            "Ángel Delgado",
    "Jose Calderon":            "José Calderón",
    "Nene":                     "Nenê",
    "Cristiano Felicio":        "Cristiano Felício",

    # Name variations — ratings name -> bref name
    "Robert Williams":          "Robert Williams III",
    "Vince Edwards":            "Vincent Edwards",
    "Wade Baldwin":             "Wade Baldwin IV",
    "Mitch Creek":              "Mitchell Creek",
    "Cameron Reynolds":         "Cam Reynolds",
    "DJ Stephens":              "D.J. Stephens",
    "BJ Johnson":               "B.J. Johnson",
    "Melvin Frazier":           "Melvin Frazier Jr.",
    "AJ Green":                 "A.J. Green",

    "Ante Zizic":               "Ante Žižić",
    "Dario Saric":              "Dario Šarić",
    "Luka Samanic":             "Luka Šamanić",
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
    """
    Load all per-season stats CSVs from data/raw/stats/.
    Each file is named nba_player_stats_YYYY-YY.csv (e.g. nba_player_stats_2025-26.csv).
    """
    dfs = []
    for f in sorted(os.listdir(STATS_DIR)):
        if f.endswith(".csv") and f.startswith("nba_player_stats_"):
            df = pd.read_csv(
                os.path.join(STATS_DIR, f),
                encoding="utf-8-sig"  # handles special characters like Dončić
            )
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No stats CSVs found in {STATS_DIR}")

    stats = pd.concat(dfs, ignore_index=True)

    # Ensure season_year column exists
    if "season_year" not in stats.columns:
        stats["season_year"] = stats["SEASON"].apply(lambda x: int(x.split("-")[0]) + 1)

    print(f"Loaded {len(stats)} stats rows across {stats['SEASON'].nunique()} seasons")
    return stats


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

def write_to_postgres_bulk(merged, stats, engine):
    print("\nClearing existing data...")
    with engine.connect() as conn:
        for table in ["features", "ratings", "stats", "players"]:
            conn.execute(text(f"""
                DO $$ BEGIN
                    IF EXISTS (SELECT FROM pg_tables WHERE tablename = '{table}') THEN
                        EXECUTE 'TRUNCATE {table} RESTART IDENTITY CASCADE';
                    END IF;
                END $$;
            """))
        conn.commit()

    # ── Players ───────────────────────────────────────────────────────────────
    # Basketball Reference has no PLAYER_ID so we generate our own integer IDs
    # from unique player names across all seasons.
    all_names = pd.Series(stats["PLAYER_NAME"].dropna().unique()).sort_values().reset_index(drop=True)
    players_df = pd.DataFrame({
        "player_id":  range(1, len(all_names) + 1),
        "full_name":  all_names,
        "first_name": all_names.apply(lambda x: str(x).split(" ")[0]),
        "last_name":  all_names.apply(lambda x: " ".join(str(x).split(" ")[1:])),
    })

    # Build a lookup from name -> generated player_id for use below
    name_to_id = dict(zip(players_df["full_name"], players_df["player_id"]))

    players_df.to_sql("players", engine, if_exists="append", index=False,
                      method="multi", chunksize=500)
    print(f"  Players inserted: {len(players_df)}")

    # ── Ratings ───────────────────────────────────────────────────────────────
    rated = merged.copy()
    # Map the matched player name -> generated player_id
    rated["player_id"] = rated["PLAYER_NAME_MATCHED"].map(name_to_id)
    rated = rated[rated["player_id"].notna()].copy()
    rated["player_id"]    = rated["player_id"].astype(int)
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
    ratings_df.to_sql("ratings", engine, if_exists="append", index=False,
                      method="multi", chunksize=500)
    print(f"  Ratings inserted: {len(ratings_df)}")

    # ── Stats ─────────────────────────────────────────────────────────────────
    all_stats = stats.copy()
    all_stats["player_id"] = all_stats["PLAYER_NAME"].map(name_to_id)
    all_stats = all_stats[all_stats["player_id"].notna()].copy()
    all_stats["player_id"] = all_stats["player_id"].astype(int)

    all_stats.rename(columns={
        "season_year":       "season_year",
        "TEAM_ABBREVIATION": "team_abbr",
        "AGE":               "age",
        "GP":                "gp",
        "MIN":               "min",
        "PTS":               "pts",
        "REB":               "reb",
        "AST":               "ast",
        "STL":               "stl",
        "BLK":               "blk",
        "TOV":               "tov",
        "FG_PCT":            "fg_pct",
        "FG3_PCT":           "fg3_pct",
        "FT_PCT":            "ft_pct",
        "NET_RATING":        "net_rating",
        "USG_PCT":           "usg_pct",
        "AST_PCT":           "ast_pct",
        "AST_TO":            "ast_to",
        "TS%":               "ts_pct",
        "PER":               "per",
    }, inplace=True)

    stats_cols = [
        "player_id", "season_year", "age",
        "gp", "min", "pts", "reb", "ast", "stl", "blk", "tov",
        "fg_pct", "fg3_pct", "ft_pct",
        "net_rating", "usg_pct", "ast_pct", "ast_to", "per",
    ]
    all_stats = (
        all_stats[stats_cols]
        .dropna(subset=["season_year"])
        .sort_values("gp", ascending=False)
        .drop_duplicates(["player_id", "season_year"], keep="first")
    )
    all_stats["net_rating"] = all_stats["net_rating"].clip(-999, 999)
    all_stats.to_sql("stats", engine, if_exists="append", index=False,
                     method="multi", chunksize=500)
    print(f"  Stats inserted:   {len(all_stats)}")

    

# ── Build features ────────────────────────────────────────────────────────────

def build_features(engine):
    print("\nBuilding features table...")

    df = pd.read_sql("""
        SELECT player_id, season_year, ovr_rating, pts, reb, ast
        FROM ml_dataset
        ORDER BY player_id, season_year
    """, engine)

    df = df.sort_values(["player_id", "season_year"])
    df["career_year"] = df.groupby("player_id").cumcount() + 1
    df["pts_delta"]   = df.groupby("player_id")["pts"].diff().round(2)
    df["reb_delta"]   = df.groupby("player_id")["reb"].diff().round(2)
    df["ast_delta"]   = df.groupby("player_id")["ast"].diff().round(2)
    df["ovr_prev"]    = df.groupby("player_id")["ovr_rating"].shift(1)
    df["ovr_delta"]   = df.groupby("player_id")["ovr_rating"].diff()

    features_df = df[[
        "player_id", "season_year", "career_year",
        "pts_delta", "reb_delta", "ast_delta",
        "ovr_prev", "ovr_delta"
    ]].copy()

    with engine.connect() as conn:
        conn.execute(text("TRUNCATE features"))
        conn.commit()

    features_df.to_sql("features", engine, if_exists="append",
                       index=False, method="multi", chunksize=500)
    print(f"  Features inserted: {len(features_df)} rows")


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
    merged.to_csv(PROCESSED_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved joined dataset to {PROCESSED_PATH}")

    engine = get_engine()
    write_to_postgres_bulk(merged, stats, engine)
    build_features(engine)

    print("\nDone! Verifying row counts...")
    conn = get_db()
    cur  = conn.cursor()
    for table in ["players", "ratings", "stats", "features"]:
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