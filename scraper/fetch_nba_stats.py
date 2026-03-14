import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

BASIC_COLS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE",
    "GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV",
    "FG_PCT", "FG3_PCT", "FT_PCT",
]

ADVANCED_COLS = [
    "PLAYER_ID", "PLAYER_NAME",
    "NET_RATING", "USG_PCT", "AST_PCT", "AST_TO", "PIE",
]

SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

all_dfs = []

for season in SEASONS:
    print(f"Fetching basic stats for {season}...")
    basic_df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense="Base",
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]
    basic_df = basic_df[BASIC_COLS]

    print(f"Fetching advanced stats for {season}...")
    advanced_df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]
    advanced_df = advanced_df[ADVANCED_COLS]

    merged_df = basic_df.merge(advanced_df, on=["PLAYER_ID", "PLAYER_NAME"])
    merged_df.insert(0, "SEASON", season)  # add season column at front
    all_dfs.append(merged_df)
    print(f"  -> {len(merged_df)} players fetched.\n")

final_df = pd.concat(all_dfs, ignore_index=True)

output_path = "data/raw/stats/nba_player_stats_2018_2026.csv"
final_df.to_csv(output_path, index=False)

print(f"Done! {len(final_df)} total rows saved to {output_path}")
print(f"\nSample (top 5 by PTS in 2024-25):")
print(
    final_df[final_df["SEASON"] == "2024-25"]
    .sort_values("PTS", ascending=False)
    .head()
    .to_string(index=False)
)