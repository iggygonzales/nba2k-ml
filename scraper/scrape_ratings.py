"""
NBA 2K Ratings Scraper — All Seasons (2K20 through 2K26)
Saves each season to data/raw/ratings/ratings_{game}.csv

Run from project root:
    python scraper/scrape_ratings.py
"""

import time
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

BASE_URL = "https://www.hoopshype.com/nba-2k/players/?game={game}&team={team}"
OUTPUT_DIR = os.path.join("data", "raw", "ratings")

GAME_TO_SEASON = {
    "nba-2k20": "2018-19",
    "nba-2k21": "2019-20",
    "nba-2k22": "2020-21",
    "nba-2k23": "2021-22",
    "nba-2k24": "2022-23",
    "nba-2k25": "2023-24",
    "nba-2k26": "2024-25",
}

NBA_TEAMS = [
    "atlanta-hawks", "boston-celtics", "brooklyn-nets", "charlotte-hornets",
    "chicago-bulls", "cleveland-cavaliers", "dallas-mavericks", "denver-nuggets",
    "detroit-pistons", "golden-state-warriors", "houston-rockets", "indiana-pacers",
    "los-angeles-clippers", "los-angeles-lakers", "memphis-grizzlies", "miami-heat",
    "milwaukee-bucks", "minnesota-timberwolves", "new-orleans-pelicans", "new-york-knicks",
    "oklahoma-city-thunder", "orlando-magic", "philadelphia-76ers", "phoenix-suns",
    "portland-trail-blazers", "sacramento-kings", "san-antonio-spurs", "toronto-raptors",
    "utah-jazz", "washington-wizards",
]


def get_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


def scrape_team(driver, game, team_slug):
    url = BASE_URL.format(game=game, team=team_slug)
    driver.get(url)
    time.sleep(3)

    players = []
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr"))
        )
        rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            data = [c.text.strip() for c in cells]
            if len(data) >= 3 and data[2].isdigit():
                players.append({
                    "RANK":        data[0],
                    "PLAYER_NAME": data[1],
                    "RATING":  int(data[2]),
                    "TEAM":        team_slug.replace("-", " ").title(),
                })
    except Exception as e:
        print(f"    Warning: {e}")

    return players


def main():
    print("NBA 2K Multi-Season Ratings Scraper")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for game, season_year in GAME_TO_SEASON.items():

        output_path = os.path.join(OUTPUT_DIR, f"ratings_{game}.csv")

        if os.path.exists(output_path):
            print(f"\n[{game}] Already scraped — skipping")
            continue

        print(f"\n[{game}] Season year: {season_year}")
        all_players = {}

        # Fresh driver per season to prevent memory buildup
        driver = get_driver()

        try:
            for i, team in enumerate(NBA_TEAMS):
                team_name = team.replace("-", " ").title()

                # Restart driver every 10 teams
                if i > 0 and i % 10 == 0:
                    print("  Restarting browser...")
                    driver.quit()
                    time.sleep(2)
                    driver = get_driver()

                players = scrape_team(driver, game, team)
                print(f"  [{i+1:2d}/30] {team_name}: {len(players)} players")

                for p in players:
                    if p["PLAYER_NAME"] not in all_players:
                        all_players[p["PLAYER_NAME"]] = p

                time.sleep(1)

        finally:
            driver.quit()

        df = pd.DataFrame(all_players.values())
        df["GAME_VERSION"] = game
        df["SEASON"]  = season_year
        df.to_csv(output_path, index=False)
        print(f"  Saved {len(df)} unique players to {output_path}")
        time.sleep(2)

    print("\nScraping complete. CSVs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()