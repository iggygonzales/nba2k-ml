# NBA 2K ML — Predicting Player Ratings with Machine Learning

![CI/CD](https://github.com/iggygonzales/nba2k-ml/actions/workflows/ci-cd.yml/badge.svg)

A full end-to-end data science and machine learning project that scrapes NBA 2K player ratings (2K20–2K26), joins them with real NBA stats from Basketball Reference, stores everything in a PostgreSQL database, trains ML models to predict 2K ratings, and serves predictions via a FastAPI + Streamlit dashboard.

The project predicts **NBA 2K27 ratings** using current 2025-26 season stats — before the game is released.

---

## Live Demo

🌐 **Dashboard:** https://nba2k-ml.streamlit.app
🔌 **API:** http://18.220.131.170:8000/docs

---

## Project Goals

- Build a complete data pipeline from raw web scraping to a production-ready database
- Train and compare regression models (XGBoost vs PyTorch) to understand what drives 2K ratings
- Deploy a FastAPI prediction endpoint containerized with Docker
- Build a Streamlit dashboard for interactive exploration
- Add a natural language query layer using the Claude API (text-to-SQL)
- Deploy the full stack to AWS EC2 with an automated CI/CD pipeline

This project was built as a portfolio piece targeting data science, ML engineering, data engineering, and AI engineering roles.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Scraping | Selenium, webdriver-manager |
| NBA Stats | Basketball Reference (via requests + pandas) |
| Data Processing | pandas, thefuzz |
| Database | PostgreSQL 16 (Docker), SQLAlchemy, psycopg2 |
| ML | scikit-learn, XGBoost, PyTorch, SHAP |
| API Serving | FastAPI, uvicorn |
| Containerization | Docker, Docker Compose |
| AI Layer | Anthropic Claude API (text-to-SQL) |
| Dashboard | Streamlit, Plotly |
| Cloud | AWS EC2 |
| CI/CD | GitHub Actions |
| Environment | Python 3.11, Jupyter, VS Code |

---

## Project Structure

```
nba2k-ml/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml         # GitHub Actions — test + deploy pipeline
│
├── data/
│   ├── raw/
│   │   ├── ratings/          # Scraped 2K ratings per season (CSV)
│   │   └── stats/            # Per-season stats CSVs (one file per season)
│   │       ├── nba_player_stats_2018-19.csv
│   │       ├── nba_player_stats_2019-20.csv
│   │       ├── ...
│   │       └── nba_player_stats_2025-26.csv
│   └── processed/
│       └── joined_dataset.csv
│
├── scraper/
│   ├── scrape_ratings.py     # Selenium scraper — HoopsHype 2K20-2K26
│   └── fetch_nba_stats.py    # Basketball Reference stats fetcher (2018-19 to 2025-26)
│
├── pipeline/
│   ├── build_database.py     # Fuzzy join + bulk insert to Postgres
│   ├── verify_db.py          # Quick DB sanity checks
│   └── db.py                 # SQLAlchemy connection helper
│
├── notebooks/
│   ├── 01_exploration.ipynb  # EDA — distributions, correlations, top players
│   ├── 02_features.ipynb     # Feature engineering
│   ├── 03_xgboost.ipynb      # XGBoost model + SHAP explainability
│   ├── 04_pytorch.ipynb      # PyTorch MLP neural network
│   └── 05_llm_queries.ipynb  # LLM natural language query layer
│
├── api/
│   ├── main.py               # FastAPI prediction endpoint
│   ├── streamlit_app.py      # Streamlit dashboard
│   ├── xgboost_model.pkl     # Trained XGBoost model
│   └── features.pkl          # Feature list
│
├── tests/
│   ├── __init__.py
│   └── test_api.py           # pytest test suite (9 tests)
│
├── db/
│   └── init.sql              # PostgreSQL schema
│
├── Dockerfile                # API container
├── docker-compose.yml        # Postgres + pgAdmin + API
├── requirements.txt          # Full local development dependencies
├── requirements-api.txt      # Lightweight dependencies for Docker container
├── .env.example
└── README.md
```

---

## Database Schema

```
players         — one row per unique player (name-generated integer ID)
seasons         — season reference table (2018-19 through 2025-26)
ratings         — 2K OVR rating per player per season (composite PK)
stats           — per-game + advanced stats per player per season
features        — engineered features (deltas, career year, etc.)
ml_dataset      — view joining all tables, ready for ML
```

The `ml_dataset` view includes a `split` column:
- `train`   — 2K20 through 2K24 (seasons 2018-19 to 2022-23)
- `test`    — 2K25 and 2K26 (seasons 2023-24 and 2024-25)
- `predict` — 2025-26 stats with no rating yet (2K27 predictions)

---

## Quick Start

### Prerequisites
- Python 3.11
- Docker Desktop
- Google Chrome

### 1. Clone the repo
```bash
git clone https://github.com/iggygonzales/nba2k-ml.git
cd nba2k-ml
```

### 2. Create virtual environment
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Mac/Linux
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Add your Postgres credentials and ANTHROPIC_API_KEY
```

### 4. Start the full stack
```bash
docker-compose up -d
```
Starts Postgres, pgAdmin, and the FastAPI server.

### 5. Scrape and build the database
```bash
python scraper/scrape_ratings.py      # ~30 mins (Selenium, scrapes HoopsHype)
python scraper/fetch_nba_stats.py     # ~5 mins (Basketball Reference)
python pipeline/build_database.py
```

`fetch_nba_stats.py` is incremental — it skips finished seasons already on disk and only refetches the current season, whose stats change as games are played.

### 6. Run the dashboard
```bash
streamlit run api/streamlit_app.py
```

Open `http://localhost:8501`

---

## Stats Data Source

Player stats are fetched from **Basketball Reference** (not the NBA API, which has become unreliable due to bot-blocking). The fetcher pulls three pages per season:

- `leagues/NBA_{year}_per_game.html` — points, rebounds, assists, shooting splits, etc.
- `leagues/NBA_{year}_advanced.html` — PER, USG%, AST%, BPM, TS%
- `leagues/NBA_{year}.html` — league averages (used to normalise PIE-equivalent stats)

Each season is saved as a separate CSV (`nba_player_stats_2018-19.csv` through `nba_player_stats_2025-26.csv`) so historical data is never re-fetched unnecessarily.

**NBA API → Basketball Reference column mapping:**

| Model feature | NBA API | Basketball Reference |
|---|---|---|
| pts, reb, ast, stl, blk, tov | identical | identical |
| fg_pct, fg3_pct, ft_pct | identical | FG%, 3P%, FT% |
| usg_pct, ast_pct | identical | USG%, AST% |
| net_rating | NET_RATING | BPM (Box Plus/Minus) |
| per | PIE | PER (Player Efficiency Rating) |

---

## CI/CD Pipeline

Every push triggers the GitHub Actions pipeline:

```
git push
    ↓
test job — installs deps, connects to DB, runs 9 pytest tests
    ↓
tests pass? ✅
    ↓
deploy job (main branch only) — SSHes into EC2, git pull, docker compose up --build
    ↓
live API updated automatically
```

- Pushes to any branch run tests only
- Pushes to `main` run tests first, then deploy if they pass
- Broken code never reaches production

---

## Dashboard Features

- **Player Lookup** — search any NBA player, see their current 2K26 rating, predicted 2K27 rating, career trajectory chart, and overrated/underrated indicator
- **Leaderboard** — top 10 rated players in 2K26, predicted 2K27 top 10, biggest rating movers in 2K26, predicted 2K27 risers and decliners
- **Ask Anything** — natural language queries powered by Claude (text-to-SQL), with quick question buttons

---

## API Endpoints

Base URL: `http://18.220.131.170:8000`

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/search/{query}` | Search players by name (accent-insensitive) |
| GET | `/player/{name}` | Predict rating for a player (current season) |
| GET | `/player/{name}/history` | Full career rating + stats history |
| GET | `/predict/2k27/{name}` | Predict 2K27 rating from 2025-26 stats |
| GET | `/leaderboard/2k27-movers` | Predicted biggest risers and decliners for 2K27 |
| GET | `/ask?q={question}` | Natural language query (text-to-SQL) |
| POST | `/predict` | Predict rating from raw stats JSON |

Interactive docs: `http://18.220.131.170:8000/docs`

### Example responses

**GET /player/Nikola Jokic**
```json
{
  "player": "Nikola Jokić",
  "season": "2024-25",
  "predicted_ovr": 96.7,
  "rounded_ovr": 97,
  "actual_ovr": 98,
  "error": 1.3,
  "model": "XGBoost"
}
```

**GET /predict/2k27/Luka Doncic**
```json
{
  "player": "Luka Dončić",
  "season": "2025-26",
  "predicted_2k27_ovr": 95.4,
  "rounded_ovr": 95,
  "last_known_ovr": 95,
  "note": "Prediction based on 2025-26 NBA stats. 2K27 not yet released."
}
```

**GET /ask?q=Who had the biggest rating jump in 2K26**
```json
{
  "question": "Who had the biggest rating jump in 2K26?",
  "sql": "SELECT player_name, ovr_rating, ovr_prev, ovr_delta ...",
  "answer": "Dyson Daniels and Jay Huff tied for the biggest jump at +9 points...",
  "data": [...]
}
```

---

## Model Results

| Model | MAE | R² | Notes |
|---|---|---|---|
| XGBoost | 1.23 | 0.933 | Primary model, best performance |
| PyTorch MLP | 1.57 | 0.904 | Neural net, 4 layers, 500 epochs |

**Top predictors (SHAP analysis):**
1. `pts` — scoring is the #1 driver, high scorers push ratings up massively
2. `ovr_prev` — last season's rating is the #2 driver. 2K is conservative — if you were rated 90 last year, you'll likely be near 90 this year
3. `net_rating` — measures their team’s point differential (points scored minus points allowed) per 100 possessions while that specific player is on the court
4. `per` — player efficiency rating, helps summarize a basketball player's overall per-minute box-score productivity into a single number
5. `gp` — games played matters, players who miss games get penalized

**Key findings:**
- XGBoost outperforms PyTorch on this dataset — expected with ~2,700 training rows
- 2K ratings are extremely stable year-over-year (avg 76.3–76.9 across all seasons)
- The model underestimates stars (reputation premium) and overestimates low-minute players


---

## Predicted 2K27 Ratings (Top 10)

Based on 2025-26 season stats:

| Player | 2K26 Rating | Predicted 2K27 | Change |
|---|---|---|---|
| Shai Gilgeous-Alexander | 98 | 97 | -1 |
| Nikola Jokić | 98 | 96 | -2 |
| Giannis Antetokounmpo | 97 | 96 | -1 |
| Luka Dončić | 95 | 96 | +1 |
| Victor Wembanyama | 94 | 95 | +1 |
| Kawhi Leonard | 92 | 95 | +3 |
| Cade Cunningham | 92 | 94 | +2 |
| Anthony Edwards | 95 | 93 | -2 |
| Stephen Curry | 94 | 93 | -1 |
| Donovan Mitchell | 93 | 93 | 0 |

---

## Dataset

| Split | Rows | Seasons | Purpose |
|---|---|---|---|
| Train | ~2,743 | 2K20–2K24 | Model training |
| Test | ~1,141 | 2K25–2K26 | Model evaluation |
| Predict | ~558 | 2025-26 | 2K27 predictions |
| **Total** | **~4,442** | **8 seasons** | |

Stats per row: PTS, REB, AST, STL, BLK, TOV, FG%, 3P%, FT%, BPM, USG%, AST%, AST/TO, PER, TS%, AGE, GP, MIN

---

## Key Design Decisions

- **Basketball Reference over NBA API** — NBA's official API began blocking automated requests; Basketball Reference provides the same stats reliably with no authentication required
- **Per-season CSV files** — stats saved as individual files per season (`nba_player_stats_2025-26.csv`) so historical data is never re-fetched; only the current in-progress season is refreshed each run
- **Incremental stats fetching** — skips finished seasons already on disk, always refreshes the current season whose stats change as games are played
- **PER over PIE** — PER (Player Efficiency Rating) replaces PIE as the player impact metric; PER is available natively from Basketball Reference and is a well-established industry standard
- **Name-generated player IDs** — player IDs are generated from sorted unique player names rather than relying on NBA API IDs, which are unavailable from Basketball Reference
- **Team-by-team scraping** — bypasses JS pagination issues on HoopsHype
- **Fuzzy name matching** — handles special characters (Luka Dončić, Nikola Jokić) and name variations across sources
- **unaccent PostgreSQL extension** — accent-insensitive API queries
- **Composite primary keys** — `(player_id, season_year)` in ratings and stats
- **Season-based train/test split** — no data leakage across years
- **Live 2K27 predictions** — model runs on latest stats every query, not cached
- **Text-to-SQL LLM layer** — Claude converts natural language to PostgreSQL
- **Bulk inserts** — SQLAlchemy `to_sql` for pipeline performance
- **XGBoost vs PyTorch** — both built and compared, XGBoost wins on accuracy
- **Streamlit caching** — leaderboard data cached 1hr, player data cached 5min
- **Separate API requirements** — `requirements-api.txt` excludes heavy packages (PyTorch, Jupyter) keeping Docker builds under 60 seconds
- **CI/CD with GitHub Actions** — tests block deploys, broken code never reaches production

---

## Roadmap

- [x] Data pipeline (scraping + Basketball Reference stats)
- [x] Per-season incremental stats fetching
- [x] PostgreSQL database with proper schema
- [x] Fuzzy name matching and join
- [x] EDA notebook
- [x] Feature engineering (career year, YoY deltas)
- [x] XGBoost regression model + SHAP explainability
- [x] PyTorch MLP model
- [x] FastAPI serving endpoint
- [x] Docker containerization
- [x] Accent-insensitive player search
- [x] LLM natural language query layer
- [x] 2K27 rating predictions
- [x] Predicted 2K27 movers leaderboard (risers + decliners)
- [x] Streamlit dashboard (player lookup, leaderboard, ask anything)
- [x] Cloud deployment (AWS EC2)
- [x] CI/CD pipeline (GitHub Actions — test + deploy)
- [ ] Position encoding as ML feature
- [ ] Deploy frontend to custom domain