# NBA 2K ML — Predicting Player Ratings with Machine Learning

A full end-to-end data science and machine learning project that scrapes NBA 2K player ratings (2K20–2K26), joins them with real NBA stats from the official NBA API, stores everything in a PostgreSQL database, and trains ML models to predict 2K ratings and future player trajectories.

The project also predicts **NBA 2K27 ratings** using current 2025-26 season stats — before the game is released.

---

## Project Goals

- Build a complete data pipeline from raw web scraping to a production-ready database
- Train regression and classification models to understand what drives 2K ratings
- Deploy a FastAPI prediction endpoint wrapped in Docker
- Add a natural language query layer using LLMs so anyone can ask questions about the data in plain English

This project was built as a portfolio piece targeting data science, ML engineering, data engineering, and AI engineering roles.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Scraping | Selenium, webdriver-manager |
| NBA Stats | nba_api, requests |
| Data Processing | pandas, thefuzz |
| Database | PostgreSQL 16 (Docker), SQLAlchemy, psycopg2 |
| ML | scikit-learn, XGBoost, PyTorch, SHAP |
| API Serving | FastAPI, Docker |
| AI Layer | Claude / OpenAI API (natural language queries) |
| Environment | Python 3.11, Jupyter, VS Code |

---

## Project Structure

```
nba2k-ml/
│
├── data/
│   ├── raw/
│   │   ├── ratings/          # Scraped 2K ratings per season (CSV)
│   │   └── stats/            # NBA API stats per season (CSV)
│   └── processed/
│       └── joined_dataset.csv
│
├── scraper/
│   ├── scrape_ratings.py     # Selenium scraper — HoopsHype 2K20-2K26
│   └── fetch_nba_stats.py    # NBA API stats fetcher 2018-19 to 2025-26
│
├── pipeline/
│   ├── build_database.py     # Fuzzy join + bulk insert to Postgres
│   ├── verify_db.py          # Quick DB sanity checks
│   └── db.py                 # SQLAlchemy connection helper
│
├── notebooks/
│   ├── 01_exploration.ipynb  # EDA
│   ├── 02_features.ipynb     # Feature engineering
│   ├── 03_xgboost.ipynb      # XGBoost model
│   ├── 04_pytorch.ipynb      # PyTorch MLP
│   └── 05_llm_queries.ipynb  # LLM natural language layer
│
├── api/
│   └── main.py               # FastAPI prediction endpoint
│
├── db/
│   └── init.sql              # PostgreSQL schema
│
├── docker-compose.yml        # Postgres + pgAdmin
├── requirements.txt
├── .env.example
└── README.md
```

---

## Database Schema

```
players         — one row per unique player (player_id from NBA API)
seasons         — season reference table (2018-19 through 2025-26)
ratings         — 2K OVR rating per player per season (composite PK)
stats           — NBA per-game + advanced stats per player per season
features        — engineered features (deltas, career year, etc.)
ml_dataset      — view joining all tables, ready for ML
```

The `ml_dataset` view includes a `split` column:
- `train` — 2K20 through 2K24 (seasons 2018-19 to 2022-23)
- `test`  — 2K25 and 2K26 (seasons 2023-24 and 2024-25)
- `predict` — 2025-26 stats with no rating yet (2K27 predictions)

---

## Setup

### Prerequisites
- Python 3.11
- Docker Desktop
- Google Chrome

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/nba2k-ml.git
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
# Edit .env with your Postgres credentials
```

### 4. Start the database
```bash
docker-compose up -d
```

### 5. Scrape ratings and fetch stats
```bash
python scraper/scrape_ratings.py      # ~30 mins
python scraper/fetch_nba_stats.py     # ~5 mins
```

### 6. Build the database
```bash
python pipeline/build_database.py
```

### 7. Verify
```bash
python pipeline/verify_db.py
```

---

## Dataset

| Split | Rows | Seasons | Purpose |
|---|---|---|---|
| Train | ~2,743 | 2K20–2K24 | Model training |
| Test | ~1,141 | 2K25–2K26 | Model evaluation |
| Predict | ~558 | 2025-26 | 2K27 predictions |
| **Total** | **~4,442** | **8 seasons** | |

Stats per row: PTS, REB, AST, STL, BLK, TOV, FG%, 3P%, FT%, NET_RTG, USG%, AST%, AST/TO, PIE, AGE, GP, MIN

---

## Key Design Decisions

- **Team-by-team scraping** — HoopsHype paginates by team, bypassing JS pagination issues
- **Fuzzy name matching** — handles special characters (Luka Dončić, Nikola Jokić) and known aliases
- **Composite primary keys** — `(player_id, season_year)` in ratings and stats tables
- **Season-based train/test split** — train on past seasons, test on recent ones (no data leakage)
- **2025-26 predict rows** — stats with no rating, used to predict 2K27 before release
- **Bulk inserts** — SQLAlchemy `to_sql` instead of row-by-row inserts for performance

---

## Roadmap

- [x] Data pipeline (scraping + NBA API)
- [x] PostgreSQL database with proper schema
- [x] Fuzzy name matching and join
- [ ] EDA notebook
- [ ] Feature engineering (career year, YoY deltas, age curves)
- [ ] XGBoost regression model + SHAP explainability
- [ ] PyTorch MLP model
- [ ] FastAPI serving endpoint
- [ ] Docker containerization
- [ ] LLM natural language query layer
- [ ] 2K27 rating predictions

---
