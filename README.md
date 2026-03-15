# NBA 2K ML — Predicting Player Ratings with Machine Learning

A full end-to-end data science and machine learning project that scrapes NBA 2K player ratings (2K20–2K26), joins them with real NBA stats from the official NBA API, stores everything in a PostgreSQL database, and trains ML models to predict 2K ratings and future player trajectories.

The project also predicts **NBA 2K27 ratings** using current 2025-26 season stats — before the game is released.

---

## Project Goals

- Build a complete data pipeline from raw web scraping to a production-ready database
- Train regression models to understand what drives 2K ratings
- Deploy a FastAPI prediction endpoint
- Add a natural language query layer using LLMs

This project was built as a portfolio piece targeting data science, ML engineering, data engineering, and AI engineering roles.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Scraping | Selenium, webdriver-manager |
| NBA Stats | nba_api |
| Data Processing | pandas, thefuzz |
| Database | PostgreSQL 16 (Docker), SQLAlchemy, psycopg2 |
| ML | scikit-learn, XGBoost, PyTorch, SHAP |
| API Serving | FastAPI, uvicorn |
| AI Layer | Claude / OpenAI API (coming soon) |
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
│   ├── 01_exploration.ipynb  # EDA — distributions, correlations, top players
│   ├── 02_features.ipynb     # Feature engineering
│   ├── 03_xgboost.ipynb      # XGBoost model + SHAP explainability
│   ├── 04_pytorch.ipynb      # PyTorch MLP neural network
│   └── 05_llm_queries.ipynb  # LLM natural language layer (coming soon)
│
├── api/
│   ├── main.py               # FastAPI prediction endpoint
│   ├── xgboost_model.pkl     # Trained XGBoost model
│   ├── features.pkl          # Feature list
│   └── scaler.pkl            # StandardScaler for PyTorch
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

### 7. Run the API
```bash
uvicorn api.main:app --reload
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/search/{query}` | Search players by name (accent-insensitive) |
| GET | `/player/{name}` | Predict rating for a player (current season) |
| GET | `/predict/2k27/{name}` | Predict 2K27 rating from 2025-26 stats |
| POST | `/predict` | Predict rating from raw stats JSON |

Interactive docs available at `http://localhost:8000/docs`

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

---

## Model Results

| Model | MAE | R² | Notes |
|---|---|---|---|
| XGBoost | 1.16 | 0.942 | Primary model, best performance |
| PyTorch MLP | 1.51 | 0.909 | Neural net, 4 layers, 500 epochs |

**Top predictors (SHAP analysis):**
1. `pts` — scoring is by far the strongest driver (0.91 correlation)
2. `ovr_prev` — last season's rating, 2K is conservative year-over-year
3. `pie` — player impact estimate
4. `usg_pct` — usage rate
5. `ast` — assists

**Key insight:** XGBoost outperforms the neural network on this dataset, which is expected with ~2,700 training rows. Gradient boosting handles tabular data better than deep learning at this scale.

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
- **unaccent PostgreSQL extension** — accent-insensitive API queries work for all player names
- **Composite primary keys** — `(player_id, season_year)` in ratings and stats tables
- **Season-based train/test split** — train on past seasons, test on recent ones (no data leakage)
- **2025-26 predict rows** — stats with no rating, used to predict 2K27 before release
- **Bulk inserts** — SQLAlchemy `to_sql` instead of row-by-row inserts for performance
- **XGBoost vs PyTorch** — both models built and compared; XGBoost wins on accuracy, PyTorch demonstrates ML engineering depth

---

## Roadmap

- [x] Data pipeline (scraping + NBA API)
- [x] PostgreSQL database with proper schema
- [x] Fuzzy name matching and join
- [x] EDA notebook
- [x] Feature engineering (career year, YoY deltas)
- [x] XGBoost regression model + SHAP explainability
- [x] PyTorch MLP model
- [x] FastAPI serving endpoint
- [x] Accent-insensitive player search
- [ ] Docker containerization of API
- [ ] LLM natural language query layer
- [ ] 2K27 rating predictions dashboard

---

