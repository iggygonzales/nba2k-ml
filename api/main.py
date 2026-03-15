"""
NBA 2K Rating Predictor — FastAPI
----------------------------------
Run locally:
    uvicorn api.main:app --reload

Endpoints:
    GET  /                      — health check
    GET  /search/{query}        — search players by name
    GET  /player/{name}         — predict rating for a player
    GET  /predict/2k27/{name}   — predict 2K27 rating from 2025-26 stats
    GET  /ask?q={question}      — natural language query
    POST /predict               — predict rating from raw stats
"""

import pickle
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import create_engine, text
from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv()

app = FastAPI(
    title="NBA 2K Rating Predictor",
    description="Predicts NBA 2K player ratings from real NBA stats using XGBoost",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model ────────────────────────────────────────────────────────────────

MODEL_PATH    = os.path.join("api", "xgboost_model.pkl")
FEATURES_PATH = os.path.join("api", "features.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(FEATURES_PATH, "rb") as f:
    FEATURES = pickle.load(f)

anthropic_client = Anthropic()

print(f"Model loaded. Features: {FEATURES}")

# ── LLM Schema ────────────────────────────────────────────────────────────────

SCHEMA = """
You are an expert NBA and NBA 2K analyst with access to a PostgreSQL database.

Database schema — ml_dataset view (use this for ALL queries):
    player_id, player_name, season, game_version, season_year,
    ovr_rating, is_rookie, age, gp, min,
    pts, reb, ast, stl, blk, tov,
    fg_pct, fg3_pct, ft_pct, net_rating, usg_pct, ast_pct, ast_to, pie,
    career_year, pts_delta, reb_delta, ast_delta, ovr_prev, ovr_delta,
    split (values: 'train', 'test', 'predict')

Key rules:
- ALWAYS filter WHERE ovr_rating IS NOT NULL unless specifically asking about predict rows
- split = 'predict' means 2025-26 season stats with no 2K rating yet (future 2K27)
- split = 'test' means 2K25 and 2K26 seasons
- split = 'train' means 2K20 through 2K24 seasons
- game_version = 'nba-2k26' is the latest rated season
- For player searches: unaccent(LOWER(player_name)) LIKE unaccent(LOWER('%name%'))
- Always ORDER BY and LIMIT results to 20 unless asked for more
- Return ONLY valid PostgreSQL SQL — no markdown, no backticks, no explanation

Examples:
Q: Top 10 players in 2K26
A: SELECT player_name, ovr_rating, pts, reb, ast FROM ml_dataset WHERE game_version = 'nba-2k26' AND ovr_rating IS NOT NULL ORDER BY ovr_rating DESC LIMIT 10

Q: Most underrated player in 2K26
A: SELECT player_name, ovr_rating, pts, usg_pct FROM ml_dataset WHERE game_version = 'nba-2k26' AND ovr_rating IS NOT NULL ORDER BY (pts - ovr_rating) DESC LIMIT 10

Q: Predicted 2K27 ratings
A: SELECT player_name, ovr_prev as last_2k26_rating, pts, reb, ast, age FROM ml_dataset WHERE split = 'predict' AND ovr_prev IS NOT NULL ORDER BY ovr_prev DESC LIMIT 10
"""


# ── DB connection ─────────────────────────────────────────────────────────────

def get_engine():
    return create_engine(os.getenv(
        "DATABASE_URL",
        "postgresql://nba2k_user:nba2k_pass@localhost:5432/nba2k_db"
    ))


def enable_unaccent(engine):
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS unaccent"))
        conn.commit()


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_python(val):
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if pd.isna(val):
        return None
    return val


def fill_deltas(row):
    for col in ["pts_delta", "reb_delta", "ast_delta"]:
        if pd.isna(row.get(col)):
            row[col] = 0.0
    if pd.isna(row.get("ovr_prev")):
        row["ovr_prev"] = 75.0
    return row


def run_prediction(row):
    input_df = pd.DataFrame([row])[FEATURES]
    pred = float(model.predict(input_df)[0])
    return max(67.0, min(99.0, pred))


def find_player_name(engine, query: str):
    df = pd.read_sql("""
        SELECT full_name FROM players
        WHERE unaccent(LOWER(full_name)) LIKE unaccent(LOWER(%(q)s))
        LIMIT 1
    """, engine, params={"q": f"%{query}%"})
    if df.empty:
        return None
    return df.iloc[0]["full_name"]


def add_2k27_predictions(df, engine):
    """Run model on predict-split players and add predicted_2k27 column."""
    if "player_name" not in df.columns:
        return df

    predict_df = pd.read_sql("""
        SELECT * FROM ml_dataset
        WHERE split = 'predict'
        AND player_name = ANY(%(names)s)
    """, engine, params={"names": df["player_name"].tolist()})

    if predict_df.empty:
        return df

    for col in ["pts_delta", "reb_delta", "ast_delta"]:
        predict_df[col] = predict_df[col].fillna(0.0)
    predict_df["ovr_prev"] = predict_df["ovr_prev"].fillna(75.0)

    preds = model.predict(predict_df[FEATURES])
    predict_df["predicted_2k27"] = np.clip(preds, 67, 99).round(1)

    df = df.merge(
        predict_df[["player_name", "predicted_2k27"]],
        on="player_name",
        how="left"
    )
    return df


def is_prediction_question(question: str) -> bool:
    keywords = ["2k27", "predict", "next year", "2025-26", "future rating"]
    return any(k in question.lower() for k in keywords)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    try:
        enable_unaccent(get_engine())
        print("unaccent extension enabled")
    except Exception as e:
        print(f"Warning: could not enable unaccent: {e}")


# ── Request schema ────────────────────────────────────────────────────────────

class PlayerStats(BaseModel):
    pts:         float
    reb:         float
    ast:         float
    stl:         float
    blk:         float
    tov:         float
    fg_pct:      float
    fg3_pct:     float
    ft_pct:      float
    net_rating:  float
    usg_pct:     float
    ast_pct:     float
    pie:         float
    age:         float
    gp:          int
    career_year: int
    pts_delta:   Optional[float] = 0.0
    reb_delta:   Optional[float] = 0.0
    ast_delta:   Optional[float] = 0.0
    ovr_prev:    Optional[float] = 75.0


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {
        "status":  "online",
        "model":   "XGBoost NBA 2K Rating Predictor",
        "version": "1.0.0"
    }


@app.get("/search/{query}")
def search_players(query: str):
    engine = get_engine()
    df = pd.read_sql("""
        SELECT DISTINCT full_name FROM players
        WHERE unaccent(LOWER(full_name)) LIKE unaccent(LOWER(%(q)s))
        ORDER BY full_name LIMIT 10
    """, engine, params={"q": f"%{query}%"})
    return {"results": df["full_name"].tolist()}


@app.post("/predict")
def predict(stats: PlayerStats):
    input_df = pd.DataFrame([stats.dict()])[FEATURES]
    pred = float(model.predict(input_df)[0])
    pred = max(67.0, min(99.0, pred))
    return {
        "predicted_ovr": round(pred, 1),
        "rounded_ovr":   round(pred),
        "confidence":    "high",
        "model":         "XGBoost"
    }


@app.get("/player/{player_name}")
def predict_for_player(player_name: str, season: Optional[str] = "2024-25"):
    engine = get_engine()

    exact_name = find_player_name(engine, player_name)
    if not exact_name:
        raise HTTPException(status_code=404,
            detail=f"Player '{player_name}' not found")

    df = pd.read_sql("""
        SELECT * FROM ml_dataset
        WHERE player_name = %(name)s AND season = %(season)s
    """, engine, params={"name": exact_name, "season": season})

    if df.empty:
        raise HTTPException(status_code=404,
            detail=f"No data for '{exact_name}' in season {season}")

    row = df.iloc[0].copy()
    row = fill_deltas(row)
    pred = run_prediction(row)
    actual = to_python(row["ovr_rating"])

    return {
        "player":        str(exact_name),
        "season":        str(season),
        "predicted_ovr": round(pred, 1),
        "rounded_ovr":   round(pred),
        "actual_ovr":    actual,
        "error":         round(abs(pred - actual), 1) if actual is not None else None,
        "stats": {
            "pts":     to_python(row["pts"]),
            "reb":     to_python(row["reb"]),
            "ast":     to_python(row["ast"]),
            "stl":     to_python(row["stl"]),
            "blk":     to_python(row["blk"]),
            "usg_pct": to_python(row["usg_pct"]),
        },
        "model": "XGBoost"
    }


@app.get("/player/{player_name}/history")
def player_history(player_name: str):
    """Get a player's full rating and stats history across all seasons."""
    engine = get_engine()

    exact_name = find_player_name(engine, player_name)
    if not exact_name:
        raise HTTPException(status_code=404,
            detail=f"Player '{player_name}' not found")

    df = pd.read_sql("""
        SELECT player_name, season, game_version, season_year,
               ovr_rating, pts, reb, ast, age, gp,
               ovr_prev, ovr_delta, career_year, split
        FROM ml_dataset
        WHERE player_name = %(name)s
        ORDER BY season_year
    """, engine, params={"name": exact_name})

    if df.empty:
        raise HTTPException(status_code=404,
            detail=f"No history found for '{exact_name}'")

    return {
        "player":  str(exact_name),
        "seasons": len(df),
        "history": [
            {k: to_python(v) for k, v in row.items()}
            for row in df.to_dict(orient="records")
        ]
    }


@app.get("/predict/2k27/{player_name}")
def predict_2k27(player_name: str):
    engine = get_engine()

    exact_name = find_player_name(engine, player_name)
    if not exact_name:
        raise HTTPException(status_code=404,
            detail=f"Player '{player_name}' not found")

    df = pd.read_sql("""
        SELECT * FROM ml_dataset
        WHERE player_name = %(name)s AND split = 'predict'
    """, engine, params={"name": exact_name})

    if df.empty:
        raise HTTPException(status_code=404,
            detail=f"No 2025-26 stats found for '{exact_name}'")

    row = df.iloc[0].copy()
    row = fill_deltas(row)
    pred = run_prediction(row)

    return {
        "player":             str(exact_name),
        "season":             "2025-26",
        "predicted_2k27_ovr": round(pred, 1),
        "rounded_ovr":        round(pred),
        "last_known_ovr":     to_python(row["ovr_prev"]),
        "current_stats": {
            "pts": to_python(row["pts"]),
            "reb": to_python(row["reb"]),
            "ast": to_python(row["ast"]),
            "age": to_python(row["age"]),
            "gp":  to_python(row["gp"]),
        },
        "model": "XGBoost",
        "note":  "Prediction based on 2025-26 NBA stats. 2K27 not yet released."
    }


@app.get("/ask")
def ask(q: str):
    """Answer a natural language question about the NBA 2K database."""
    engine = get_engine()

    # Step 1 — Generate SQL
    sql_response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=SCHEMA,
        messages=[{"role": "user", "content": q}]
    )
    sql = sql_response.content[0].text.strip()

    # Step 2 — Run SQL
    try:
        df = pd.read_sql(sql, engine)
    except Exception as e:
        raise HTTPException(status_code=400,
            detail=f"SQL Error: {e}\nGenerated SQL: {sql}")

    if df.empty:
        return {"question": q, "sql": sql, "answer": "No results found.", "data": []}

    # Step 3 — Add live 2K27 predictions if needed
    if is_prediction_question(q):
        df = add_2k27_predictions(df, engine)

    # Step 4 — Generate natural language answer
    answer_response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system="You are an NBA 2K analyst. Answer concisely and insightfully based on the data. When predicted_2k27 column is present, use those as the predicted ratings.",
        messages=[{
            "role": "user",
            "content": f"Question: {q}\n\nData:\n{df.to_string(index=False)}\n\nAnswer:"
        }]
    )

    return {
        "question": q,
        "sql":      sql,
        "answer":   answer_response.content[0].text.strip(),
        "data":     [{k: to_python(v) for k, v in row.items()}
                     for row in df.head(20).to_dict(orient="records")]
    }
