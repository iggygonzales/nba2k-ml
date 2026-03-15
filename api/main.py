"""
NBA 2K Rating Predictor — FastAPI
----------------------------------
Serves XGBoost model predictions via REST API.

Run locally:
    uvicorn api.main:app --reload

Endpoints:
    GET  /                      — health check
    GET  /search/{query}        — search for players by name
    GET  /player/{name}         — predict rating for a player from the DB
    GET  /predict/2k27/{name}   — predict 2K27 rating from 2025-26 stats
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
from dotenv import load_dotenv

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

print(f"Model loaded. Features: {FEATURES}")


# ── DB connection ─────────────────────────────────────────────────────────────

def get_engine():
    return create_engine(os.getenv(
        "DATABASE_URL",
        "postgresql://nba2k_user:nba2k_pass@localhost:5432/nba2k_db"
    ))


def enable_unaccent(engine):
    """Enable unaccent extension if not already enabled."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS unaccent"))
        conn.commit()


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_python(val):
    """Convert numpy types to native Python for JSON serialization."""
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
    """Fill missing delta features with 0 and ovr_prev with 75."""
    for col in ["pts_delta", "reb_delta", "ast_delta"]:
        if pd.isna(row.get(col)):
            row[col] = 0.0
    if pd.isna(row.get("ovr_prev")):
        row["ovr_prev"] = 75.0
    return row


def run_prediction(row):
    """Run model prediction and clamp to valid OVR range."""
    input_df = pd.DataFrame([row])[FEATURES]
    pred = float(model.predict(input_df)[0])
    return max(67.0, min(99.0, pred))


def find_player_name(engine, query: str):
    """
    Find exact player name using unaccent for accent-insensitive matching.
    'Jokic' matches 'Nikola Jokić', 'Doncic' matches 'Luka Dončić', etc.
    """
    df = pd.read_sql("""
        SELECT full_name FROM players
        WHERE unaccent(LOWER(full_name)) LIKE unaccent(LOWER(%(q)s))
        LIMIT 1
    """, engine, params={"q": f"%{query}%"})
    if df.empty:
        return None
    return df.iloc[0]["full_name"]


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


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    """Enable unaccent extension on startup."""
    try:
        enable_unaccent(get_engine())
        print("unaccent extension enabled")
    except Exception as e:
        print(f"Warning: could not enable unaccent: {e}")


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
        ORDER BY full_name
        LIMIT 10
    """, engine, params={"q": f"%{query}%"})
    return {"results": df["full_name"].tolist()}


@app.post("/predict")
def predict(stats: PlayerStats):
    """Predict 2K OVR rating from raw player stats."""
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
    """Look up a player from the DB and predict their rating."""
    engine = get_engine()

    exact_name = find_player_name(engine, player_name)
    if not exact_name:
        raise HTTPException(status_code=404,
            detail=f"Player '{player_name}' not found")

    df = pd.read_sql("""
        SELECT * FROM ml_dataset
        WHERE player_name = %(name)s
        AND season = %(season)s
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


@app.get("/predict/2k27/{player_name}")
def predict_2k27(player_name: str):
    """Predict a player's 2K27 rating using their 2025-26 stats."""
    engine = get_engine()

    exact_name = find_player_name(engine, player_name)
    if not exact_name:
        raise HTTPException(status_code=404,
            detail=f"Player '{player_name}' not found")

    df = pd.read_sql("""
        SELECT * FROM ml_dataset
        WHERE player_name = %(name)s
        AND split = 'predict'
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