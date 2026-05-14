"""
API Tests
---------
Run with: pytest tests/
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)


def test_health_check():
    """API should return online status."""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "online"
    assert r.json()["model"] == "XGBoost NBA 2K Rating Predictor"


def test_search_player():
    """Search should return results for a known player."""
    r = client.get("/search/lebron")
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) > 0
    assert any("LeBron" in name for name in results)


def test_search_accent_player():
    """Search should handle accent-insensitive matching."""
    r = client.get("/search/jokic")
    assert r.status_code == 200
    results = r.json()["results"]
    assert any("Jokić" in name for name in results)


def test_player_current_rating():
    """Should return current rating for a known player."""
    r = client.get("/player/LeBron James")
    assert r.status_code == 200
    data = r.json()
    assert data["player"] == "LeBron James"
    assert data["actual_ovr"] is not None
    assert data["predicted_ovr"] is not None


def test_player_not_found():
    """Should return 404 for unknown player."""
    r = client.get("/player/Fake Player Name XYZ")
    assert r.status_code == 404


def test_player_history():
    """Should return full career history."""
    r = client.get("/player/LeBron James/history")
    assert r.status_code == 200
    data = r.json()
    assert data["player"] == "LeBron James"
    assert data["seasons"] >= 1
    assert len(data["history"]) >= 1


def test_predict_2k27():
    """Should return 2K27 prediction for a known player."""
    r = client.get("/predict/2k27/LeBron James")
    assert r.status_code == 200
    data = r.json()
    assert "predicted_2k27_ovr" in data
    assert 67 <= data["rounded_ovr"] <= 99


def test_predict_2k27_not_found():
    """Should return 404 for unknown player."""
    r = client.get("/predict/2k27/Fake Player XYZ")
    assert r.status_code == 404


def test_predict_raw_stats():
    """Should return prediction from raw stats."""
    stats = {
        "pts": 25.0, "reb": 7.0, "ast": 8.0,
        "stl": 1.2, "blk": 0.5, "tov": 3.0,
        "fg_pct": 0.50, "fg3_pct": 0.35, "ft_pct": 0.75,
        "net_rating": 5.0, "usg_pct": 0.30, "ast_pct": 0.35,
        "pie": 0.15, "age": 28.0, "gp": 70,
        "career_year": 5, "pts_delta": 2.0,
        "reb_delta": 0.5, "ast_delta": 1.0, "ovr_prev": 88.0
    }
    r = client.post("/predict", json=stats)
    assert r.status_code == 200
    data = r.json()
    assert "predicted_ovr" in data
    assert 67 <= data["rounded_ovr"] <= 99