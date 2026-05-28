"""Automated integration tests for the BlitzMate REST API (FastAPI backend)."""

import pytest
from fastapi.testclient import TestClient
from server.app.main import app


@pytest.fixture(scope="module")
def client():
    """Lifespan-aware TestClient fixture."""
    with TestClient(app) as c:
        yield c


def test_health(client):
    """GET /health -> 200, status="ok", max_depth=6."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["engine"] == "BlitzMate"
    assert data["max_depth"] == 6


def test_best_move_valid_fen(client):
    """POST /api/best-move starting position, depth=2."""
    payload = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "depth": 2,
    }
    response = client.post("/api/best-move", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["move"] is not None
    assert isinstance(data["move"], str)
    assert data["san"] is not None
    assert data["game_over"] is False
    assert data["status"] == "white_to_move"
    assert data["depth"] == 2
    assert "score_cp" in data
    assert "nodes" in data
    assert "time_ms" in data


def test_best_move_invalid_fen(client):
    """POST /api/best-move invalid FEN -> HTTP 400."""
    payload = {
        "fen": "invalid FEN string",
        "depth": 2,
    }
    response = client.post("/api/best-move", json=payload)
    assert response.status_code == 400
    assert "detail" in response.json()


def test_best_move_depth_too_high(client):
    """POST /api/best-move depth > 6 -> HTTP 422 (Pydantic validation)."""
    payload = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "depth": 7,
    }
    response = client.post("/api/best-move", json=payload)
    assert response.status_code == 422


def test_best_move_checkmate_position(client):
    """POST /api/best-move checkmate FEN -> 200, move=null, game_over=true, status="checkmate"."""
    # Scholar's mate checkmate position (Queen on f7 defended by Bishop on c4)
    checkmate_fen = "r1bqkbnr/pppp1Qpp/2n5/4p3/2B1P3/8/PPPP1PPP/RNB1KBNR b KQkq - 4 4"
    payload = {
        "fen": checkmate_fen,
        "depth": 2,
    }
    response = client.post("/api/best-move", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["move"] is None
    assert data["san"] is None
    assert data["game_over"] is True
    assert data["status"] == "checkmate"


def test_play_valid_move(client):
    """POST /api/play starting position + legal user move -> 200, legal=True, engine responds."""
    payload = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "user_move": "e2e4",
        "depth": 2,
    }
    response = client.post("/api/play", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["legal"] is True
    assert data["game_over"] is False
    assert data["engine_move"] is not None
    assert data["engine_san"] is not None
    assert data["fen"] is not None
    assert data["status"] in ("white_to_move", "black_to_move", "check")
    assert data["error"] is None


def test_play_illegal_move(client):
    """POST /api/play starting position + illegal but parseable move -> 200, legal=false, error set."""
    payload = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "user_move": "e2e5",
        "depth": 2,
    }
    response = client.post("/api/play", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["legal"] is False
    assert data["engine_move"] is None
    assert data["engine_san"] is None
    assert data["game_over"] is False
    assert data["error"] is not None


def test_play_invalid_uci(client):
    """POST /api/play starting position + completely invalid UCI -> 400."""
    payload = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "user_move": "invalid_move",
        "depth": 2,
    }
    response = client.post("/api/play", json=payload)
    assert response.status_code == 400
    assert "detail" in response.json()


def test_play_user_checkmates(client):
    """POST /api/play user checkmates -> game_over=true, engine_move=null."""
    # Fool's Mate checkmate position setup (Black checkmates White with Qh4)
    fen = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
    # Black plays Qh4#
    payload = {
        "fen": fen,
        "user_move": "d8h4",
        "depth": 2,
    }
    response = client.post("/api/play", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["legal"] is True
    assert data["game_over"] is True
    assert data["status"] == "checkmate"
    assert data["engine_move"] is None
    assert data["engine_san"] is None
    assert data["error"] is None
