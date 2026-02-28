"""FastAPI REST interface for the engine."""

import threading

import chess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from engine.core.search import SearchEngine
from engine.core.bitboard_evaluator import BitboardEvaluator
from engine.config import CONFIG

app = FastAPI(title=CONFIG.ui.engine_name, version="1.0.0")

# Shared engine instance (preserves TT across requests).
engine = SearchEngine(BitboardEvaluator(), depth=CONFIG.search.depth)
board = chess.Board()
_board_lock = threading.Lock()


class FenRequest(BaseModel):
    fen: str


class MoveRequest(BaseModel):
    move: str  # UCI format e.g. "e2e4"


class SearchRequest(BaseModel):
    depth: Optional[int] = None


@app.get("/board")
def get_board():
    with _board_lock:
        return {
            "fen": board.fen(),
            "turn": "white" if board.turn == chess.WHITE else "black",
            "legal_moves": [m.uci() for m in board.legal_moves],
            "is_game_over": board.is_game_over(),
            "result": board.result() if board.is_game_over() else None,
        }


@app.post("/position")
def set_position(req: FenRequest):
    with _board_lock:
        try:
            board.set_fen(req.fen)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid FEN: {e}")
        return {"fen": board.fen()}


@app.post("/move")
def make_move(req: MoveRequest):
    with _board_lock:
        try:
            move = chess.Move.from_uci(req.move)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid UCI move: {req.move}")
        if move not in board.legal_moves:
            raise HTTPException(status_code=400, detail=f"Illegal move: {req.move}")
        board.push(move)
        return {"fen": board.fen(), "move": req.move}


@app.post("/search")
def search_move(req: SearchRequest = SearchRequest()):
    with _board_lock:
        if board.is_game_over():
            raise HTTPException(status_code=400, detail="Game is already over")
        depth = req.depth or CONFIG.search.depth
        engine.max_depth = depth
        search_board = board.copy()

    best, ponder, score = engine.search_best_move(search_board)
    return {
        "best_move": best.uci() if best else None,
        "ponder": ponder.uci() if ponder else None,
        "score": score,
        "fen": search_board.fen(),
    }


@app.post("/reset")
def reset_board():
    with _board_lock:
        board.reset()
        engine.tt.clear()
        return {"fen": board.fen()}
