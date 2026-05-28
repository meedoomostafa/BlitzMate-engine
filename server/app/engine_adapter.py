"""Thin adapter wrapping the BlitzMate engine core for the REST API.

Design decisions
----------------
- One shared ``SearchEngine`` instance (preserves transposition table).
- Searches run in ``asyncio.to_thread`` so the FastAPI event loop stays free.
- An ``asyncio.Lock`` serialises access to the engine (single-threaded search).
- ``stdout`` is suppressed during search because the engine prints UCI info
  lines that would pollute server logs.
- Results are cached in an in-memory TTL cache keyed on ``(fen, depth)``.
- Cooperative cancellation is achieved using ``threading.Timer`` and ``engine.stop()``.
"""

import asyncio
import contextlib
import io
import time
import threading
from typing import Optional, Tuple

import chess
from cachetools import TTLCache

from engine.core.bitboard_evaluator import BitboardEvaluator
from engine.core.search import SearchEngine, MATE_SCORE

from server.app.config import (
    CACHE_MAX_SIZE,
    CACHE_TTL_S,
    DEFAULT_DEPTH,
    MAX_DEPTH,
    REQUEST_TIMEOUT_S,
)


class EngineError(Exception):
    """Raised when the engine fails to produce a result."""


class EngineAdapter:
    """Async-safe wrapper around :class:`SearchEngine`."""

    def __init__(self) -> None:
        self._engine: Optional[SearchEngine] = None
        self._lock = asyncio.Lock()
        self._cache: TTLCache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL_S)



    def startup(self) -> None:
        """Initialise the engine (call once at server start)."""
        self._engine = SearchEngine(BitboardEvaluator(), depth=DEFAULT_DEPTH)

    def shutdown(self) -> None:
        """Clean up engine resources."""
        if self._engine is not None:
            self._engine.close()
            self._engine = None


    async def get_best_move(
        self,
        fen: str,
        depth: int = DEFAULT_DEPTH,
    ) -> dict:
        """Return the engine's best move for the given FEN."""
        self._validate_depth(depth)
        board = self._parse_fen(fen)

        if board.is_game_over():
            status = self._game_status(board)
            if board.is_checkmate():
                score = -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
            else:
                score = 0
            return {
                "move": None,
                "san": None,
                "depth": depth,
                "score_cp": score,
                "nodes": 0,
                "time_ms": 0,
                "game_over": True,
                "status": status,
            }

        cache_key = (fen, depth)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        async with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

            search_result = await self._search_with_timeout_under_lock(board, depth)
            search_result["game_over"] = False
            search_result["status"] = self._game_status(board)
            
            self._cache[cache_key] = search_result
            return search_result

    async def play_move(
        self,
        fen: str,
        user_move: str,
        depth: int = DEFAULT_DEPTH,
    ) -> dict:
        """Validate and apply the user's move, then return the engine's response."""
        self._validate_depth(depth)
        board = self._parse_fen(fen)

        try:
            move = chess.Move.from_uci(user_move)
        except (ValueError, chess.InvalidMoveError):
            raise ValueError(f"Invalid UCI move format: {user_move!r}")

        if move not in board.legal_moves:
            return {
                "fen": board.fen(),
                "engine_move": None,
                "engine_san": None,
                "legal": False,
                "game_over": board.is_game_over(),
                "status": self._game_status(board),
                "error": f"Move {user_move} is illegal in the current position.",
            }

        board.push(move)

        if board.is_game_over():
            return {
                "fen": board.fen(),
                "engine_move": None,
                "engine_san": None,
                "legal": True,
                "game_over": True,
                "status": self._game_status(board),
                "error": None,
            }

        async with self._lock:
            search_result = await self._search_with_timeout_under_lock(board, depth)

        engine_uci = search_result["move"]
        engine_san = search_result["san"]

        if engine_uci is None:
            raise EngineError("Engine failed to return a move.")

        engine_move = chess.Move.from_uci(engine_uci)
        board.push(engine_move)

        return {
            "fen": board.fen(),
            "engine_move": engine_uci,
            "engine_san": engine_san,
            "legal": True,
            "game_over": board.is_game_over(),
            "status": self._game_status(board),
            "error": None,
        }


    async def _search_with_timeout_under_lock(self, board: chess.Board, depth: int) -> dict:
        """Run search under lock, with cooperative cancellation and outer timeout."""
        assert self._engine is not None, "Engine not initialised"

        self._engine.max_depth = depth
        
        self._engine.nodes = 0
        
        self._engine._stop_event.clear()
        
        timer = threading.Timer(REQUEST_TIMEOUT_S, self._engine.stop)
        timer.start()
        
        grace_period = 2.0
        outer_timeout = REQUEST_TIMEOUT_S + grace_period
        
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._search_sync, board, depth),
                timeout=outer_timeout,
            )
            return result
        except asyncio.TimeoutError:
            self._engine.stop()
            raise asyncio.TimeoutError("Engine search timed out.")
        finally:
            timer.cancel()
            self._engine._stop_event.clear()

    def _search_sync(self, board: chess.Board, depth: int) -> dict:
        """Run a blocking search. Must be called from a thread."""
        assert self._engine is not None, "Engine not initialised"

        start = time.monotonic()

        with contextlib.redirect_stdout(io.StringIO()):
            best, _ponder, score = self._engine.search_best_move(board)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        uci = best.uci() if best is not None else None
        san = board.san(best) if best is not None else None
        nodes = self._engine.nodes

        return {
            "move": uci,
            "san": san,
            "depth": depth,
            "score_cp": score,
            "nodes": nodes,
            "time_ms": elapsed_ms,
        }

    @staticmethod
    def _validate_depth(depth: int) -> None:
        if depth < 1 or depth > MAX_DEPTH:
            raise ValueError(
                f"Depth must be between 1 and {MAX_DEPTH}, got {depth}."
            )

    @staticmethod
    def _parse_fen(fen: str) -> chess.Board:
        try:
            return chess.Board(fen)
        except ValueError as exc:
            raise ValueError(f"Invalid FEN: {exc}") from exc

    @staticmethod
    def _game_status(board: chess.Board) -> str:
        """Derive a human-readable game status string."""
        if board.is_checkmate():
            return "checkmate"
        if board.is_stalemate():
            return "stalemate"
        if board.is_insufficient_material():
            return "draw"
        if board.can_claim_draw():
            return "draw"
        if board.is_check():
            return "check"
        return "white_to_move" if board.turn == chess.WHITE else "black_to_move"
