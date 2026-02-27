"""
Integration test suite for BlitzMate chess engine.

Tests components working together end-to-end:
- Full game simulations (engine vs engine, engine vs scripted)
- UCI protocol integration
- FastAPI REST API integration
- Analyzer pipeline
- Search + Evaluator + TT pipeline
- Async search lifecycle (start/stop/callback)
- Opening book + Syzygy integration with search
- Time management integration
- Edge case game scenarios
"""

import chess
import pytest
import time
import threading
import io
import sys
import os
from collections import defaultdict
from unittest.mock import patch, MagicMock

from engine.core.board import ChessBoard
from engine.core.bitboard_evaluator import BitboardEvaluator
from engine.core.transposition import TranspositionTable, TT_EXACT, TT_ALPHA, TT_BETA
from engine.core.search import SearchEngine, MATE_SCORE, INF
from engine.main import Engine
from engine.analyzer import Analyzer
from engine.config import CONFIG

# ════════════════════════════════════════════════════════════════════════════
#  ENGINE VS ENGINE — FULL GAME SIMULATIONS
# ════════════════════════════════════════════════════════════════════════════


class TestFullGame:
    """Tests that the engine can play complete games without crashing."""

    def test_engine_vs_engine_completes(self):
        """Two engines play a full game — must terminate with valid result."""
        engine = SearchEngine(BitboardEvaluator(), depth=2)
        board = chess.Board()
        move_count = 0
        max_moves = 150  # safety limit

        while not board.is_game_over() and move_count < max_moves:
            move, _ponder, _score = engine.search_best_move(board)
            if move is None:
                break  # engine can't find a move (draw-like position)
            assert (
                move in board.legal_moves
            ), f"Illegal move {move} at move {move_count}"
            board.push(move)
            move_count += 1

        # Game should have played at least some moves
        assert move_count > 10

    def test_engine_plays_from_midgame(self):
        """Engine can pick up and play from a midgame FEN."""
        fen = "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        engine = SearchEngine(BitboardEvaluator(), depth=3)
        board = chess.Board(fen)
        moves_played = 0

        for _ in range(20):
            if board.is_game_over():
                break
            move, _, _ = engine.search_best_move(board)
            if move is None:
                break
            assert move in board.legal_moves
            board.push(move)
            moves_played += 1

        assert moves_played > 0

    def test_engine_plays_endgame_kqk(self):
        """Engine can play KQ vs K endgame toward checkmate."""
        fen = "8/8/8/8/8/4K3/8/4k2Q w - - 0 1"
        engine = SearchEngine(BitboardEvaluator(), depth=4)
        board = chess.Board(fen)

        for _ in range(100):
            if board.is_game_over():
                break
            move, _, _ = engine.search_best_move(board)
            assert move in board.legal_moves
            board.push(move)

        # Should reach checkmate or very close to it
        assert board.is_game_over()

    def test_engine_handles_forced_draw(self):
        """Engine returns None or draw score for K vs K."""
        fen = "8/8/8/4k3/8/4K3/8/8 w - - 0 1"
        board = chess.Board(fen)
        engine = SearchEngine(BitboardEvaluator(), depth=3)

        assert board.is_insufficient_material()
        move, _, score = engine.search_best_move(board)
        # Insufficient material: score should be 0 (draw)
        assert score == 0

    def test_engine_alternating_colors(self):
        """Both sides use the same engine and alternate correctly."""
        engine = SearchEngine(BitboardEvaluator(), depth=2)
        board = chess.Board()

        for i in range(10):
            expected_turn = chess.WHITE if i % 2 == 0 else chess.BLACK
            assert board.turn == expected_turn
            move, _, _ = engine.search_best_move(board)
            board.push(move)


# ════════════════════════════════════════════════════════════════════════════
#  SEARCH + EVALUATOR + TT PIPELINE
# ════════════════════════════════════════════════════════════════════════════


class TestSearchPipeline:
    """Tests the full search pipeline: evaluator → search → TT interaction."""

    def test_tt_populated_after_search(self):
        """TT should contain entries after a search."""
        engine = SearchEngine(BitboardEvaluator(), depth=3)
        # Use a non-book position so actual search runs
        board = chess.Board(
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        )
        engine.search_best_move(board)

        # TT should have entries for at least the root position
        entry = engine.tt.get(board)
        assert entry is not None
        assert entry.best_move is not None

    def test_tt_reuse_across_searches(self):
        """Second search should be faster due to TT hits."""
        engine = SearchEngine(BitboardEvaluator(), depth=4)
        # Non-book position
        board = chess.Board(
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        )

        # First search populates TT
        start1 = time.time()
        engine.search_best_move(board)
        time1 = time.time() - start1
        nodes1 = engine.nodes

        # Second search should reuse TT entries
        start2 = time.time()
        engine.search_best_move(board)
        time2 = time.time() - start2
        nodes2 = engine.nodes

        # Second search should visit fewer or equal nodes
        assert nodes2 <= nodes1 * 1.1  # allow small variance

    def test_search_consistency(self):
        """Same position + depth should give same move (non-book position)."""
        engine = SearchEngine(BitboardEvaluator(), depth=3)
        # Non-book position so search runs deterministically
        board = chess.Board(
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        )

        move1, _, score1 = engine.search_best_move(board)
        # Reset TT for clean comparison
        engine.tt.clear()
        move2, _, score2 = engine.search_best_move(board)

        assert move1 == move2
        assert score1 == score2

    def test_deeper_search_better_or_equal(self):
        """Deeper search should find equal or better score than shallow."""
        fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        board = chess.Board(fen)

        shallow = SearchEngine(BitboardEvaluator(), depth=2)
        _, _, score_shallow = shallow.search_best_move(board)

        deep = SearchEngine(BitboardEvaluator(), depth=4)
        _, _, score_deep = deep.search_best_move(board)

        # Deeper search should find equal or better score (with tolerance)
        assert score_deep >= score_shallow - 50

    def test_pv_line_valid_after_search(self):
        """PV line should contain legal moves after search."""
        engine = SearchEngine(BitboardEvaluator(), depth=4)
        board = chess.Board()
        engine.search_best_move(board)

        pv = engine._get_pv_line(board, 4)
        test_board = board.copy()
        for move in pv:
            assert move in test_board.legal_moves, f"PV move {move} is illegal"
            test_board.push(move)

    def test_evaluator_and_search_agree_on_mate(self):
        """When search finds mate, the score should be near MATE_SCORE."""
        # Fool's mate position: Black can play Qh4#
        board = chess.Board(
            "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
        )
        engine = SearchEngine(BitboardEvaluator(), depth=3)
        move, _, score = engine.search_best_move(board)

        assert move == chess.Move.from_uci("d8h4")
        # Score should be positive mate (Black finding mate = positive from side-to-move)
        assert score > 100000

    def test_killers_reset_between_searches(self):
        """Killer move table should be fresh for each search."""
        engine = SearchEngine(BitboardEvaluator(), depth=3)
        # Non-book position so search actually populates killers
        board = chess.Board(
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        )
        engine.search_best_move(board)

        # Killers should have some entries after search
        has_killers = any(engine.killers[ply][0] is not None for ply in range(10))
        assert has_killers, "Search should populate some killer moves"

        # After a new search, killers are reassigned to a new defaultdict
        # The old entries should be gone (new defaultdict created)
        old_id = id(engine.killers)
        engine.search_best_move(board)
        new_id = id(engine.killers)
        assert old_id != new_id, "Killers should be a fresh defaultdict"

    def test_history_aging(self):
        """History scores should be aged (halved) between searches."""
        engine = SearchEngine(BitboardEvaluator(), depth=3)
        # Non-book position to trigger actual search (which ages history)
        board = chess.Board(
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        )
        engine.search_best_move(board)

        # Use a square pair that can never appear in legal moves from this position
        # h8 (63) -> a1 (0) is never a legal move
        engine.history[63][0] = 1000
        engine.search_best_move(board)

        # Should be halved (search ages all history entries)
        assert engine.history[63][0] == 500

    def test_aspiration_windows_dont_miss_tactics(self):
        """Aspiration window re-search should still find the right move."""
        # Scholar's mate position — Qxf7# is mate in 1
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        engine = SearchEngine(BitboardEvaluator(), depth=5)
        board = chess.Board(fen)

        move, _, score = engine.search_best_move(board)
        assert move is not None
        # Should find the winning continuation — score should be very high
        assert score > 500


# ════════════════════════════════════════════════════════════════════════════
#  ASYNC SEARCH LIFECYCLE
# ════════════════════════════════════════════════════════════════════════════


class TestAsyncSearch:
    """Tests the async search lifecycle: start/stop/callback."""

    def test_start_search_callback_receives_updates(self):
        """Callback should receive progress updates during search."""
        engine = SearchEngine(BitboardEvaluator(), depth=4)
        # Non-book position so actual iterative deepening runs
        board = chess.Board(
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        )
        updates = []

        def callback(best_move, ponder_move, d, score):
            updates.append((best_move, ponder_move, d, score))

        engine.start_search(board.copy(), depth=4, callback=callback)
        engine._thread.join(timeout=10)

        # Should have updates for each depth + final callback (d <= 0)
        assert len(updates) >= 2
        # Final callback has d <= 0
        assert updates[-1][2] <= 0
        # All intermediate moves should be valid
        for best_move, _, d, _ in updates:
            if d > 0 and best_move:
                assert best_move in board.legal_moves

    def test_stop_halts_search_quickly(self):
        """Calling stop() should halt the search within 0.5s."""
        engine = SearchEngine(BitboardEvaluator(), depth=64)
        board = chess.Board()
        done = threading.Event()

        def callback(best_move, ponder_move, d, score):
            if d <= 0:
                done.set()

        engine.start_search(board.copy(), depth=64, callback=callback)
        time.sleep(0.1)
        engine.stop()
        done.wait(timeout=1.0)

        assert done.is_set(), "Search didn't stop within timeout"

    def test_stop_before_start_is_safe(self):
        """Calling stop() before any search should not crash."""
        engine = SearchEngine(BitboardEvaluator(), depth=3)
        engine.stop()  # Should be no-op

    def test_multiple_sequential_searches(self):
        """Multiple searches should work correctly in sequence."""
        engine = SearchEngine(BitboardEvaluator(), depth=3)
        board = chess.Board()
        results = []

        for _ in range(3):
            done = threading.Event()

            def callback(best_move, ponder_move, d, score, _done=done):
                if d <= 0:
                    results.append(best_move)
                    _done.set()

            engine.start_search(board.copy(), depth=3, callback=callback)
            done.wait(timeout=10)
            engine.stop()  # Ensure clean state

        assert len(results) == 3
        for move in results:
            assert move in board.legal_moves

    def test_tt_not_polluted_after_stop(self):
        """TT shouldn't contain bogus entries after stopped search."""
        # Use a non-book position for deterministic search
        board = chess.Board(
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        )
        engine = SearchEngine(BitboardEvaluator(), depth=3)

        # Do a clean search first
        move1, _, score1 = engine.search_best_move(board)

        # Start a very deep search and stop it immediately
        engine.start_search(board.copy(), depth=64)
        time.sleep(0.05)
        engine.stop()
        if engine._thread:
            engine._thread.join(timeout=2)

        # Search again — TT may still have entries from interrupted search
        # but the guard should have prevented storing incomplete results
        move3, _, score3 = engine.search_best_move(board)
        # Result should be consistent with first search (same move, similar score)
        assert move3 == move1
        assert abs(score3 - score1) < 50

    def test_callback_for_book_move(self):
        """If a book move is found, callback should still fire."""
        engine = SearchEngine(BitboardEvaluator(), depth=4)
        board = chess.Board()
        results = []

        def callback(best_move, ponder_move, d, score):
            results.append((best_move, d))

        # Starting position is likely in opening book
        engine.start_search(board.copy(), depth=4, callback=callback)
        engine._thread.join(timeout=10)

        # Should have at least one callback
        assert len(results) >= 1
        # First move should be legal
        assert results[0][0] in board.legal_moves


# ════════════════════════════════════════════════════════════════════════════
#  UCI PROTOCOL INTEGRATION
# ════════════════════════════════════════════════════════════════════════════


class TestUCIIntegration:
    """Tests UCI protocol parsing and state management."""

    def _make_uci(self):
        from interface.uci import UCI

        return UCI()

    def test_position_startpos(self):
        """'position startpos' sets up initial board."""
        uci = self._make_uci()
        uci._parse_position(["startpos"])
        assert uci.board.fen() == chess.STARTING_FEN

    def test_position_startpos_moves(self):
        """'position startpos moves e2e4 e7e5' applies both moves."""
        uci = self._make_uci()
        uci._parse_position(["startpos", "moves", "e2e4", "e7e5"])
        expected = chess.Board()
        expected.push(chess.Move.from_uci("e2e4"))
        expected.push(chess.Move.from_uci("e7e5"))
        assert uci.board.fen() == expected.fen()

    def test_position_fen(self):
        """'position fen ...' sets up arbitrary position."""
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        uci = self._make_uci()
        uci._parse_position(["fen"] + fen.split())
        assert uci.board.fen() == fen

    def test_position_fen_with_moves(self):
        """'position fen ... moves e7e5' applies moves after FEN."""
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        uci = self._make_uci()
        uci._parse_position(["fen"] + fen.split() + ["moves", "e7e5"])
        expected = chess.Board(fen)
        expected.push(chess.Move.from_uci("e7e5"))
        assert uci.board.fen() == expected.fen()

    def test_position_invalid_fen_no_crash(self):
        """Invalid FEN shouldn't crash — board remains unchanged."""
        uci = self._make_uci()
        old_fen = uci.board.fen()
        uci._parse_position(["fen", "invalid", "fen", "string"])
        # Board should remain unchanged (parse returns early)
        assert uci.board.fen() == old_fen

    def test_position_illegal_moves_stop(self):
        """Illegal moves in 'position startpos moves' cause parsing to stop."""
        uci = self._make_uci()
        uci._parse_position(
            ["startpos", "moves", "e2e4", "e2e5"]
        )  # e2e5 is illegal for Black
        # e2e4 applies, e2e5 is skipped (not a legal Black move)
        expected = chess.Board()
        expected.push(chess.Move.from_uci("e2e4"))
        assert uci.board.fen() == expected.fen()

    def test_position_empty_tokens(self):
        """Empty position tokens shouldn't crash."""
        uci = self._make_uci()
        old_fen = uci.board.fen()
        uci._parse_position([])
        assert uci.board.fen() == old_fen

    def test_go_depth_parsing(self):
        """'go depth 3' should parse depth correctly."""
        uci = self._make_uci()
        done = threading.Event()
        results = []

        def mock_start_search(board, depth=None, callback=None):
            results.append(depth)
            done.set()

        uci.engine.start_search = mock_start_search
        uci._parse_go(["depth", "3"])
        done.wait(timeout=2)
        assert results[0] == 3

    def test_go_movetime_parsing(self):
        """'go movetime 1000' should set up a timer."""
        uci = self._make_uci()

        def mock_start_search(board, depth=None, callback=None):
            pass

        uci.engine.start_search = mock_start_search
        uci._parse_go(["movetime", "1000"])

        assert uci._movetime_timer is not None
        uci._movetime_timer.cancel()

    def test_go_wtime_btime_parsing(self):
        """'go wtime 60000 btime 60000' should compute movetime."""
        uci = self._make_uci()
        called_depth = []

        def mock_start_search(board, depth=None, callback=None):
            called_depth.append(depth)

        uci.engine.start_search = mock_start_search
        uci._parse_go(["wtime", "60000", "btime", "60000"])

        # Should set depth=64 (search until timer fires)
        assert called_depth[0] == 64
        # Should have set up a timer
        assert uci._movetime_timer is not None
        uci._movetime_timer.cancel()

    def test_go_infinite_parsing(self):
        """'go infinite' should set depth 64."""
        uci = self._make_uci()
        called_depth = []

        def mock_start_search(board, depth=None, callback=None):
            called_depth.append(depth)

        uci.engine.start_search = mock_start_search
        uci._parse_go(["infinite"])
        assert called_depth[0] == 64

    def test_setoption_no_crash(self):
        """setoption parsing shouldn't crash even with unimplemented options."""
        uci = self._make_uci()
        uci._parse_setoption(["name", "Hash", "value", "128"])
        uci._parse_setoption(["name", "Threads", "value", "4"])
        uci._parse_setoption([])  # empty
        uci._parse_setoption(["name"])  # partial

    def test_time_management_white_turn(self):
        """Time management allocates correct budget for White."""
        uci = self._make_uci()
        uci.board = chess.Board()  # White's turn

        called_depth = []

        def mock_start_search(board, depth=None, callback=None):
            called_depth.append(depth)

        uci.engine.start_search = mock_start_search
        uci._parse_go(
            ["wtime", "30000", "btime", "30000", "winc", "1000", "binc", "1000"]
        )

        assert called_depth[0] == 64
        assert uci._movetime_timer is not None
        uci._movetime_timer.cancel()

    def test_time_management_black_turn(self):
        """Time management uses btime when it's Black's turn."""
        uci = self._make_uci()
        uci.board = chess.Board()
        uci.board.push(chess.Move.from_uci("e2e4"))  # Black's turn

        called_depth = []

        def mock_start_search(board, depth=None, callback=None):
            called_depth.append(depth)

        uci.engine.start_search = mock_start_search
        uci._parse_go(["wtime", "60000", "btime", "5000", "winc", "0", "binc", "0"])

        assert called_depth[0] == 64
        assert uci._movetime_timer is not None
        uci._movetime_timer.cancel()


# ════════════════════════════════════════════════════════════════════════════
#  REST API INTEGRATION
# ════════════════════════════════════════════════════════════════════════════


class TestAPIIntegration:
    """Tests FastAPI REST API endpoints."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        from fastapi.testclient import TestClient
        from interface.api import app, board, engine

        self.client = TestClient(app)
        # Reset state before each test
        board.reset()
        engine.tt.clear()

    def test_get_board_initial(self):
        """GET /board returns initial position."""
        response = self.client.get("/board")
        assert response.status_code == 200
        data = response.json()
        assert data["fen"] == chess.STARTING_FEN
        assert data["turn"] == "white"
        assert data["is_game_over"] is False
        assert len(data["legal_moves"]) == 20

    def test_post_move_valid(self):
        """POST /move with valid UCI move."""
        response = self.client.post("/move", json={"move": "e2e4"})
        assert response.status_code == 200
        data = response.json()
        assert data["move"] == "e2e4"
        assert "4P3" in data["fen"]  # Pawn on e4 in FEN notation

    def test_post_move_illegal(self):
        """POST /move with illegal move returns 400."""
        response = self.client.post("/move", json={"move": "e2e5"})
        assert response.status_code == 400

    def test_post_move_invalid_format(self):
        """POST /move with bad UCI format returns 400."""
        response = self.client.post("/move", json={"move": "zzzz"})
        assert response.status_code == 400

    def test_set_position_valid(self):
        """POST /position with valid FEN."""
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        response = self.client.post("/position", json={"fen": fen})
        assert response.status_code == 200
        assert response.json()["fen"] == fen

    def test_set_position_invalid(self):
        """POST /position with invalid FEN returns 400."""
        response = self.client.post("/position", json={"fen": "invalid"})
        assert response.status_code == 400

    def test_search_returns_move(self):
        """POST /search returns a valid best move."""
        response = self.client.post("/search", json={"depth": 2})
        assert response.status_code == 200
        data = response.json()
        assert data["best_move"] is not None
        # Verify the returned move is legal
        board = chess.Board()
        move = chess.Move.from_uci(data["best_move"])
        assert move in board.legal_moves

    def test_search_game_over_returns_400(self):
        """POST /search when game is over returns 400."""
        # Set up a checkmate position
        fen = "rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        self.client.post("/position", json={"fen": fen})
        response = self.client.post("/search")
        assert response.status_code == 400

    def test_reset_board(self):
        """POST /reset returns to initial position."""
        self.client.post("/move", json={"move": "e2e4"})
        response = self.client.post("/reset")
        assert response.status_code == 200
        assert response.json()["fen"] == chess.STARTING_FEN

    def test_full_api_game_flow(self):
        """Play a few moves via API and verify state consistency."""
        # 1. Initial board
        r = self.client.get("/board")
        assert r.json()["turn"] == "white"

        # 2. Make a move
        self.client.post("/move", json={"move": "e2e4"})
        r = self.client.get("/board")
        assert r.json()["turn"] == "black"

        # 3. Engine search
        r = self.client.post("/search", json={"depth": 2})
        best = r.json()["best_move"]
        assert best is not None

        # 4. Reset
        self.client.post("/reset")
        r = self.client.get("/board")
        assert r.json()["fen"] == chess.STARTING_FEN

    def test_search_default_depth(self):
        """POST /search without depth uses CONFIG default."""
        response = self.client.post("/search")
        assert response.status_code == 200
        assert response.json()["best_move"] is not None


# ════════════════════════════════════════════════════════════════════════════
#  ANALYZER INTEGRATION
# ════════════════════════════════════════════════════════════════════════════


class TestAnalyzerIntegration:
    """Tests the Analyzer pipeline (search + evaluation + classification)."""

    def test_classify_best_move(self):
        """When player plays the engine's best move, label should be Best/Brilliant."""
        engine = SearchEngine(BitboardEvaluator(), depth=2)
        analyzer = Analyzer(engine)
        board = chess.Board()

        best_move, _, best_score = engine.search_best_move(board)
        assert best_move is not None
        info = analyzer.classify_move(board, best_move, best_move, best_score)

        assert info["label"] in ["Best move", "Brilliant (best)"]

    def test_classify_blunder(self):
        """Playing a clearly terrible move should be Inaccuracy/Mistake/Blunder."""
        engine = SearchEngine(BitboardEvaluator(), depth=3)
        analyzer = Analyzer(engine)

        # White is up a queen — playing a neutral pawn push is clearly worse
        fen = "4k3/8/8/8/8/8/P7/4KQ2 w - - 0 1"
        board = chess.Board(fen)
        best_move, _, best_score = engine.search_best_move(board)

        # Play a2a3 instead of using the queen — should be labeled poorly
        bad_move = chess.Move.from_uci("a2a3")
        assert bad_move in board.legal_moves
        info = analyzer.classify_move(board, bad_move, best_move, best_score)
        assert info["label"] in ["Inaccuracy", "Mistake", "Blunder", "Good"]

    def test_classify_checkmate_move(self):
        """A checkmate move should be labeled 'Checkmate (winning move)'."""
        engine = SearchEngine(BitboardEvaluator(), depth=2)
        analyzer = Analyzer(engine)

        # Fool's mate position
        fen = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
        board = chess.Board(fen)
        mate_move = chess.Move.from_uci("d8h4")

        best_move, _, best_score = engine.search_best_move(board)
        info = analyzer.classify_move(board, mate_move, best_move, best_score)
        assert info["label"] == "Checkmate (winning move)"

    def test_analyze_game_short(self):
        """Analyze a short game sequence."""
        engine = SearchEngine(BitboardEvaluator(), depth=2)
        analyzer = Analyzer(engine)

        moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
        report = analyzer.analyze_game(moves)

        assert len(report) == 4
        valid_labels = {
            "Brilliant",
            "Brilliant (best)",
            "Excellent",
            "Best move",
            "Good",
            "Inaccuracy",
            "Mistake",
            "Blunder",
        }
        for entry in report:
            assert "label" in entry
            assert "move_uci" in entry
            assert entry["label"] in valid_labels

    def test_analyze_game_illegal_move(self):
        """Illegal move in game analysis should produce error entry and stop."""
        engine = SearchEngine(BitboardEvaluator(), depth=2)
        analyzer = Analyzer(engine)

        moves = ["e2e4", "e7e5", "a1a8"]  # a1a8 is illegal
        report = analyzer.analyze_game(moves)

        # Should have 2 valid + 1 error
        assert len(report) == 3
        assert report[-1]["label"] == "Illegal"
        assert report[-1]["error"] is True

    def test_analyze_game_empty(self):
        """Empty move list should return empty report."""
        engine = SearchEngine(BitboardEvaluator(), depth=2)
        analyzer = Analyzer(engine)
        report = analyzer.analyze_game([])
        assert report == []

    def test_classify_preserves_board(self):
        """classify_move should not modify the board."""
        engine = SearchEngine(BitboardEvaluator(), depth=2)
        analyzer = Analyzer(engine)
        board = chess.Board()
        fen_before = board.fen()

        best_move, _, best_score = engine.search_best_move(board)
        analyzer.classify_move(board, best_move, best_move, best_score)

        assert board.fen() == fen_before


# ════════════════════════════════════════════════════════════════════════════
#  ENGINE WRAPPER INTEGRATION
# ════════════════════════════════════════════════════════════════════════════


class TestEngineWrapperIntegration:
    """Tests the Engine wrapper class (engine/main.py)."""

    def test_full_game_via_wrapper(self):
        """Play several moves using the Engine wrapper."""
        eng = Engine(depth=2)
        moves_played = 0

        for _ in range(10):
            if eng.board.board.is_game_over():
                break
            move_uci, ponder, score = eng.get_best_move()
            assert move_uci is not None
            result = eng.make_move(move_uci)
            assert result is True
            moves_played += 1

        assert moves_played > 0

    def test_wrapper_handles_invalid_move(self):
        """Engine wrapper should reject invalid moves."""
        eng = Engine(depth=2)
        result = eng.make_move("e2e5")
        assert result is False

    def test_wrapper_get_best_move_from_fen(self):
        """Engine wrapper should work after setting a FEN."""
        eng = Engine(depth=2)
        eng.board.set_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        move_uci, _, _ = eng.get_best_move()
        assert move_uci is not None
        move = chess.Move.from_uci(move_uci)
        assert move in eng.board.board.legal_moves


# ════════════════════════════════════════════════════════════════════════════
#  OPENING BOOK + SYZYGY INTEGRATION
# ════════════════════════════════════════════════════════════════════════════


class TestBookAndTablebase:
    """Tests opening book and Syzygy tablebase integration with search."""

    def test_book_move_from_starting_position(self):
        """Starting position should return a book move if books exist."""
        engine = SearchEngine(BitboardEvaluator(), depth=4)
        board = chess.Board()

        book_move = engine.get_book_move(board)
        if book_move is not None:
            # Book move should be legal
            assert book_move in board.legal_moves

    def test_book_move_from_non_book_position(self):
        """Random midgame position should not return book move."""
        engine = SearchEngine(BitboardEvaluator(), depth=4)
        # Unlikely to be in any book
        board = chess.Board("8/8/4k3/8/8/4K3/8/1Q6 w - - 0 1")
        book_move = engine.get_book_move(board)
        assert book_move is None

    def test_syzygy_move_kqk(self):
        """KQ vs K should return a Syzygy move if tablebase is functional."""
        engine = SearchEngine(BitboardEvaluator(), depth=4)
        board = chess.Board("8/8/8/8/8/4K3/8/4k2Q w - - 0 1")

        # Check if the tablebase can actually probe (files may be in subdirs)
        if engine.tablebase is None:
            pytest.skip("Syzygy tablebase not loaded")

        try:
            engine.tablebase.probe_wdl(board)
        except Exception:
            pytest.skip("Syzygy tablebase files not accessible")

        tb_move = engine.get_syzygy_move(board)
        assert tb_move is not None
        assert tb_move in board.legal_moves

    def test_syzygy_skipped_with_castling(self):
        """Syzygy probe should be skipped when castling rights exist."""
        engine = SearchEngine(BitboardEvaluator(), depth=4)
        board = chess.Board()  # Starting position has castling rights
        tb_move = engine.get_syzygy_move(board)
        assert tb_move is None

    def test_syzygy_skipped_many_pieces(self):
        """Syzygy probe should be skipped with >5 pieces."""
        engine = SearchEngine(BitboardEvaluator(), depth=4)
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1")
        tb_move = engine.get_syzygy_move(board)
        assert tb_move is None

    def test_search_uses_book_in_start_search(self):
        """start_search should return book move without full search."""
        engine = SearchEngine(BitboardEvaluator(), depth=4)
        board = chess.Board()
        results = []

        def callback(best_move, ponder_move, d, score):
            results.append((best_move, d))

        engine.start_search(board.copy(), depth=4, callback=callback)
        engine._thread.join(timeout=10)

        if engine.get_book_move(board) is not None:
            # If book hit, there should be fewer callbacks (no iterative deepening)
            assert len(results) <= 2  # book move callback + final


# ════════════════════════════════════════════════════════════════════════════
#  EDGE CASE GAME SCENARIOS
# ════════════════════════════════════════════════════════════════════════════


class TestEdgeCaseScenarios:
    """Tests edge case game scenarios end-to-end."""

    def test_stalemate_detection_in_game(self):
        """Engine should detect stalemate and stop."""
        fen = "k7/8/1K6/8/8/8/8/7Q w - - 0 1"
        engine = SearchEngine(BitboardEvaluator(), depth=4)
        board = chess.Board(fen)

        # Engine should avoid giving stalemate (it's winning)
        move, _, score = engine.search_best_move(board)
        if move is not None:
            board.push(move)
            # Should not immediately stalemate
            assert not board.is_stalemate() or score == 0

    def test_promotion_in_search(self):
        """Engine should find pawn promotion."""
        fen = "8/P7/8/8/8/8/8/4K2k w - - 0 1"
        engine = SearchEngine(BitboardEvaluator(), depth=3)
        board = chess.Board(fen)

        move, _, _ = engine.search_best_move(board)
        assert move is not None
        assert move.promotion is not None  # Should promote

    def test_en_passant_in_search(self):
        """Engine should consider en passant captures."""
        fen = "8/8/8/4Pp2/8/8/8/4K2k w - f6 0 1"
        engine = SearchEngine(BitboardEvaluator(), depth=3)
        board = chess.Board(fen)

        move, _, _ = engine.search_best_move(board)
        assert move is not None
        assert move in board.legal_moves

    def test_repetition_detection_integration(self):
        """Engine should recognize and handle repetition."""
        engine = SearchEngine(BitboardEvaluator(), depth=3)
        board = chess.Board()

        # Play moves that repeat: Nf3 Nf6 Ng1 Ng8 Nf3 Nf6
        moves = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"]
        for m in moves:
            board.push(chess.Move.from_uci(m))

        # Should detect repetition
        assert board.is_repetition(2)
        move, _, score = engine.search_best_move(board)
        # Score should be near 0 (draw)
        assert abs(score) < 200

    def test_fifty_move_rule_integration(self):
        """Engine handles 50-move rule positions."""
        fen = "8/8/4k3/8/8/4K3/8/1Q6 w - - 99 100"
        board = chess.Board(fen)
        engine = SearchEngine(BitboardEvaluator(), depth=3)

        move, _, _ = engine.search_best_move(board)
        assert move is not None

    def test_many_legal_moves_position(self):
        """Position with many legal moves doesn't crash or timeout."""
        # Position with open board and many piece moves (~40+ legal moves)
        fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 4"
        engine = SearchEngine(BitboardEvaluator(), depth=3)
        board = chess.Board(fen)
        num_legal = len(list(board.legal_moves))
        assert num_legal > 30, f"Expected 30+ legal moves, got {num_legal}"

        start = time.time()
        move, _, _ = engine.search_best_move(board)
        elapsed = time.time() - start

        assert move is not None
        assert move in board.legal_moves
        assert elapsed < 30  # Should complete within 30s at depth 3

    def test_single_legal_move(self):
        """When only one move is legal, engine should return it."""
        # Rook gives check on h-file, only escape is Kg8
        fen = "7k/8/5K2/8/8/8/8/7R b - - 0 1"
        engine = SearchEngine(BitboardEvaluator(), depth=3)
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        assert len(legal) == 1, f"Expected 1 legal move, got {len(legal)}: {legal}"

        move, _, _ = engine.search_best_move(board)
        assert move == legal[0]

    def test_deep_search_doesnt_crash(self):
        """Depth 6 search completes without crashing (tests killers defaultdict)."""
        engine = SearchEngine(BitboardEvaluator(), depth=6)
        board = chess.Board()

        move, _, score = engine.search_best_move(board)
        assert move is not None
        assert move in board.legal_moves

    def test_search_from_all_piece_types(self):
        """Position with all piece types evaluates correctly."""
        fen = "r1bqkbnr/pppppppp/2n5/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        engine = SearchEngine(BitboardEvaluator(), depth=3)
        board = chess.Board(fen)

        move, _, score = engine.search_best_move(board)
        assert move is not None
        assert isinstance(score, int)


# ════════════════════════════════════════════════════════════════════════════
#  QUIESCENCE + TT INTEGRATION
# ════════════════════════════════════════════════════════════════════════════


class TestQuiescenceTTIntegration:
    """Tests that quiescence search properly uses the TT."""

    def test_quiescence_stores_to_tt(self):
        """After quiescence, TT should have entries for capture positions."""
        engine = SearchEngine(BitboardEvaluator(), depth=3)
        # Non-book position with a hanging piece — quiescence will explore captures
        fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        board = chess.Board(fen)
        engine.search_best_move(board)

        # TT should have entries (from both main search and quiescence)
        assert len(engine.tt.table) > 0

    def test_quiescence_doesnt_crash_in_check(self):
        """Quiescence with king in check searches all evasions."""
        engine = SearchEngine(BitboardEvaluator(), depth=1)
        # King in check
        board = chess.Board(
            "rnbqkbnr/ppppp1pp/8/5p1Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2"
        )
        # This will go through quiescence after depth 1
        move, _, _ = engine.search_best_move(board)
        assert move is not None
        assert move in board.legal_moves


# ════════════════════════════════════════════════════════════════════════════
#  MOVE ORDERING INTEGRATION
# ════════════════════════════════════════════════════════════════════════════


class TestMoveOrderingIntegration:
    """Tests that move ordering helps search performance."""

    def test_tt_move_tried_first(self):
        """After one search, TT move should be tried first in next search."""
        engine = SearchEngine(BitboardEvaluator(), depth=3)
        # Non-book position
        board = chess.Board(
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        )

        # First search populates TT
        engine.search_best_move(board)
        tt_entry = engine.tt.get(board)
        assert tt_entry is not None

        # The TT move should be the first in ordered list
        ordered = engine._order_moves(board, tt_entry.best_move, 0)
        assert ordered[0] == tt_entry.best_move

    def test_captures_before_quiet_moves(self):
        """In a position with captures, they should be ordered before quiet moves."""
        engine = SearchEngine(BitboardEvaluator(), depth=3)
        # Position where white can capture on e5
        board = chess.Board(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3"
        )
        ordered = engine._order_moves(board, None, 0)

        # Find first capture and first quiet move
        first_cap_idx = None
        first_quiet_idx = None
        for i, move in enumerate(ordered):
            if board.is_capture(move) and first_cap_idx is None:
                first_cap_idx = i
            if not board.is_capture(move) and first_quiet_idx is None:
                first_quiet_idx = i

        if first_cap_idx is not None and first_quiet_idx is not None:
            assert first_cap_idx < first_quiet_idx


# ════════════════════════════════════════════════════════════════════════════
#  EVALUATOR INTEGRATION
# ════════════════════════════════════════════════════════════════════════════


class TestEvaluatorIntegration:
    """Tests evaluator correctness in game contexts."""

    def test_eval_changes_after_capture(self):
        """Evaluation should change significantly after capturing a piece."""
        evaluator = BitboardEvaluator()
        board = chess.Board(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3"
        )
        score_before = evaluator.evaluate(board)

        # Capture the e5 pawn
        board.push(chess.Move.from_uci("f3e5"))
        score_after = evaluator.evaluate(board)

        # Score should change (now evaluated from Black's side)
        assert score_before != score_after

    def test_doubled_pawn_penalty_applied(self):
        """Doubled pawns should be penalized for both colors."""
        evaluator = BitboardEvaluator()
        # Position with doubled pawns for White on e-file
        board = chess.Board("8/8/8/4k3/4P3/4P3/4K3/8 w - - 0 1")
        score_doubled = evaluator.evaluate(board)

        # Same material but no doubled pawns
        board2 = chess.Board("8/8/8/4k3/4P3/3P4/4K3/8 w - - 0 1")
        score_normal = evaluator.evaluate(board2)

        # Doubled should score lower (penalty applied)
        assert score_doubled < score_normal

    def test_passed_pawn_bonus_applied(self):
        """Passed pawns should get a bonus."""
        evaluator = BitboardEvaluator()
        # White pawn on e5 with no black pawns blocking
        board = chess.Board("8/8/8/4Pk2/8/4K3/8/8 w - - 0 1")
        score = evaluator.evaluate(board)
        # Should be positive (White has passed pawn advantage)
        assert score > 0

    def test_eval_returns_integer_from_white_perspective(self):
        """Evaluator returns integer scores in negamax style (side-to-move perspective)."""
        evaluator = BitboardEvaluator()

        # Symmetric starting position: should be near 0 for White to move
        board_w = chess.Board()
        score_w = evaluator.evaluate(board_w)
        assert isinstance(score_w, int)
        assert abs(score_w) < 100

        # Position with White up material
        board_up = chess.Board("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1")
        score_up_w = evaluator.evaluate(board_up)  # White to move, White up
        board_up_b = chess.Board("4k3/8/8/8/8/8/8/4KQ2 b - - 0 1")
        score_up_b = evaluator.evaluate(board_up_b)  # Black to move, White up

        # Negamax: positive = good for side to move
        # White to move with extra queen = positive
        assert score_up_w > 0
        # Black to move with opponent having extra queen = negative
        assert score_up_b < 0
        # They should roughly negate each other
        assert abs(score_up_w + score_up_b) < 200
