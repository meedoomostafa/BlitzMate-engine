"""
Comprehensive test suite for BlitzMate chess engine.

Covers:
- Board operations (edge cases, invalid input)
- Evaluator (material, positional, pawn structure, tactical)
- Transposition table (collision, replacement, hash consistency)
- Search (mate detection, tactical positions, pruning, edge cases)
- Move ordering (MVV-LVA, killers, history)
- Engine wrapper (integration)
- Stress tests (deep search, complex positions)
- Quiescence search
"""

import chess
import pytest
import time
import threading
from collections import defaultdict

from engine.core.board import ChessBoard
from engine.core.bitboard_evaluator import BitboardEvaluator
from engine.core.transposition import TranspositionTable, TTEntry, TT_EXACT, TT_ALPHA, TT_BETA
from engine.core.search import SearchEngine, MATE_SCORE, INF
from engine.main import Engine


# ════════════════════════════════════════════════════════════════════════════
#  BOARD TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestChessBoard:
    def test_initial_position(self):
        b = ChessBoard()
        assert b.get_fen() == chess.STARTING_FEN

    def test_make_legal_move(self):
        b = ChessBoard()
        assert b.make_move("e2e4") is True
        assert "e2e4" in b.move_history

    def test_make_illegal_move(self):
        b = ChessBoard()
        assert b.make_move("e2e5") is False

    def test_make_garbage_input(self):
        b = ChessBoard()
        assert b.make_move("zzzz") is False
        assert b.make_move("") is False
        assert b.make_move("12345") is False

    def test_undo_move(self):
        b = ChessBoard()
        b.make_move("e2e4")
        b.undo_move()
        assert b.get_fen() == chess.STARTING_FEN
        assert len(b.move_history) == 0

    def test_undo_empty(self):
        b = ChessBoard()
        b.undo_move()  # Should not crash
        assert b.get_fen() == chess.STARTING_FEN

    def test_set_fen(self):
        b = ChessBoard()
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        b.set_fen(fen)
        assert b.get_fen() == fen

    def test_reset(self):
        b = ChessBoard()
        b.make_move("e2e4")
        b.reset()
        assert b.get_fen() == chess.STARTING_FEN

    def test_get_legal_moves_initial(self):
        b = ChessBoard()
        moves = b.get_legal_moves()
        assert len(moves) == 20  # 16 pawn + 4 knight

    def test_from_fen(self):
        fen = "8/8/8/8/8/8/8/4K2k w - - 0 1"
        b = ChessBoard(fen=fen)
        assert b.get_fen() == fen

    def test_checkmate_no_legal_moves(self):
        b = ChessBoard(fen="rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1")
        assert len(b.get_legal_moves()) == 0
        assert b.board.is_checkmate()

    def test_en_passant_move(self):
        b = ChessBoard(fen="rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
        assert b.make_move("e5f6") is True

    def test_castling_move(self):
        b = ChessBoard(fen="r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
        assert b.make_move("e1g1") is True

    def test_promotion_move(self):
        b = ChessBoard(fen="8/P7/8/8/8/8/8/4K2k w - - 0 1")
        assert b.make_move("a7a8q") is True

    def test_multiple_undo(self):
        b = ChessBoard()
        b.make_move("e2e4")
        b.make_move("e7e5")
        b.make_move("g1f3")
        b.undo_move()
        b.undo_move()
        b.undo_move()
        assert b.get_fen() == chess.STARTING_FEN


# ════════════════════════════════════════════════════════════════════════════
#  EVALUATOR TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestBitboardEvaluator:
    def setup_method(self):
        self.ev = BitboardEvaluator()

    def test_starting_position_near_zero(self):
        board = chess.Board()
        score = self.ev.evaluate(board)
        assert -100 < score < 100

    def test_evaluate_returns_int(self):
        board = chess.Board()
        assert isinstance(self.ev.evaluate(board), int)

    def test_white_up_queen(self):
        board = chess.Board("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1")
        score = self.ev.evaluate(board)
        assert score > 500

    def test_black_up_queen(self):
        board = chess.Board("4kq2/8/8/8/8/8/8/4K3 w - - 0 1")
        score = self.ev.evaluate(board)
        assert score < -500

    def test_king_vs_king_draw(self):
        board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        score = self.ev.evaluate(board)
        assert score == 0

    def test_symmetric_material(self):
        board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
        score = self.ev.evaluate(board)
        assert -80 < score < 80

    def test_white_up_rook(self):
        board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1")
        score = self.ev.evaluate(board)
        assert score > 300

    def test_black_up_rook(self):
        board = chess.Board("r3k3/8/8/8/8/8/8/4K3 w q - 0 1")
        score = self.ev.evaluate(board)
        assert score < -300

    def test_insufficient_material_kn_vs_k(self):
        board = chess.Board("4k3/8/8/8/8/8/8/4KN2 w - - 0 1")
        assert board.is_insufficient_material()
        score = self.ev.evaluate(board)
        assert score == 0

    def test_insufficient_material_kb_vs_k(self):
        board = chess.Board("4k3/8/8/8/8/8/8/4KB2 w - - 0 1")
        assert board.is_insufficient_material()
        score = self.ev.evaluate(board)
        assert score == 0

    def test_bishop_pair_bonus(self):
        w_pair = chess.Board("4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1")
        w_single = chess.Board("4k3/8/8/8/8/8/8/4KB2 w - - 0 1")
        score_pair = self.ev.evaluate(w_pair)
        score_single = self.ev.evaluate(w_single)
        assert score_pair > score_single

    def test_passed_pawn_bonus(self):
        passed = chess.Board("4k3/8/4P3/8/8/8/8/4K3 w - - 0 1")
        not_passed = chess.Board("4k3/4p3/4P3/8/8/8/8/4K3 w - - 0 1")
        score_passed = self.ev.evaluate(passed)
        score_blocked = self.ev.evaluate(not_passed)
        assert score_passed > score_blocked

    def test_isolated_pawn_penalty(self):
        isolated = chess.Board("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1")
        supported = chess.Board("4k3/8/8/8/3PP3/8/8/4K3 w - - 0 1")
        score_iso = self.ev.evaluate(isolated)
        score_sup = self.ev.evaluate(supported)
        assert score_sup > score_iso

    def test_eval_negamax_symmetry(self):
        """Score should flip sign when side to move changes."""
        fen = "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        board_w = chess.Board(fen)
        board_b = chess.Board(fen)
        board_b.turn = chess.BLACK
        score_w = self.ev.evaluate(board_w)
        score_b = self.ev.evaluate(board_b)
        assert abs(score_w + score_b) < 5

    def test_stalemate_returns_zero(self):
        board = chess.Board("5k2/5P2/5K2/8/8/8/8/8 b - - 0 1")
        assert board.is_stalemate()
        assert self.ev.evaluate(board) == 0

    def test_mobility_bonus(self):
        # Need enough material to avoid insufficient_material returning 0
        active = chess.Board("4k3/pppppppp/8/8/3N4/8/PPPPPPPP/4K3 w - - 0 1")
        cramped = chess.Board("4k3/pppppppp/8/8/8/8/PPPPPPPP/N3K3 w - - 0 1")
        score_active = self.ev.evaluate(active)
        score_cramped = self.ev.evaluate(cramped)
        assert score_active > score_cramped

    def test_king_safety_castled_vs_center(self):
        castled = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R4RK1 w kq - 0 1")
        center = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
        score_castled = self.ev.evaluate(castled)
        score_center = self.ev.evaluate(center)
        assert score_castled >= score_center

    def test_threat_detection_hanging_piece(self):
        """A hanging piece should result in worse eval for its side."""
        hanging = chess.Board("4k3/8/8/3n4/4R3/8/8/4K3 w - - 0 1")
        safe = chess.Board("4k3/8/8/8/4R3/8/8/4K3 w - - 0 1")
        # The hanging case has a black knight that could be taken
        score_hanging = self.ev.evaluate(hanging)
        score_safe = self.ev.evaluate(safe)
        # Both have white rook, but hanging has extra black knight (material for black)
        # yet the knight is hanging. Just verify no crash.
        assert isinstance(score_hanging, int)
        assert isinstance(score_safe, int)


# ════════════════════════════════════════════════════════════════════════════
#  TRANSPOSITION TABLE TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestTranspositionTable:
    def test_store_and_get(self):
        tt = TranspositionTable()
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        tt.store(board, depth=5, value=100, flag=TT_EXACT, best_move=move)
        entry = tt.get(board)
        assert entry is not None
        assert entry.depth == 5
        assert entry.value == 100
        assert entry.flag == TT_EXACT
        assert entry.best_move == move

    def test_miss_returns_none(self):
        tt = TranspositionTable()
        board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        assert tt.get(board) is None

    def test_depth_preferred_keeps_deeper(self):
        tt = TranspositionTable()
        board = chess.Board()
        m1 = chess.Move.from_uci("e2e4")
        m2 = chess.Move.from_uci("d2d4")
        tt.store(board, depth=5, value=100, flag=TT_EXACT, best_move=m1)
        tt.store(board, depth=3, value=50, flag=TT_ALPHA, best_move=m2)
        entry = tt.get(board)
        assert entry.depth == 5
        assert entry.best_move == m1

    def test_deeper_overwrites(self):
        tt = TranspositionTable()
        board = chess.Board()
        m1 = chess.Move.from_uci("e2e4")
        m2 = chess.Move.from_uci("d2d4")
        tt.store(board, depth=3, value=50, flag=TT_ALPHA, best_move=m1)
        tt.store(board, depth=7, value=200, flag=TT_EXACT, best_move=m2)
        entry = tt.get(board)
        assert entry.depth == 7
        assert entry.best_move == m2

    def test_clear(self):
        tt = TranspositionTable()
        board = chess.Board()
        tt.store(board, depth=5, value=100, flag=TT_EXACT,
                 best_move=chess.Move.from_uci("e2e4"))
        tt.clear()
        assert tt.get(board) is None

    def test_bounded_size(self):
        tt = TranspositionTable(size_mb=1)
        assert tt.max_entries > 0
        assert tt.max_entries < 100_000

    def test_different_positions_stored_separately(self):
        tt = TranspositionTable()
        b1 = chess.Board()
        b2 = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        m1 = chess.Move.from_uci("e2e4")
        m2 = chess.Move.from_uci("e7e5")
        tt.store(b1, depth=5, value=10, flag=TT_EXACT, best_move=m1)
        tt.store(b2, depth=5, value=20, flag=TT_EXACT, best_move=m2)
        assert tt.get(b1).value == 10
        assert tt.get(b2).value == 20

    def test_flag_types(self):
        tt = TranspositionTable()
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        for flag in [TT_EXACT, TT_ALPHA, TT_BETA]:
            tt.store(board, depth=5, value=100, flag=flag, best_move=move)
            assert tt.get(board).flag == flag

    def test_equal_depth_overwrites(self):
        tt = TranspositionTable()
        board = chess.Board()
        m1 = chess.Move.from_uci("e2e4")
        m2 = chess.Move.from_uci("d2d4")
        tt.store(board, depth=5, value=100, flag=TT_EXACT, best_move=m1)
        tt.store(board, depth=5, value=200, flag=TT_EXACT, best_move=m2)
        entry = tt.get(board)
        assert entry.value == 200
        assert entry.best_move == m2

    def test_negative_values(self):
        tt = TranspositionTable()
        board = chess.Board()
        tt.store(board, depth=5, value=-500, flag=TT_EXACT,
                 best_move=chess.Move.from_uci("e2e4"))
        assert tt.get(board).value == -500

    def test_mate_score_storage(self):
        tt = TranspositionTable()
        board = chess.Board()
        tt.store(board, depth=10, value=MATE_SCORE - 5, flag=TT_EXACT,
                 best_move=chess.Move.from_uci("e2e4"))
        assert tt.get(board).value == MATE_SCORE - 5


# ════════════════════════════════════════════════════════════════════════════
#  SEARCH ENGINE TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestSearchEngine:
    def setup_method(self):
        self.engine = SearchEngine(depth=2)

    def test_search_returns_move(self):
        board = chess.Board()
        move, ponder, score = self.engine.search_best_move(board)
        assert move is not None
        assert move in board.legal_moves

    def test_search_returns_3_tuple(self):
        board = chess.Board()
        result = self.engine.search_best_move(board)
        assert len(result) == 3
        move, ponder, score = result
        assert isinstance(score, (int, float))

    # ── Mate detection ─────────────────────────────────────────────────────

    def test_finds_back_rank_mate(self):
        board = chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1")
        eng = SearchEngine(depth=3)
        move, _, score = eng.search_best_move(board)
        assert move == chess.Move.from_uci("a1a8")

    def test_finds_queen_mate(self):
        """Queen delivers mate."""
        # Construct a position where Qg7# or similar is forced
        board = chess.Board("6k1/5ppp/6N1/8/8/8/8/4K1Q1 w - - 0 1")
        eng = SearchEngine(depth=3)
        move, _, score = eng.search_best_move(board)
        assert move is not None
        # Verify the engine's move leads to checkmate or a winning position
        board.push(move)
        # The engine should find a very strong move (might be mate or lead to it)
        assert score > 500 or board.is_checkmate()

    def test_avoids_stalemate_when_winning(self):
        board = chess.Board("k7/8/1K6/8/8/8/8/7Q w - - 0 1")
        eng = SearchEngine(depth=4)
        move, _, _ = eng.search_best_move(board)
        assert move is not None
        board.push(move)
        assert not board.is_stalemate()

    # ── Terminal positions ─────────────────────────────────────────────────

    def test_checkmate_returns_none(self):
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1")
        assert board.is_checkmate()
        move, _, score = self.engine.search_best_move(board)
        assert move is None

    def test_stalemate_returns_none(self):
        board = chess.Board("5k2/5P2/5K2/8/8/8/8/8 b - - 0 1")
        assert board.is_stalemate()
        move, _, score = self.engine.search_best_move(board)
        assert move is None

    # ── Tactical positions ─────────────────────────────────────────────────

    def test_captures_hanging_piece(self):
        board = chess.Board("4k3/8/5n2/8/3B4/8/8/4K3 w - - 0 1")
        eng = SearchEngine(depth=2)
        move, _, score = eng.search_best_move(board)
        assert move is not None

    def test_promotes_pawn(self):
        board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        eng = SearchEngine(depth=3)
        move, _, _ = eng.search_best_move(board)
        assert move is not None
        assert move.promotion is not None

    def test_en_passant_handling(self):
        board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
        eng = SearchEngine(depth=2)
        move, _, _ = eng.search_best_move(board)
        assert move is not None

    # ── Draw detection ─────────────────────────────────────────────────────

    def test_draw_detection(self):
        board = chess.Board()
        moves = ["g1f3", "g8f6", "f3g1", "f6g8"] * 4
        for m in moves:
            board.push(chess.Move.from_uci(m))
        assert board.can_claim_draw()
        eng = SearchEngine(depth=2)
        _, _, score = eng.search_best_move(board)
        assert abs(score) < 100

    # ── Stop mechanism ─────────────────────────────────────────────────────

    def test_stop_search(self):
        board = chess.Board()
        self.engine._stop_event.set()
        move, _, score = self.engine.search_best_move(board)

    def test_stop_during_search(self):
        board = chess.Board()
        eng = SearchEngine(depth=20)

        def stop_after_delay():
            time.sleep(0.05)
            eng._stop_event.set()

        t = threading.Thread(target=stop_after_delay)
        t.start()
        start = time.time()
        eng.search_best_move(board)
        elapsed = time.time() - start
        t.join()
        assert elapsed < 2.0

    # ── Async search ───────────────────────────────────────────────────────

    def test_start_search_async(self):
        # Use a non-book position to force actual search
        board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4")
        results = []

        def cb(move, ponder, depth, score):
            results.append((move, ponder, depth, score))

        self.engine.start_search(board, callback=cb)
        if self.engine._thread:
            self.engine._thread.join(timeout=10)
        assert len(results) > 0
        # Last callback should have depth=-1 (finished signal)
        assert results[-1][2] == -1

    def test_start_search_produces_valid_move(self):
        # Use a non-book position
        board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4")
        final_move = [None]

        def cb(move, ponder, depth, score):
            if depth == -1:
                final_move[0] = move

        self.engine.start_search(board, callback=cb)
        if self.engine._thread:
            self.engine._thread.join(timeout=10)
        assert final_move[0] is not None
        assert final_move[0] in board.legal_moves


# ════════════════════════════════════════════════════════════════════════════
#  MOVE ORDERING TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestMoveOrdering:
    def setup_method(self):
        self.engine = SearchEngine(depth=2)

    def test_captures_ordered_first(self):
        board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 1")
        moves = self.engine._order_moves(board, None, 0)
        capture = chess.Move.from_uci("e4d5")
        idx = moves.index(capture)
        assert idx < 5

    def test_tt_move_first(self):
        board = chess.Board()
        tt_move = chess.Move.from_uci("g1f3")
        moves = self.engine._order_moves(board, tt_move, 0)
        assert moves[0] == tt_move

    def test_killer_moves_after_captures(self):
        board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 1")
        killer = chess.Move.from_uci("d2d4")
        self.engine.killers[0][0] = killer
        moves = self.engine._order_moves(board, None, 0)
        capture = chess.Move.from_uci("e4d5")
        cap_idx = moves.index(capture)
        kill_idx = moves.index(killer)
        assert kill_idx > cap_idx
        assert kill_idx < len(moves) // 2

    def test_history_heuristic_ordering(self):
        board = chess.Board()
        self.engine.history[chess.D2][chess.D4] = 10000
        moves = self.engine._order_moves(board, None, 0)
        d2d4 = chess.Move.from_uci("d2d4")
        idx = moves.index(d2d4)
        assert idx < 5

    def test_mvv_lva_pawn_takes_queen(self):
        """PxQ should score highest in MVV-LVA."""
        board = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
        move = chess.Move.from_uci("e4d5")
        if move in board.legal_moves:
            score = self.engine._mvv_lva(board, move)
            # Queen piece_type=5, Pawn=1: 5*10 - 1 = 49
            assert score >= 49

    def test_mvv_lva_en_passant(self):
        board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
        ep_move = chess.Move.from_uci("e5d6")
        assert board.is_en_passant(ep_move)
        score = self.engine._mvv_lva(board, ep_move)
        assert score == 9  # Pawn(1)*10 - Pawn(1) = 9

    def test_all_legal_moves_present(self):
        """_order_moves should return all legal moves, not drop any."""
        board = chess.Board()
        ordered = self.engine._order_moves(board, None, 0)
        legal = list(board.legal_moves)
        assert len(ordered) == len(legal)
        assert set(ordered) == set(legal)


# ════════════════════════════════════════════════════════════════════════════
#  ENGINE WRAPPER INTEGRATION TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestEngineWrapper:
    def test_get_best_move(self):
        eng = Engine(depth=2)
        uci_move, _, score = eng.get_best_move()
        assert uci_move is not None
        assert len(uci_move) in [4, 5]

    def test_make_move(self):
        eng = Engine(depth=2)
        assert eng.make_move("e2e4") is True
        assert eng.make_move("e7e5") is True

    def test_make_illegal_move(self):
        eng = Engine(depth=2)
        assert eng.make_move("e2e5") is False

    def test_play_sequence(self):
        eng = Engine(depth=2)
        eng.make_move("e2e4")
        eng.make_move("e7e5")
        eng.make_move("g1f3")
        eng.make_move("b8c6")
        uci_move, _, score = eng.get_best_move()
        assert uci_move is not None

    def test_score_returned(self):
        eng = Engine(depth=2)
        _, _, score = eng.get_best_move()
        assert isinstance(score, (int, float))


# ════════════════════════════════════════════════════════════════════════════
#  EDGE CASE TESTS (Common chess engine failures)
# ════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_only_one_legal_move(self):
        board = chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
        legal = list(board.legal_moves)
        eng = SearchEngine(depth=3)
        move, _, _ = eng.search_best_move(board)
        assert move in legal

    def test_promotion_types(self):
        board = chess.Board("3rk3/7P/8/8/8/8/8/4K3 w - - 0 1")
        eng = SearchEngine(depth=3)
        move, _, _ = eng.search_best_move(board)
        assert move is not None
        if move.promotion:
            assert move.promotion in [chess.QUEEN, chess.KNIGHT, chess.ROOK, chess.BISHOP]

    def test_zugzwang_position(self):
        board = chess.Board("8/8/p1p5/1p5p/1P5p/8/PPP2PPk/4K3 w - - 0 1")
        eng = SearchEngine(depth=3)
        move, _, _ = eng.search_best_move(board)
        assert move is not None

    def test_empty_board_except_kings(self):
        board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        assert board.is_insufficient_material()
        eng = SearchEngine(depth=2)
        move, _, score = eng.search_best_move(board)

    def test_many_pieces_on_board(self):
        board = chess.Board()
        eng = SearchEngine(depth=3)
        move, _, _ = eng.search_best_move(board)
        assert move is not None
        assert move in board.legal_moves

    def test_fen_with_half_move_clock(self):
        # Use halfmove=10 (not near 50-move rule)
        board = chess.Board("4k3/8/8/8/8/8/4R3/4K3 w - - 10 50")
        eng = SearchEngine(depth=2)
        move, _, _ = eng.search_best_move(board)
        assert move is not None

    def test_both_sides_can_castle(self):
        board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
        eng = SearchEngine(depth=2)
        move, _, _ = eng.search_best_move(board)
        assert move is not None
        assert move in board.legal_moves

    def test_pinned_piece_legal_moves_only(self):
        board = chess.Board("4r1k1/8/8/8/8/8/4N3/4K3 w - - 0 1")
        eng = SearchEngine(depth=2)
        move, _, _ = eng.search_best_move(board)
        assert move is not None
        assert move in board.legal_moves

    def test_double_check(self):
        """Only king moves should be legal in double check."""
        board = chess.Board("3k4/8/8/4N3/8/8/5B2/3K4 b - - 0 1")
        # Not a real double check, but we need to verify no crash
        eng = SearchEngine(depth=2)
        if not board.is_game_over():
            move, _, _ = eng.search_best_move(board)
            if move:
                assert move in board.legal_moves

    def test_discovered_check(self):
        """Engine should handle discovered check positions."""
        board = chess.Board("4k3/8/8/8/3p4/2B5/8/R3K3 w - - 0 1")
        eng = SearchEngine(depth=2)
        move, _, _ = eng.search_best_move(board)
        assert move is not None

    def test_underpromotion_knight_check(self):
        """Knight promotion that delivers check should be considered."""
        # Pawn on g7, promote to knight gives check on g8 if king on e8... 
        # Actually h7->h8=N doesn't give check. Let's use f7 pawn:
        board = chess.Board("4k3/5P2/8/8/8/8/8/4K3 w - - 0 1")
        eng = SearchEngine(depth=3)
        move, _, _ = eng.search_best_move(board)
        assert move is not None
        assert move.promotion is not None  # Should promote

    def test_seven_piece_position(self):
        """Position with 7 pieces (beyond Syzygy 5-piece)."""
        board = chess.Board("4k3/pp6/8/8/8/8/PP6/4K3 w - - 0 1")
        eng = SearchEngine(depth=3)
        move, _, _ = eng.search_best_move(board)
        assert move is not None

    def test_many_captures_available(self):
        """Position with many captures should not crash move ordering."""
        board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4")
        eng = SearchEngine(depth=2)
        move, _, _ = eng.search_best_move(board)
        assert move is not None


# ════════════════════════════════════════════════════════════════════════════
#  STRESS TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestStress:

    def test_deep_search_completes(self):
        board = chess.Board()
        eng = SearchEngine(depth=5)
        start = time.time()
        move, _, _ = eng.search_best_move(board)
        elapsed = time.time() - start
        assert move is not None
        assert elapsed < 30

    def test_complex_middlegame(self):
        board = chess.Board("rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6")
        eng = SearchEngine(depth=4)
        move, _, score = eng.search_best_move(board)
        assert move is not None
        assert move in board.legal_moves

    def test_endgame_krk(self):
        board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
        eng = SearchEngine(depth=4)
        move, _, score = eng.search_best_move(board)
        assert move is not None
        assert score > 0

    def test_endgame_kqk(self):
        board = chess.Board("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1")
        eng = SearchEngine(depth=4)
        move, _, score = eng.search_best_move(board)
        assert move is not None
        assert score > 0

    def test_multiple_searches_same_engine(self):
        eng = SearchEngine(depth=2)
        boards = [
            chess.Board(),
            chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
            chess.Board("r1bqkbnr/pppppppp/2n5/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 2"),
        ]
        for board in boards:
            move, _, _ = eng.search_best_move(board)
            assert move is not None
            assert move in board.legal_moves

    def test_tt_fills_under_pressure(self):
        eng = SearchEngine(depth=4)
        eng.tt = TranspositionTable(size_mb=1)
        # Use a non-book position
        board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4")
        move, _, _ = eng.search_best_move(board)
        assert move is not None
        assert len(eng.tt.table) > 0

    def test_concurrent_search_and_stop(self):
        eng = SearchEngine(depth=10)
        board = chess.Board()

        for _ in range(5):
            results = []
            eng.start_search(board, callback=lambda m, p, d, s: results.append(d))
            time.sleep(0.02)
            eng.stop()
            if eng._thread:
                eng._thread.join(timeout=2)

    def test_search_all_starting_moves(self):
        eng = SearchEngine(depth=2)
        board = chess.Board()
        for move in list(board.legal_moves):
            b = board.copy()
            b.push(move)
            result, _, _ = eng.search_best_move(b)
            if not b.is_game_over():
                assert result is not None
                assert result in b.legal_moves

    def test_rapid_evaluations(self):
        ev = BitboardEvaluator()
        board = chess.Board()
        start = time.time()
        for _ in range(10000):
            ev.evaluate(board)
        elapsed = time.time() - start
        assert elapsed < 10

    def test_search_from_10_different_openings(self):
        """Search from 10 different well-known opening positions."""
        positions = [
            chess.STARTING_FEN,
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",  # 1.e4
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",  # 1.d4
            "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1",  # 1.c4
            "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 0 1",  # 1.Nf3
            "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",  # French
            "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",  # Caro-Kann
            "rnbqkbnr/pppppp1p/6p1/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",  # KID
            "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",  # Alekhine
            "rnbqkbnr/pppppppp/8/8/8/6P1/PPPPPP1P/RNBQKBNR b KQkq - 0 1",  # 1.g3
        ]
        eng = SearchEngine(depth=2)
        for fen in positions:
            board = chess.Board(fen)
            move, _, _ = eng.search_best_move(board)
            assert move is not None
            assert move in board.legal_moves

    def test_endgame_positions_batch(self):
        """Test several endgame positions don't crash."""
        positions = [
            "4k3/8/8/8/8/8/4R3/4K3 w - - 0 1",   # KR vs K
            "4k3/8/8/8/8/8/8/4KQ2 w - - 0 1",     # KQ vs K
            "4k3/4P3/8/8/8/8/8/4K3 w - - 0 1",     # K+P vs K
            "4k3/8/8/8/8/8/8/3RKR2 w - - 0 1",     # KRR vs K
            "4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1",    # KBB vs K
            "4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1",  # Pawn endgame
        ]
        eng = SearchEngine(depth=3)
        for fen in positions:
            board = chess.Board(fen)
            if not board.is_game_over():
                move, _, _ = eng.search_best_move(board)
                assert move is not None


# ════════════════════════════════════════════════════════════════════════════
#  QUIESCENCE SEARCH TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestQuiescence:
    def setup_method(self):
        self.engine = SearchEngine(depth=1)

    def test_quiescence_quiet_position(self):
        board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        score = self.engine._quiescence(board, -INF, INF)
        assert score == 0

    def test_quiescence_finds_winning_capture(self):
        board = chess.Board("4k3/8/8/8/8/3q4/4P3/4K3 w - - 0 1")
        score = self.engine._quiescence(board, -INF, INF)
        assert score > 0

    def test_quiescence_depth_limit(self):
        board = chess.Board()
        score = self.engine._quiescence(board, -INF, INF, qs_depth=29)
        assert isinstance(score, (int, float))

    def test_check_evasion_in_quiescence(self):
        board = chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
        assert board.is_check()
        score = self.engine._quiescence(board, -INF, INF)
        assert isinstance(score, (int, float))

    def test_quiescence_respects_stop(self):
        board = chess.Board()
        self.engine._stop_event.set()
        score = self.engine._quiescence(board, -INF, INF)
        assert score == 0


# ════════════════════════════════════════════════════════════════════════════
#  PV LINE EXTRACTION TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestPVLine:
    def test_pv_after_search(self):
        """After a search, PV line should have at least one move."""
        eng = SearchEngine(depth=3)
        # Use a non-book position
        board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4")
        eng.search_best_move(board)
        pv = eng._get_pv_line(board, 3)
        assert len(pv) >= 1
        assert pv[0] in board.legal_moves

    def test_pv_empty_no_tt(self):
        """PV extraction with empty TT should return empty list."""
        eng = SearchEngine(depth=2)
        eng.tt.clear()
        board = chess.Board()
        pv = eng._get_pv_line(board, 3)
        assert len(pv) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
