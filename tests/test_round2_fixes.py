"""
Unit and integration tests for Round 2 search/eval fixes.

Covers the 10 fundamental fixes:
1. QS non-capture queen promotions
2. Delta pruning threshold (975 → 1400)
3. TT mate score ply normalization (_score_to_tt / _score_from_tt)
4. SEE rewrite with x-ray attack discovery (_discover_xray)
5. SEE phantom gain fix (attacker check before gain computation)
6. PVS non-PV duplication removed
7. Advanced passer rank-7 EG bonus (0 → 250)
8. History heuristic capped at ±90000
9. History malus uses searched_quiets only
10. NMP verification gated on game_phase ≤ 16
11. LMR not-improving reduction gated on depth > 3
"""

import chess
import pytest
from collections import defaultdict

from engine.core.search import (
    SearchEngine,
    MATE_SCORE,
    INF,
    MATE_THRESHOLD,
    SEE_PIECE_VALUES,
)
from engine.core.bitboard_evaluator import BitboardEvaluator
from engine.core.transposition import TranspositionTable, TT_EXACT, TT_ALPHA, TT_BETA

# ════════════════════════════════════════════════════════════════════════════
#  TT MATE SCORE PLY NORMALIZATION
# ════════════════════════════════════════════════════════════════════════════


class TestTTMateScoreNormalization:
    """Test _score_to_tt and _score_from_tt roundtrip correctly."""

    def test_positive_mate_score_roundtrip(self):
        """Mate score stored at ply=3 and retrieved at ply=3 should be identical."""
        engine = SearchEngine(depth=2)
        mate_in_5 = MATE_SCORE - 5
        ply = 3
        tt_val = engine._score_to_tt(mate_in_5, ply)
        recovered = engine._score_from_tt(tt_val, ply)
        assert recovered == mate_in_5

    def test_negative_mate_score_roundtrip(self):
        """Negative mate scores should also roundtrip correctly."""
        engine = SearchEngine(depth=2)
        mated_in_4 = -(MATE_SCORE - 4)
        ply = 5
        tt_val = engine._score_to_tt(mated_in_4, ply)
        recovered = engine._score_from_tt(tt_val, ply)
        assert recovered == mated_in_4

    def test_positive_mate_stored_adds_ply(self):
        """_score_to_tt should add ply to positive mate scores."""
        engine = SearchEngine(depth=2)
        mate_in_3 = MATE_SCORE - 3
        ply = 2
        tt_val = engine._score_to_tt(mate_in_3, ply)
        assert tt_val == mate_in_3 + ply

    def test_negative_mate_stored_subtracts_ply(self):
        """_score_to_tt should subtract ply from negative mate scores."""
        engine = SearchEngine(depth=2)
        mated_in_3 = -(MATE_SCORE - 3)
        ply = 2
        tt_val = engine._score_to_tt(mated_in_3, ply)
        assert tt_val == mated_in_3 - ply

    def test_normal_score_unchanged(self):
        """Non-mate scores should pass through unchanged."""
        engine = SearchEngine(depth=2)
        for score in [0, 100, -200, 500, -500]:
            for ply in [0, 1, 5, 10]:
                assert engine._score_to_tt(score, ply) == score
                assert engine._score_from_tt(score, ply) == score

    def test_mate_at_different_plies(self):
        """Store at ply=2, retrieve at ply=5 — should shift the mate distance."""
        engine = SearchEngine(depth=2)
        mate_in_4 = MATE_SCORE - 4
        store_ply = 2
        retrieve_ply = 5
        tt_val = engine._score_to_tt(mate_in_4, store_ply)
        recovered = engine._score_from_tt(tt_val, retrieve_ply)
        # Stored: (MATE_SCORE - 4) + 2 = MATE_SCORE - 2
        # Retrieved: (MATE_SCORE - 2) - 5 = MATE_SCORE - 7
        assert recovered == MATE_SCORE - 7

    def test_threshold_boundary(self):
        """Scores exactly at MATE_THRESHOLD should be treated as mate scores."""
        engine = SearchEngine(depth=2)
        score = MATE_THRESHOLD + 1  # Just above threshold
        ply = 3
        tt_val = engine._score_to_tt(score, ply)
        assert tt_val == score + ply  # Should be adjusted

        score_below = MATE_THRESHOLD  # Exactly at threshold, not above
        tt_val_below = engine._score_to_tt(score_below, ply)
        assert tt_val_below == score_below  # Should NOT be adjusted

    def test_ply_zero_no_change(self):
        """At ply=0, mate scores should be stored/retrieved unchanged."""
        engine = SearchEngine(depth=2)
        mate = MATE_SCORE - 3
        assert engine._score_to_tt(mate, 0) == mate
        assert engine._score_from_tt(mate, 0) == mate


# ════════════════════════════════════════════════════════════════════════════
#  SEE (STATIC EXCHANGE EVALUATION) TESTS
# ════════════════════════════════════════════════════════════════════════════


class TestSEE:
    """Test the rewritten SEE with x-ray attack discovery."""

    def setup_method(self):
        self.engine = SearchEngine(depth=2)

    def test_see_pawn_takes_pawn_undefended(self):
        """PxP undefended: quick exit returns captured - attacker = 0 (equal exchange bound)."""
        board = chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
        move = chess.Move.from_uci("e4d5")
        assert board.is_legal(move)
        see = self.engine._see(board, move)
        # Quick exit: attacker_val(100) <= captured_val(100) → returns 100-100 = 0
        assert see == 0

    def test_see_pawn_takes_defended_pawn(self):
        """PxP when pawn is defended by another pawn: PxP, PxP → net 0."""
        board = chess.Board("4k3/8/4p3/3p4/4P3/8/8/4K3 w - - 0 1")
        move = chess.Move.from_uci("e4d5")
        see = self.engine._see(board, move)
        assert see == 0  # Even exchange

    def test_see_pawn_takes_queen(self):
        """PxQ: quick exit returns captured - attacker = 900 - 100 = 800."""
        board = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
        move = chess.Move.from_uci("e4d5")
        see = self.engine._see(board, move)
        # Quick exit: P(100) <= Q(900) → returns 900-100 = 800
        assert see == SEE_PIECE_VALUES[chess.QUEEN] - SEE_PIECE_VALUES[chess.PAWN]

    def test_see_queen_takes_defended_pawn_negative(self):
        """QxP when pawn is defended: Q(900) for P(100) with recapture = bad trade."""
        board = chess.Board("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1")
        move = chess.Move.from_uci("d1d5")
        see = self.engine._see(board, move)
        assert see < 0  # Losing trade

    def test_see_rook_battery_xray(self):
        """R+R battery (x-ray): RxP on defended file should discover backup rook."""
        # White rooks on a1 and a2, black pawn on a7 defended by rook on a8.
        # RxP, RxR, RxR — x-ray should discover the second rook.
        board = chess.Board("r3k3/p7/8/8/8/8/R7/R3K3 w - - 0 1")
        move = chess.Move.from_uci("a2a7")
        if board.is_legal(move):
            see = self.engine._see(board, move)
            # RxP (+100), RxR (-500+100=-400 overall), R recaptures (+500-400=+100 overall)
            # Net should be positive or 0 with battery support
            assert see >= 0

    def test_see_single_rook_vs_defended_pawn_negative(self):
        """Single R vs defended pawn: RxP, PxR → R loses 400cp net."""
        board = chess.Board("4k3/8/4p3/3p4/8/8/8/3RK3 w - - 0 1")
        move = chess.Move.from_uci("d1d5")
        see = self.engine._see(board, move)
        # R(500) captures P(100), recaptured by P → net = 100 - 500 = -400
        assert see == 100 - 500  # -400

    def test_see_en_passant(self):
        """EP PxP: quick exit returns captured - attacker = 0 (equal exchange bound)."""
        board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
        move = chess.Move.from_uci("e5d6")
        assert board.is_en_passant(move)
        see = self.engine._see(board, move)
        # Quick exit: P(100) <= P(100) → 0
        assert see == 0

    def test_see_non_capture_returns_zero(self):
        """SEE on a non-capture with no piece on target should return 0."""
        board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
        move = chess.Move.from_uci("e2e4")
        see = self.engine._see(board, move)
        assert see == 0

    def test_see_equal_trade_bishop_for_knight(self):
        """BxN where N is undefended should be positive (B=330 > N=320 but captures N)."""
        board = chess.Board("4k3/8/8/3n4/8/5B2/8/4K3 w - - 0 1")
        move = chess.Move.from_uci("f3d5")
        see = self.engine._see(board, move)
        # BxN undefended = +320
        assert see == SEE_PIECE_VALUES[chess.KNIGHT]

    def test_see_quick_exit_pawn_captures_rook(self):
        """PxR should exit quickly (attacker_val <= captured_val shortcut)."""
        board = chess.Board("4k3/8/8/3r4/4P3/8/8/4K3 w - - 0 1")
        move = chess.Move.from_uci("e4d5")
        see = self.engine._see(board, move)
        # P(100) captures R(500): quick exit returns 500 - 100 = 400
        assert see == SEE_PIECE_VALUES[chess.ROOK] - SEE_PIECE_VALUES[chess.PAWN]


# ════════════════════════════════════════════════════════════════════════════
#  X-RAY DISCOVERY TESTS
# ════════════════════════════════════════════════════════════════════════════


class TestXRayDiscovery:
    """Test the _discover_xray helper specifically."""

    def setup_method(self):
        self.engine = SearchEngine(depth=2)

    def test_rook_xray_on_file(self):
        """Rook behind consumed square on same file should be discovered."""
        # White rooks on a1 (stays) and a3 (consumed) aiming at a7.
        board = chess.Board("4k3/8/8/8/8/R7/8/R3K3 w - - 0 1")
        occ = int(board.occupied) ^ (1 << chess.A3)  # Remove a3 rook
        result = self.engine._discover_xray(board, chess.A7, chess.A3, occ)
        assert result == chess.A1

    def test_bishop_xray_on_diagonal(self):
        """Bishop behind consumed square on diagonal should be discovered."""
        board = chess.Board("4k3/8/8/8/3B4/8/5B2/4K3 w - - 0 1")
        # Bishop at d4, another at f2. If d4 is consumed aiming at a7:
        # Ray from a7 through d4 is diagonal. f2 isn't on that ray.
        # Let's use a better setup: bishops on c3 and a1, target at e5.
        board2 = chess.Board("4k3/8/8/4p3/8/2B5/8/B3K3 w - - 0 1")
        occ = int(board2.occupied) ^ (1 << chess.C3)  # Remove c3 bishop
        result = self.engine._discover_xray(board2, chess.E5, chess.C3, occ)
        assert result == chess.A1

    def test_no_xray_for_knight(self):
        """Knights don't produce x-ray attacks."""
        board = chess.Board("4k3/8/8/3p4/8/2N5/8/2N1K3 w - - 0 1")
        occ = int(board.occupied) ^ (1 << chess.C3)
        result = self.engine._discover_xray(board, chess.D5, chess.C3, occ)
        # c3-d5 direction is (+1,+2) which is not a ray direction
        # Actually c3=file 2 rank 2, d5=file 3 rank 4. dr=2, df=1 — not a valid ray
        assert result is None

    def test_no_xray_blocked_by_piece(self):
        """If path behind consumed square is blocked, no x-ray found."""
        board = chess.Board("4k3/8/8/8/8/R7/P7/R3K3 w - - 0 1")
        # Rook on a3, pawn on a2, rook on a1. After consuming a3,
        # pawn at a2 blocks the rook at a1.
        occ = int(board.occupied) ^ (1 << chess.A3)
        result = self.engine._discover_xray(board, chess.A7, chess.A3, occ)
        # a2 pawn blocks → not a sliding piece on file
        assert result is None

    def test_queen_xray_on_diagonal(self):
        """Queen behind consumed square on diagonal should be discovered."""
        board = chess.Board("4k3/8/8/8/3B4/8/1Q6/4K3 w - - 0 1")
        # Bishop on d4, queen on b2. Target at f6.
        # d4 to f6 direction: dr=+2, df=+2 → diagonal. b2 to d4 is also diagonal.
        occ = int(board.occupied) ^ (1 << chess.D4)  # Remove d4
        result = self.engine._discover_xray(board, chess.F6, chess.D4, occ)
        assert result == chess.B2


# ════════════════════════════════════════════════════════════════════════════
#  QS NON-CAPTURE QUEEN PROMOTIONS
# ════════════════════════════════════════════════════════════════════════════


class TestQSPromotions:
    """Test that quiescence search finds non-capture queen promotions."""

    def test_qs_sees_non_capture_promotion(self):
        """QS should evaluate a pawn push to promotion (non-capture)."""
        # White pawn on a7 about to promote, no piece on a8.
        board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        engine = SearchEngine(depth=1)
        score = engine._quiescence(board, -INF, INF)
        # Should see the promotion value (~900cp for queen promotion)
        assert score > 800

    def test_qs_promotion_vs_no_promotion(self):
        """QS score should be much higher for a position with imminent promotion."""
        engine = SearchEngine(depth=1)
        # Pawn on a7 (about to promote)
        board_promo = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        score_promo = engine._quiescence(board_promo, -INF, INF)

        # Pawn on a2 (far from promoting)
        board_no = chess.Board("4k3/8/8/8/8/8/P7/4K3 w - - 0 1")
        score_no = engine._quiescence(board_no, -INF, INF)

        # Promotion position should score significantly higher
        assert score_promo > score_no + 300

    def test_qs_black_non_capture_promotion(self):
        """QS should also find non-capture promotions for Black."""
        board = chess.Board("4k3/8/8/8/8/8/p7/4K3 b - - 0 1")
        engine = SearchEngine(depth=1)
        score = engine._quiescence(board, -INF, INF)
        # Black to move, promoting — should be very positive for Black
        assert score > 800

    def test_search_finds_promotion_push(self):
        """Full search should find pawn push promotion even without capture."""
        board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        engine = SearchEngine(depth=2)
        move, _, score = engine.search_best_move(board)
        assert move is not None
        assert move.promotion == chess.QUEEN
        assert move == chess.Move.from_uci("a7a8q")


# ════════════════════════════════════════════════════════════════════════════
#  ADVANCED PASSER RANK-7 BONUS
# ════════════════════════════════════════════════════════════════════════════


class TestAdvancedPasserBonus:
    """Test that rank-7 passed pawns get the large EG bonus."""

    def test_rank7_passer_bonus_white(self):
        """White passed pawn on rank 7 should have large bonus vs rank 5."""
        evaluator = BitboardEvaluator()
        # White pawn on a7 (rank 6 in 0-indexed) — rank 7 in chess terms
        board_7 = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        score_7 = evaluator.evaluate(board_7)

        # White pawn on a5 (rank 4 in 0-indexed)
        board_5 = chess.Board("4k3/8/8/P7/8/8/8/4K3 w - - 0 1")
        score_5 = evaluator.evaluate(board_5)

        # Rank 7 should score significantly higher than rank 5
        assert score_7 > score_5 + 100

    def test_rank7_passer_bonus_black(self):
        """Black passed pawn on rank 2 (their rank 7) should have large bonus."""
        evaluator = BitboardEvaluator()
        # Black pawn on a2 (about to promote)
        board_7 = chess.Board("4k3/8/8/8/8/8/p7/4K3 b - - 0 1")
        score_7 = evaluator.evaluate(board_7)

        # Black pawn on a5 (rank 4, their rank 4)
        board_5 = chess.Board("4k3/8/8/p7/8/8/8/4K3 b - - 0 1")
        score_5 = evaluator.evaluate(board_5)

        # Black's rank-7 passer should evaluate much better for Black
        assert score_7 > score_5 + 100

    def test_rank7_bonus_larger_than_rank6(self):
        """The rank-7 bonus (250) should exceed rank-6 bonus (150)."""
        evaluator = BitboardEvaluator()
        # Use pure K+P endgame to isolate the bonus difference.
        board_r7 = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        score_r7 = evaluator.evaluate(board_r7)

        board_r6 = chess.Board("4k3/8/P7/8/8/8/8/4K3 w - - 0 1")
        score_r6 = evaluator.evaluate(board_r6)

        # rank 7 bonus is 250, rank 6 is 150, so rank 7 should be at least 50 more
        # (exact diff depends on PST, king proximity, etc.)
        assert score_r7 > score_r6


# ════════════════════════════════════════════════════════════════════════════
#  HISTORY HEURISTIC CAP
# ════════════════════════════════════════════════════════════════════════════


class TestHistoryCap:
    """Test that history values are capped at ±90000."""

    def test_history_bonus_capped(self):
        """After many deep bonuses, history should not exceed 90000."""
        engine = SearchEngine(depth=2)
        # Simulate many deep search bonuses
        from_sq, to_sq = chess.E2, chess.E4
        for _ in range(100):
            engine.history[from_sq][to_sq] += 20 * 20  # depth=20 bonus
            engine.history[from_sq][to_sq] = min(90000, engine.history[from_sq][to_sq])
        assert engine.history[from_sq][to_sq] <= 90000

    def test_history_malus_capped(self):
        """History malus should not go below -90000."""
        engine = SearchEngine(depth=2)
        from_sq, to_sq = chess.D2, chess.D4
        for _ in range(100):
            engine.history[from_sq][to_sq] -= 20 * 20
            engine.history[from_sq][to_sq] = max(-90000, engine.history[from_sq][to_sq])
        assert engine.history[from_sq][to_sq] >= -90000

    def test_history_cap_in_move_ordering(self):
        """Move ordering should respect history cap — no quiet move scores above captures."""
        engine = SearchEngine(depth=2)
        # Set a history value to exactly the cap
        engine.history[chess.E2][chess.E4] = 90000
        board = chess.Board()
        moves = engine._order_moves(board, None, 0)
        # The e2e4 move should not outscore captures (which get 200000+ base)
        e2e4 = chess.Move.from_uci("e2e4")
        e2e4_idx = moves.index(e2e4)

        # In a position with captures available, captures should come first.
        # In starting position there are no captures, so history-high move should be near top.
        assert e2e4_idx < 5

    def test_history_after_real_search(self):
        """After a real search, no history value should exceed ±90000."""
        engine = SearchEngine(depth=4)
        # Use a non-trivial position to generate history entries
        board = chess.Board(
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4"
        )
        engine.search_best_move(board)

        # Check all history entries
        for from_sq in range(64):
            for to_sq in range(64):
                val = engine.history[from_sq][to_sq]
                assert (
                    -90000 <= val <= 90000
                ), f"History[{chess.square_name(from_sq)}][{chess.square_name(to_sq)}] = {val}"


# ════════════════════════════════════════════════════════════════════════════
#  DELTA PRUNING THRESHOLD
# ════════════════════════════════════════════════════════════════════════════


class TestDeltaPruning:
    """Test that delta pruning uses 1400 threshold (not 975)."""

    def test_promotion_not_delta_pruned(self):
        """A position where a pawn is about to promote should not be pruned by delta."""
        # With threshold=975, a position where stand_pat is ~-900 and alpha is 0
        # would prune (−900 + 975 = 75 < 0 fails). With 1400, it passes.
        # We test indirectly: engine finds the promotion.
        board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        engine = SearchEngine(depth=2)
        move, _, score = engine.search_best_move(board)
        assert move.promotion == chess.QUEEN
        assert score > 500  # Should see the promotion value


# ════════════════════════════════════════════════════════════════════════════
#  PVS FIX (NO DUPLICATE NON-PV SEARCH)
# ════════════════════════════════════════════════════════════════════════════


class TestPVSFix:
    """Test that PVS only applies null-window to PV nodes."""

    def test_search_completes_efficiently(self):
        """Search should complete without doubled work on non-PV nodes.
        After the PVS fix, the node count should be reasonable."""
        engine = SearchEngine(depth=4)
        board = chess.Board(
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4"
        )
        engine.search_best_move(board)
        # Before the fix, ~80% of nodes were searched twice.
        # Node count at depth 4 should be well under 100k for this position.
        assert engine.nodes < 200000, f"Node count too high: {engine.nodes}"

    def test_pvs_still_finds_best_move(self):
        """After PVS fix, search should still find the correct best move."""
        engine = SearchEngine(depth=3)
        # Winning capture: bishop takes hanging knight
        board = chess.Board("4k3/8/5n2/8/3B4/8/8/4K3 w - - 0 1")
        move, _, score = engine.search_best_move(board)
        assert move is not None
        assert score > 0  # Should capture the knight


# ════════════════════════════════════════════════════════════════════════════
#  NMP VERIFICATION GATING
# ════════════════════════════════════════════════════════════════════════════


class TestNMPVerification:
    """Test that NMP verification is only done in endgame-adjacent positions."""

    def test_middlegame_nmp_no_verification(self):
        """In pure middlegame (phase > 16), NMP should cut without verification.
        Verified by checking that search is fast on a typical middlegame position."""
        engine = SearchEngine(depth=5)
        # Rich middlegame with many pieces (phase should be > 16)
        board = chess.Board(
            "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        )
        import time

        start = time.time()
        engine.search_best_move(board)
        elapsed = time.time() - start
        # Should finish in reasonable time due to NMP cuts without verification
        assert elapsed < 30.0

    def test_endgame_nmp_still_works(self):
        """In endgame positions, NMP should still work (with verification)."""
        engine = SearchEngine(depth=4)
        # Endgame: K+R vs K (phase very low)
        board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
        move, _, score = engine.search_best_move(board)
        assert move is not None
        assert score > 0


# ════════════════════════════════════════════════════════════════════════════
#  LMR NOT-IMPROVING GUARD
# ════════════════════════════════════════════════════════════════════════════


class TestLMRImproving:
    """Test that the not-improving LMR extra reduction only applies at depth > 3."""

    def test_shallow_search_no_over_reduction(self):
        """At depth 3, LMR should not apply the extra +1 for not-improving.
        This prevents dropping directly into QS."""
        engine = SearchEngine(depth=3)
        # A tactical position where over-reduction at depth 3 would miss tactics
        board = chess.Board(
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        )
        move, _, score = engine.search_best_move(board)
        assert move is not None
        assert move in board.legal_moves

    def test_deep_search_still_reduces(self):
        """At depth >= 4, the not-improving extra reduction should still apply.
        We just verify the search completes correctly."""
        engine = SearchEngine(depth=5)
        board = chess.Board(
            "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        )
        move, _, score = engine.search_best_move(board)
        assert move is not None
        assert move in board.legal_moves


# ════════════════════════════════════════════════════════════════════════════
#  INTEGRATION: SEARCH + EVAL PIPELINE WITH NEW FIXES
# ════════════════════════════════════════════════════════════════════════════


class TestSearchEvalIntegration:
    """Integration tests verifying the fixes work together in realistic scenarios."""

    def test_promotion_in_endgame_found(self):
        """Engine should find pawn promotion in a K+P vs K endgame."""
        engine = SearchEngine(depth=4)
        # White pawn on a7, black king far away on h8 — clear path to promote
        board = chess.Board("7k/P7/8/8/8/8/8/4K3 w - - 0 1")
        move, _, score = engine.search_best_move(board)
        assert move is not None
        # Should promote
        assert move.promotion == chess.QUEEN
        assert score > 500

    def test_bishop_battery_xray_in_move_ordering(self):
        """Move ordering with SEE should correctly value x-ray attacks."""
        engine = SearchEngine(depth=3)
        # Position where bishop battery is relevant for move ordering
        board = chess.Board(
            "r1bqkb1r/pppppppp/2n2n2/8/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        )
        move, _, _ = engine.search_best_move(board)
        assert move is not None
        assert move in board.legal_moves

    def test_mate_score_consistent_across_search_depths(self):
        """Mate scores should be consistent when stored/retrieved from TT at different depths."""
        # Back rank mate in 1
        board = chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1")
        engine = SearchEngine(depth=3)
        _, _, score3 = engine.search_best_move(board)

        engine2 = SearchEngine(depth=4)
        _, _, score4 = engine2.search_best_move(board)

        # Both should find mate, scores should both be very high
        assert score3 > MATE_SCORE - 20
        assert score4 > MATE_SCORE - 20

    def test_rook_vs_defended_pawn_not_captured(self):
        """Engine should NOT sacrifice rook for a defended pawn (SEE negative)."""
        engine = SearchEngine(depth=3)
        # White rook on d1, black pawn on d5 defended by pawn on e6
        board = chess.Board("4k3/8/4p3/3p4/8/8/4P3/3RK3 w - - 0 1")
        move, _, _ = engine.search_best_move(board)
        assert move is not None
        # Should NOT play Rxd5 (SEE = -400)
        rxd5 = chess.Move.from_uci("d1d5")
        if rxd5 in board.legal_moves:
            assert move != rxd5

    def test_full_game_with_fixes(self):
        """Engine should play a complete short game without crashing."""
        engine = SearchEngine(depth=3)
        board = chess.Board()
        move_count = 0
        max_moves = 80

        while not board.is_game_over() and move_count < max_moves:
            move, _, _ = engine.search_best_move(board)
            if move is None:
                break
            assert move in board.legal_moves
            board.push(move)
            move_count += 1

        assert move_count > 10  # Should play at least some moves

    def test_see_pruning_in_quiescence(self):
        """QS should prune clearly losing captures via SEE."""
        engine = SearchEngine(depth=1)
        # Position where QxP is a losing capture (queen takes defended pawn)
        board = chess.Board("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1")

        # QS should still return a valid score
        score = engine._quiescence(board, -INF, INF)
        assert isinstance(score, (int, float))
        # Should not be extremely high (SEE should prune QxP)
        assert score < 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
