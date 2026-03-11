"""
BitboardEvaluator Module
========================

This module implements a static evaluation function for the chess engine using
64-bit integer masks (Bitboards). It is designed for performance and tactical
awareness.

Key Features:
    - O(1) Pawn Structure Analysis using bitwise operations.
    - Tapered Evaluation (interpolating between Middle Game and End Game).
    - Mobility and King Safety heuristics.
    - Pre-calculated attack masks for performance optimization.

Author: meedoomostafa
License: MIT
"""

import chess
from typing import List, Dict, Optional
from engine.config import CONFIG

# --- Constants & Pre-computations ---

# Represents a vertical file (File A) as a 64-bit integer.
# 0x0101010101010101 = Binary 00000001 repeated 8 times (A1, A2... A8).
FILE_A: int = 0x0101010101010101

# Array of bitmasks for all 8 files (File A to File H).
FILES: List[int] = [FILE_A << i for i in range(8)]


class BitboardEvaluator:
    """
    A high-performance static evaluator for chess positions using bitboards.

    This class is stateful only regarding configuration and pre-computed tables.
    The evaluation itself is stateless with respect to the board history.
    """

    def __init__(self) -> None:
        """
        Initializes the evaluator and pre-calculates expensive bitmasks.
        """
        self.cfg = CONFIG.eval
        self.last_phase = 24  # Cached phase from most recent evaluate() call.

        # Phase weights for game phase calculation (class-level to avoid re-creation)
        self.phase_weights = {
            chess.KNIGHT: 1,
            chess.BISHOP: 1,
            chess.ROOK: 2,
            chess.QUEEN: 4,
            chess.KING: 0,
        }

        # Pre-compute neighbor masks for Isolated Pawn detection.
        # neighbor_masks[i] contains bits for files adjacent to file i.
        self.neighbor_masks: List[int] = [0] * 8
        for f in range(8):
            if f > 0:
                self.neighbor_masks[f] |= FILES[f - 1]
            if f < 7:
                self.neighbor_masks[f] |= FILES[f + 1]

        # Pre-compute "Front Spans" for Passed Pawn detection.
        self.white_passed_masks: List[int] = [0] * 64
        self.black_passed_masks: List[int] = [0] * 64
        self._init_passed_masks()

        # Pre-compute King Safety masks (pawn shield squares).
        self.king_shield_masks: List[int] = [0] * 64
        self._init_king_masks()

        # King centralization table for endgame (Chebyshev distance from center).
        # Higher bonus for squares closer to the center (e4/d4/e5/d5).
        self._king_center_bonus: List[int] = [0] * 64
        for sq in range(64):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            # Distance from center (3.5, 3.5).
            center_dist = max(abs(f - 3.5), abs(r - 3.5))
            # Bonus: 0 at corner (dist ~3.5), up to ~20 at center (dist ~0.5).
            self._king_center_bonus[sq] = int((4 - center_dist) * 6)

    def _init_passed_masks(self) -> None:
        """Build front-span bitmasks for passed pawn detection."""
        for sq in range(64):
            f, r = chess.square_file(sq), chess.square_rank(sq)

            # White: ranks above the pawn.
            w_front = 0
            for rr in range(r + 1, 8):
                w_front |= 1 << chess.square(f, rr)
                if f > 0:
                    w_front |= 1 << chess.square(f - 1, rr)
                if f < 7:
                    w_front |= 1 << chess.square(f + 1, rr)
            self.white_passed_masks[sq] = w_front

            # Black: ranks below the pawn.
            b_front = 0
            for rr in range(0, r):
                b_front |= 1 << chess.square(f, rr)
                if f > 0:
                    b_front |= 1 << chess.square(f - 1, rr)
                if f < 7:
                    b_front |= 1 << chess.square(f + 1, rr)
            self.black_passed_masks[sq] = b_front

    def _init_king_masks(self) -> None:
        """Build pawn-shield bitmasks for king safety evaluation."""
        for sq in range(64):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            w_shield = 0
            # Shield squares: one rank ahead, same and adjacent files.
            if r < 7:
                w_shield |= 1 << chess.square(f, r + 1)
                if f > 0:
                    w_shield |= 1 << chess.square(f - 1, r + 1)
                if f < 7:
                    w_shield |= 1 << chess.square(f + 1, r + 1)
            self.king_shield_masks[sq] = w_shield

    def evaluate(self, board: chess.Board) -> int:
        """Return static eval in centipawns, positive favors side to move."""
        if board.is_insufficient_material():
            return 0
        if board.is_stalemate():
            return 0

        # Retrieve pawn bitboards once.
        white_pawns = board.pieces_mask(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces_mask(chess.PAWN, chess.BLACK)

        mg_score = 0
        eg_score = 0
        phase = 0

        for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            p_name = chess.piece_name(pt).upper()
            mg_val = self.cfg.MATERIAL_MG.get(p_name, 0)
            eg_val = self.cfg.MATERIAL_EG.get(p_name, 0)

            table_mg = getattr(self.cfg, f"PST_{p_name}_MG", None)
            table_eg = getattr(self.cfg, f"PST_{p_name}_EG", None)

            # Evaluate White Pieces
            squares = board.pieces(pt, chess.WHITE)
            count = len(squares)
            phase += count * self.phase_weights[pt]

            mg_score += count * mg_val
            eg_score += count * eg_val

            if table_mg:
                for sq in squares:
                    mg_score += table_mg[sq]
            if table_eg:
                for sq in squares:
                    eg_score += table_eg[sq]

            # Evaluate Black Pieces
            squares = board.pieces(pt, chess.BLACK)
            count = len(squares)
            phase += count * self.phase_weights[pt]

            mg_score -= count * mg_val
            eg_score -= count * eg_val

            if table_mg:
                # Mirror square index for Black (A1 <-> A8).
                for sq in squares:
                    mg_score -= table_mg[chess.square_mirror(sq)]
            if table_eg:
                for sq in squares:
                    eg_score -= table_eg[chess.square_mirror(sq)]

        # 3.1 Pawn material and PST.
        w_pawn_sqs = board.pieces(chess.PAWN, chess.WHITE)
        b_pawn_sqs = board.pieces(chess.PAWN, chess.BLACK)

        mg_score += len(w_pawn_sqs) * self.cfg.MATERIAL_MG["PAWN"]
        eg_score += len(w_pawn_sqs) * self.cfg.MATERIAL_EG["PAWN"]
        for sq in w_pawn_sqs:
            mg_score += self.cfg.PST_PAWN_MG[sq]
            eg_score += self.cfg.PST_PAWN_EG[sq]

        mg_score -= len(b_pawn_sqs) * self.cfg.MATERIAL_MG["PAWN"]
        eg_score -= len(b_pawn_sqs) * self.cfg.MATERIAL_EG["PAWN"]
        for sq in b_pawn_sqs:
            mirror_sq = chess.square_mirror(sq)
            mg_score -= self.cfg.PST_PAWN_MG[mirror_sq]
            eg_score -= self.cfg.PST_PAWN_EG[mirror_sq]

        # 3.2 Pawn structure (doubled, isolated, passed).
        w_pawn_struct = self._eval_pawns_bitwise(white_pawns, black_pawns, chess.WHITE)
        b_pawn_struct = self._eval_pawns_bitwise(black_pawns, white_pawns, chess.BLACK)

        # Pawn structure weighted differently per phase.
        mg_score += int(w_pawn_struct * 0.85)
        mg_score -= int(b_pawn_struct * 0.85)
        eg_score += int(w_pawn_struct * 1.0)
        eg_score -= int(b_pawn_struct * 1.0)

        # 4.1 Bishop pair bonus.
        if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
            mg_score += self.cfg.BISHOP_PAIR_BONUS
            eg_score += self.cfg.BISHOP_PAIR_BONUS
        if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
            mg_score -= self.cfg.BISHOP_PAIR_BONUS
            eg_score -= self.cfg.BISHOP_PAIR_BONUS

        # 4.2 Mobility (tapered: rook mobility valued higher in endgame).
        mg_mob, eg_mob = self._eval_mobility(board, white_pawns, black_pawns)
        mg_score += mg_mob
        eg_score += eg_mob

        # 4.3 King safety (critical in middlegame, diminishes in endgame).
        ks_score = self._eval_king_safety(board, white_pawns, black_pawns)
        mg_score += ks_score
        # Scale king safety into EG: floor of 25% so it never fully vanishes
        # while rooks/queens remain. Pure pawn endings still get ~0.
        eg_ks_factor = max(0.25, 0.15 * phase / 24)
        eg_score += int(ks_score * eg_ks_factor)

        # 4.4 Threat detection.
        threat_score = self._eval_threats(board)
        mg_score += threat_score
        eg_score += threat_score

        # 4.5 Development penalty in opening/middlegame.
        dev_score = self._eval_development(board, phase)
        mg_score += dev_score

        # 4.6 Rook on open/semi-open files.
        rook_file_score = self._eval_rook_files(board, white_pawns, black_pawns)
        mg_score += rook_file_score
        eg_score += rook_file_score

        # 4.7 Pawn storm near friendly king.
        pawn_storm_score = self._eval_pawn_king_proximity(
            board, white_pawns, black_pawns
        )
        mg_score += pawn_storm_score

        # 4.8 Knight outpost evaluation.
        outpost_score = self._eval_knight_outposts(board, white_pawns, black_pawns)
        mg_score += outpost_score
        eg_score += outpost_score

        # 4.9 Queen centrality.
        queen_score = self._eval_queen_activity(board, phase)
        mg_score += queen_score

        # 4.10 King-pawn proximity (endgame only — king support for passed pawns).
        kp_score = self._eval_king_pawn_proximity_eg(board, white_pawns, black_pawns)
        eg_score += kp_score

        # 4.11 Rook behind passed pawn and connected passed pawns (endgame).
        adv_passer_score = self._eval_advanced_passer_bonuses(
            board, white_pawns, black_pawns
        )
        eg_score += adv_passer_score

        # 4.12 King centralization bonus (endgame: king should move to center).
        w_king_sq = board.king(chess.WHITE)
        b_king_sq = board.king(chess.BLACK)
        if w_king_sq is not None and b_king_sq is not None:
            eg_score += self._king_center_bonus[w_king_sq]
            eg_score -= self._king_center_bonus[b_king_sq]

        # 4.13 Rook on 7th rank bonus.
        rook7_mg, rook7_eg = self._eval_rook_7th(board)
        mg_score += rook7_mg
        eg_score += rook7_eg

        # 4.14 Connected rooks bonus.
        conn_rook_score = self._eval_connected_rooks(board)
        mg_score += conn_rook_score
        eg_score += conn_rook_score

        # 4.15 Space advantage.
        space_score = self._eval_space(board, white_pawns, black_pawns)
        mg_score += space_score

        # 4.16 Tempo bonus: small advantage for side to move.
        TEMPO_BONUS = 10
        if board.turn == chess.WHITE:
            mg_score += TEMPO_BONUS
        else:
            mg_score -= TEMPO_BONUS

        # 4.17 Bishop vs Knight imbalance based on pawn structure.
        bn_score = self._eval_bishop_knight_imbalance(board, white_pawns, black_pawns)
        mg_score += bn_score
        eg_score += bn_score

        # Tapered eval: blend MG/EG scores by remaining material.
        phase = min(phase, 24)
        self.last_phase = phase  # Cache for external use (e.g., search pruning).
        final_score = (mg_score * phase + eg_score * (24 - phase)) // 24

        # Return relative to side to move (negamax convention).
        if board.turn == chess.BLACK:
            return -final_score
        return final_score

    def _eval_pawns_bitwise(self, my_pawns: int, opp_pawns: int, color: bool) -> int:
        """Score pawn structure: doubled, isolated, and passed pawns via bitwise ops."""
        score = 0

        # Doubled pawns: penalize extra pawns per file.
        for f in range(8):
            pawns_on_file = (my_pawns & FILES[f]).bit_count()
            if pawns_on_file > 1:
                score += (pawns_on_file - 1) * self.cfg.DOUBLED_PAWN_PENALTY

        # Per-pawn checks: isolation and passed status.
        for sq in chess.SquareSet(my_pawns):
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)

            # Isolated Check
            if (my_pawns & self.neighbor_masks[file]) == 0:
                score += self.cfg.ISOLATED_PAWN_PENALTY

            # Backward pawn: cannot advance safely and no friendly pawn support.
            # A pawn is backward if:
            #   1) No friendly pawn on adjacent files at same or lower rank.
            #   2) The stop square is controlled by an enemy pawn.
            if color == chess.WHITE:
                stop_sq = chess.square(file, rank + 1) if rank < 7 else None
                # Check if any friendly pawn on adjacent files at same or lower rank.
                has_support = False
                for adj_f in [file - 1, file + 1]:
                    if 0 <= adj_f <= 7:
                        for r in range(0, rank + 1):
                            if (1 << chess.square(adj_f, r)) & my_pawns:
                                has_support = True
                                break
                    if has_support:
                        break
                if not has_support and stop_sq is not None:
                    # Check if stop square is attacked by enemy pawn.
                    # Enemy (black) pawn at (adj_f, rank+2) attacks (file, rank+1).
                    stop_attacked = False
                    for adj_f in [file - 1, file + 1]:
                        if 0 <= adj_f <= 7 and rank + 2 <= 7:
                            if (1 << chess.square(adj_f, rank + 2)) & opp_pawns:
                                stop_attacked = True
                                break
                    if stop_attacked:
                        score -= 15  # Backward pawn penalty.
            else:
                stop_sq = chess.square(file, rank - 1) if rank > 0 else None
                has_support = False
                for adj_f in [file - 1, file + 1]:
                    if 0 <= adj_f <= 7:
                        for r in range(rank, 8):
                            if (1 << chess.square(adj_f, r)) & my_pawns:
                                has_support = True
                                break
                    if has_support:
                        break
                if not has_support and stop_sq is not None:
                    # Enemy (white) pawn at (adj_f, rank-2) attacks (file, rank-1).
                    stop_attacked = False
                    for adj_f in [file - 1, file + 1]:
                        if 0 <= adj_f <= 7 and rank - 2 >= 0:
                            if (1 << chess.square(adj_f, rank - 2)) & opp_pawns:
                                stop_attacked = True
                                break
                    if stop_attacked:
                        score -= 15

            # Passed Check
            passed_mask = (
                self.white_passed_masks[sq]
                if color == chess.WHITE
                else self.black_passed_masks[sq]
            )
            if (passed_mask & opp_pawns) == 0:
                rel_rank = rank if color == chess.WHITE else 7 - rank
                if 0 <= rel_rank < 8:
                    score += self.cfg.PASSED_PAWN_BONUS[rel_rank]

        return score

    def _eval_king_pawn_proximity_eg(
        self, board: chess.Board, white_pawns: int, black_pawns: int
    ) -> int:
        """Endgame king-pawn proximity: reward king supporting own passed pawns
        and approaching enemy passed pawns to block them.

        Uses Chebyshev distance (king moves diagonally).
        Bonuses:
          - Friendly king close to own passed pawn:  +(7 - dist) * 15
          - Opponent king far from our passed pawn:  +(dist - 1) * 10
        """
        score = 0

        w_king_sq = board.king(chess.WHITE)
        b_king_sq = board.king(chess.BLACK)
        if w_king_sq is None or b_king_sq is None:
            return 0

        w_king_file, w_king_rank = chess.square_file(w_king_sq), chess.square_rank(
            w_king_sq
        )
        b_king_file, b_king_rank = chess.square_file(b_king_sq), chess.square_rank(
            b_king_sq
        )

        # White passed pawns.
        for sq in chess.SquareSet(white_pawns):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            passed_mask = self.white_passed_masks[sq]
            if (passed_mask & black_pawns) == 0:
                # Friendly king proximity bonus.
                dist = max(abs(w_king_file - f), abs(w_king_rank - r))
                score += (7 - dist) * 15
                # Opponent king distance bonus (far = good for us).
                opp_dist = max(abs(b_king_file - f), abs(b_king_rank - r))
                score += (opp_dist - 1) * 10

        # Black passed pawns.
        for sq in chess.SquareSet(black_pawns):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            passed_mask = self.black_passed_masks[sq]
            if (passed_mask & white_pawns) == 0:
                # Friendly king proximity bonus.
                dist = max(abs(b_king_file - f), abs(b_king_rank - r))
                score -= (7 - dist) * 15
                # Opponent king distance bonus (far = good for black).
                opp_dist = max(abs(w_king_file - f), abs(w_king_rank - r))
                score -= (opp_dist - 1) * 10

        return score

    def _eval_advanced_passer_bonuses(
        self, board: chess.Board, white_pawns: int, black_pawns: int
    ) -> int:
        """Endgame bonuses: rook behind passed pawn and connected passed pawns."""
        ROOK_BEHIND_PASSER_BONUS = 50
        CONNECTED_PASSER_BONUS = 20  # Halved: each pair counted from both sides.
        # Extra EG bonus for advanced passers (by relative rank).
        # Rank 7 (about to promote) gets the largest bonus.
        ADVANCED_PASSER_EG_BONUS = [0, 0, 0, 0, 10, 50, 150, 250]
        score = 0

        # --- White ---
        w_rooks = board.pieces_mask(chess.ROOK, chess.WHITE)
        for sq in chess.SquareSet(white_pawns):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            passed_mask = self.white_passed_masks[sq]
            if (passed_mask & black_pawns) != 0:
                continue  # Not a passed pawn.
            # Extra bonus for advanced passers.
            if 0 <= r < 8:
                score += ADVANCED_PASSER_EG_BONUS[r]
            # Rook behind passed pawn: any white rook on same file behind (lower rank).
            file_mask = FILES[f]
            rooks_on_file = w_rooks & file_mask
            for rsq in chess.SquareSet(rooks_on_file):
                if chess.square_rank(rsq) < r:
                    score += ROOK_BEHIND_PASSER_BONUS
                    break
            # Connected passers: adjacent file has a passed pawn within 1 rank.
            for adj_f in [f - 1, f + 1]:
                if 0 <= adj_f <= 7:
                    adj_file_pawns = white_pawns & FILES[adj_f]
                    for adj_sq in chess.SquareSet(adj_file_pawns):
                        adj_passed = self.white_passed_masks[adj_sq]
                        if (adj_passed & black_pawns) == 0 and abs(
                            chess.square_rank(adj_sq) - r
                        ) <= 1:
                            score += CONNECTED_PASSER_BONUS
                            break

        # --- Black ---
        b_rooks = board.pieces_mask(chess.ROOK, chess.BLACK)
        for sq in chess.SquareSet(black_pawns):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            passed_mask = self.black_passed_masks[sq]
            if (passed_mask & white_pawns) != 0:
                continue
            # Extra bonus for advanced passers (use mirrored rank for black).
            rel_rank = 7 - r
            if 0 <= rel_rank < 8:
                score -= ADVANCED_PASSER_EG_BONUS[rel_rank]
            file_mask = FILES[f]
            rooks_on_file = b_rooks & file_mask
            for rsq in chess.SquareSet(rooks_on_file):
                if chess.square_rank(rsq) > r:
                    score -= ROOK_BEHIND_PASSER_BONUS
                    break
            for adj_f in [f - 1, f + 1]:
                if 0 <= adj_f <= 7:
                    adj_file_pawns = black_pawns & FILES[adj_f]
                    for adj_sq in chess.SquareSet(adj_file_pawns):
                        adj_passed = self.black_passed_masks[adj_sq]
                        if (adj_passed & white_pawns) == 0 and abs(
                            chess.square_rank(adj_sq) - r
                        ) <= 1:
                            score -= CONNECTED_PASSER_BONUS
                            break

        return score

    def _eval_mobility(
        self, board: chess.Board, white_pawns: int = 0, black_pawns: int = 0
    ) -> tuple:
        """Safe mobility score: attack squares minus enemy pawn-controlled squares.

        Returns (mg_mobility, eg_mobility) tuple with tapered weights.
        In endgames, rook mobility is much more valuable.
        """
        mg_score = 0
        eg_score = 0
        MG_MOBILITY_WEIGHTS = {
            chess.KNIGHT: 8,
            chess.BISHOP: 8,
            chess.ROOK: 5,
            chess.QUEEN: 3,
        }
        EG_MOBILITY_WEIGHTS = {
            chess.KNIGHT: 6,
            chess.BISHOP: 7,
            chess.ROOK: 10,
            chess.QUEEN: 5,
        }

        white_pawns_mask = white_pawns or board.pieces_mask(chess.PAWN, chess.WHITE)
        black_pawns_mask = black_pawns or board.pieces_mask(chess.PAWN, chess.BLACK)

        # Pawn attack masks via bitboard shifts.
        w_pawn_attacks = ((white_pawns_mask & ~FILES[7]) << 9) | (
            (white_pawns_mask & ~FILES[0]) << 7
        )

        b_pawn_attacks = ((black_pawns_mask & ~FILES[0]) >> 9) | (
            (black_pawns_mask & ~FILES[7]) >> 7
        )

        for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for sq in board.pieces(pt, chess.WHITE):
                attacks = board.attacks_mask(sq)
                # Exclude squares controlled by enemy pawns.
                safe_squares = attacks & ~b_pawn_attacks
                mob = chess.popcount(safe_squares)
                mg_score += mob * MG_MOBILITY_WEIGHTS[pt]
                eg_score += mob * EG_MOBILITY_WEIGHTS[pt]

            for sq in board.pieces(pt, chess.BLACK):
                attacks = board.attacks_mask(sq)
                safe_squares = attacks & ~w_pawn_attacks
                mob = chess.popcount(safe_squares)
                mg_score -= mob * MG_MOBILITY_WEIGHTS[pt]
                eg_score -= mob * EG_MOBILITY_WEIGHTS[pt]

        return mg_score, eg_score

    def _eval_king_safety(
        self,
        board: chess.Board,
        white_pawns: int,
        black_pawns: int,
    ) -> int:
        """King safety: castling rights, pawn shield, open files, attacker pressure,
        and enemy pawn proximity."""
        score = 0
        MISSING_SHIELD_PENALTY = 35
        PIECE_SHIELD_PENALTY = (
            15  # Reduced penalty when friendly non-pawn piece covers shield square.
        )
        LOST_CASTLING_PENALTY = 100
        KING_DRIFT_PENALTY = 50
        OPEN_FILE_PENALTY = self.cfg.KING_OPEN_FILE_PENALTY
        SEMI_OPEN_FILE_PENALTY = self.cfg.KING_SEMI_OPEN_FILE_PENALTY

        # Attacker weights for king zone pressure.
        ATTACKER_WEIGHTS = {
            chess.KNIGHT: 20,
            chess.BISHOP: 20,
            chess.ROOK: 30,
            chess.QUEEN: 50,
        }

        # Enemy pawn storm: penalties for pawns attacking king zone squares.
        PAWN_ZONE_ATTACK_PENALTY = 30  # Per king-zone square attacked by enemy pawn.
        PAWN_ZONE_QUEEN_EXTRA = 25  # Extra per square when enemy has queen.
        PAWN_NEAR_KING_BASE = 25  # Per-unit for advanced pawns near king.

        # Precompute pawn attack masks via bitboard shifts.
        b_pawn_atk = ((black_pawns & ~FILES[0]) >> 9) | ((black_pawns & ~FILES[7]) >> 7)
        w_pawn_atk = ((white_pawns & ~FILES[7]) << 9) | ((white_pawns & ~FILES[0]) << 7)

        # --- White King ---
        w_king_sq = board.king(chess.WHITE)
        if w_king_sq is not None:
            w_king_file = chess.square_file(w_king_sq)
            w_king_rank = chess.square_rank(w_king_sq)

            # Penalize lost castling rights.
            if w_king_sq == chess.E1:
                can_castle = (board.castling_rights & chess.BB_H1) or (
                    board.castling_rights & chess.BB_A1
                )
                if not can_castle:
                    score -= LOST_CASTLING_PENALTY
            # Penalize king drift when opponent has heavy pieces.
            elif w_king_sq not in [chess.G1, chess.C1, chess.B1]:
                if (
                    board.pieces(chess.QUEEN, chess.BLACK)
                    or len(board.pieces(chess.ROOK, chess.BLACK)) >= 2
                ):
                    score -= KING_DRIFT_PENALTY

            # Pawn shield: full credit for pawns, partial credit for friendly pieces.
            shield_mask = self.king_shield_masks[w_king_sq] if w_king_sq < 56 else 0
            if shield_mask:
                pawns_in_shield = (
                    shield_mask & board.pieces_mask(chess.PAWN, chess.WHITE)
                ).bit_count()
                pieces_in_shield = (
                    shield_mask & board.occupied_co[chess.WHITE]
                ).bit_count() - pawns_in_shield
                max_shield = chess.popcount(shield_mask)
                fully_empty = max_shield - pawns_in_shield - pieces_in_shield
                if fully_empty > 0:
                    score -= fully_empty * MISSING_SHIELD_PENALTY
                if pieces_in_shield > 0:
                    score -= pieces_in_shield * PIECE_SHIELD_PENALTY

            # Open/semi-open files adjacent to king.
            for f in range(max(0, w_king_file - 1), min(8, w_king_file + 2)):
                file_mask = FILES[f]
                own_pawns_on_file = file_mask & white_pawns
                opp_pawns_on_file = file_mask & black_pawns
                if not own_pawns_on_file and not opp_pawns_on_file:
                    score -= OPEN_FILE_PENALTY
                elif not own_pawns_on_file:
                    score -= SEMI_OPEN_FILE_PENALTY

            # Enemy pawn proximity: penalize advanced Black pawns near White king.
            w_king_zone_mask = (
                board.attacks_mask(w_king_sq) | chess.BB_SQUARES[w_king_sq]
            )
            pawn_zone_attacks = chess.popcount(b_pawn_atk & w_king_zone_mask)

            if pawn_zone_attacks > 0:
                score -= pawn_zone_attacks * PAWN_ZONE_ATTACK_PENALTY
                if board.pieces(chess.QUEEN, chess.BLACK):
                    score -= pawn_zone_attacks * PAWN_ZONE_QUEEN_EXTRA

            # Penalty for advanced enemy pawns within distance 2 of king.
            for sq in chess.SquareSet(black_pawns):
                p_file = chess.square_file(sq)
                p_rank = chess.square_rank(sq)
                dist = max(abs(p_file - w_king_file), abs(p_rank - w_king_rank))
                if dist <= 2:
                    advancement = 7 - p_rank  # Black pawn: rank 2 → advancement 5.
                    if advancement >= 4:
                        score -= (advancement - 3) * PAWN_NEAR_KING_BASE * (3 - dist)

            # King zone attacker pressure (non-linear scaling).
            if board.pieces(chess.QUEEN, chess.BLACK):
                attacker_score = 0
                attacker_count = 0
                king_zone = board.attacks(w_king_sq) | chess.SquareSet(
                    chess.BB_SQUARES[w_king_sq]
                )
                for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    for sq in board.pieces(pt, chess.BLACK):
                        piece_attacks = board.attacks_mask(sq)
                        if piece_attacks & int(king_zone):
                            attacker_count += 1
                            attacker_score += ATTACKER_WEIGHTS[pt]

                # Scale non-linearly: more attackers = disproportionate danger.
                if attacker_count >= 2:
                    score -= attacker_score * attacker_count // 2

        # --- Black King ---
        b_king_sq = board.king(chess.BLACK)
        if b_king_sq is not None:
            b_king_file = chess.square_file(b_king_sq)
            b_king_rank = chess.square_rank(b_king_sq)

            # Castling rights check.
            if b_king_sq == chess.E8:
                can_castle = (board.castling_rights & chess.BB_H8) or (
                    board.castling_rights & chess.BB_A8
                )
                if not can_castle:
                    score += LOST_CASTLING_PENALTY
            elif b_king_sq not in [chess.G8, chess.C8, chess.B8]:
                if (
                    board.pieces(chess.QUEEN, chess.WHITE)
                    or len(board.pieces(chess.ROOK, chess.WHITE)) >= 2
                ):
                    score += KING_DRIFT_PENALTY

            # Pawn shield (rank below for Black): partial credit for non-pawn pieces.
            f, r = b_king_file, b_king_rank
            shield_penalty = 0
            if r > 0:
                for df in [-1, 0, 1]:
                    if 0 <= f + df <= 7:
                        sq = chess.square(f + df, r - 1)
                        piece = board.piece_at(sq)
                        if (
                            piece
                            and piece.piece_type == chess.PAWN
                            and piece.color == chess.BLACK
                        ):
                            pass  # Full shield, no penalty.
                        elif piece and piece.color == chess.BLACK:
                            shield_penalty += PIECE_SHIELD_PENALTY
                        else:
                            shield_penalty += MISSING_SHIELD_PENALTY
            if shield_penalty > 0:
                score += shield_penalty

            # Open/semi-open files near black king.
            for f_idx in range(max(0, b_king_file - 1), min(8, b_king_file + 2)):
                file_mask = FILES[f_idx]
                own_pawns_on_file = file_mask & black_pawns
                opp_pawns_on_file = file_mask & white_pawns
                if not own_pawns_on_file and not opp_pawns_on_file:
                    score += OPEN_FILE_PENALTY
                elif not own_pawns_on_file:
                    score += SEMI_OPEN_FILE_PENALTY

            # Enemy pawn proximity: penalize advanced White pawns near Black king.
            b_king_zone_mask = (
                board.attacks_mask(b_king_sq) | chess.BB_SQUARES[b_king_sq]
            )
            w_pawn_zone_attacks = chess.popcount(w_pawn_atk & b_king_zone_mask)

            if w_pawn_zone_attacks > 0:
                score += w_pawn_zone_attacks * PAWN_ZONE_ATTACK_PENALTY
                if board.pieces(chess.QUEEN, chess.WHITE):
                    score += w_pawn_zone_attacks * PAWN_ZONE_QUEEN_EXTRA

            # Penalty for advanced White pawns within distance 2 of Black king.
            for sq in chess.SquareSet(white_pawns):
                p_file = chess.square_file(sq)
                p_rank = chess.square_rank(sq)
                dist = max(abs(p_file - b_king_file), abs(p_rank - b_king_rank))
                if dist <= 2:
                    advancement = p_rank  # White pawn: rank 5 → advancement 5.
                    if advancement >= 4:
                        score += (advancement - 3) * PAWN_NEAR_KING_BASE * (3 - dist)

            # King zone attacker pressure.
            if board.pieces(chess.QUEEN, chess.WHITE):
                attacker_score = 0
                attacker_count = 0
                king_zone = board.attacks(b_king_sq) | chess.SquareSet(
                    chess.BB_SQUARES[b_king_sq]
                )
                for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    for sq in board.pieces(pt, chess.WHITE):
                        piece_attacks = board.attacks_mask(sq)
                        if piece_attacks & int(king_zone):
                            attacker_count += 1
                            attacker_score += ATTACKER_WEIGHTS[pt]

                if attacker_count >= 2:
                    score += attacker_score * attacker_count // 2

        return score

    def _eval_threats(self, board: chess.Board) -> int:
        """Detect hanging pieces and pawn-attack threats via bitboard ops."""
        score = 0
        THREAT_WEIGHTS_LIST = [0, 40, 80, 85, 120, 200, 0]  # Indexed by piece_type

        # Precompute pawn attack masks.
        wp = board.pieces_mask(chess.PAWN, chess.WHITE)
        bp = board.pieces_mask(chess.PAWN, chess.BLACK)
        w_pawn_atk = ((wp & ~FILES[7]) << 9) | ((wp & ~FILES[0]) << 7)
        b_pawn_atk = ((bp & ~FILES[0]) >> 9) | ((bp & ~FILES[7]) >> 7)

        # Check each color's non-pawn pieces attacked by enemy pawns
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            weight = THREAT_WEIGHTS_LIST[pt]

            # White pieces attacked by black pawns
            w_pieces = board.pieces_mask(pt, chess.WHITE)
            threatened_w = w_pieces & b_pawn_atk
            if threatened_w:
                score -= weight * chess.popcount(threatened_w)

            # Black pieces attacked by white pawns
            b_pieces = board.pieces_mask(pt, chess.BLACK)
            threatened_b = b_pieces & w_pawn_atk
            if threatened_b:
                score += weight * chess.popcount(threatened_b)

        # Hanging piece detection (skip already-counted pawn-attacked pieces).
        for color in (chess.WHITE, chess.BLACK):
            enemy = not color
            sign = 1 if color == chess.BLACK else -1

            for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                weight = THREAT_WEIGHTS_LIST[pt]
                if weight == 0:
                    continue

                for sq in board.pieces(pt, color):
                    # Skip if already counted as pawn-attacked (non-pawns only)
                    if pt > chess.PAWN:
                        sq_mask = 1 << sq
                        enemy_pawn_atk = (
                            w_pawn_atk if color == chess.BLACK else b_pawn_atk
                        )
                        if sq_mask & enemy_pawn_atk:
                            continue  # Already scored above

                    atk_mask = board.attackers_mask(enemy, sq)
                    if not atk_mask:
                        continue

                    def_mask = board.attackers_mask(color, sq)
                    n_def = chess.popcount(def_mask)

                    if n_def == 0:
                        score += sign * weight
                    elif chess.popcount(atk_mask) > n_def:
                        score += sign * (weight >> 1)

        return score

    def _eval_development(self, board: chess.Board, phase: int) -> int:
        """Penalize minor pieces on starting squares during opening/middlegame."""
        if phase < 14:  # Endgame: skip.
            return 0

        score = 0
        penalty = self.cfg.UNDEVELOPED_PENALTY
        phase_factor = min(phase, 24) / 24.0  # Stronger in opening.

        # White undeveloped knights
        for sq in [chess.B1, chess.G1]:
            p = board.piece_at(sq)
            if p and p.piece_type == chess.KNIGHT and p.color == chess.WHITE:
                score -= int(penalty * phase_factor)

        # White undeveloped bishops
        for sq in [chess.C1, chess.F1]:
            p = board.piece_at(sq)
            if p and p.piece_type == chess.BISHOP and p.color == chess.WHITE:
                score -= int(penalty * phase_factor)

        # Black undeveloped knights
        for sq in [chess.B8, chess.G8]:
            p = board.piece_at(sq)
            if p and p.piece_type == chess.KNIGHT and p.color == chess.BLACK:
                score += int(penalty * phase_factor)

        # Black undeveloped bishops
        for sq in [chess.C8, chess.F8]:
            p = board.piece_at(sq)
            if p and p.piece_type == chess.BISHOP and p.color == chess.BLACK:
                score += int(penalty * phase_factor)

        return score

    def _eval_rook_files(
        self, board: chess.Board, white_pawns: int, black_pawns: int
    ) -> int:
        """Bonus for rooks on open (no pawns) or semi-open (no friendly pawns) files."""
        score = 0
        open_bonus = self.cfg.ROOK_OPEN_FILE_BONUS
        semi_open_bonus = self.cfg.ROOK_SEMI_OPEN_FILE_BONUS

        for sq in board.pieces(chess.ROOK, chess.WHITE):
            f = chess.square_file(sq)
            file_mask = FILES[f]
            own_pawns = file_mask & white_pawns
            opp_pawns = file_mask & black_pawns
            if not own_pawns and not opp_pawns:
                score += open_bonus
            elif not own_pawns:
                score += semi_open_bonus

        for sq in board.pieces(chess.ROOK, chess.BLACK):
            f = chess.square_file(sq)
            file_mask = FILES[f]
            own_pawns = file_mask & black_pawns
            opp_pawns = file_mask & white_pawns
            if not own_pawns and not opp_pawns:
                score -= open_bonus
            elif not own_pawns:
                score -= semi_open_bonus

        return score

    def _eval_pawn_king_proximity(
        self, board: chess.Board, white_pawns: int, black_pawns: int
    ) -> int:
        """Penalize advanced pawns near the friendly king that weaken the shelter."""
        score = 0
        PAWN_ADVANCE_PENALTY = 15  # Per rank of advancement near king

        # --- White King ---
        w_king_sq = board.king(chess.WHITE)
        if w_king_sq is not None:
            king_file = chess.square_file(w_king_sq)
            king_rank = chess.square_rank(w_king_sq)

            # Only relevant when king is near the back rank.
            if king_rank <= 1:
                for f in range(max(0, king_file - 1), min(8, king_file + 2)):
                    file_mask = FILES[f]
                    pawns_on_file = file_mask & white_pawns
                    if pawns_on_file:
                        for sq in chess.SquareSet(pawns_on_file):
                            pawn_rank = chess.square_rank(sq)
                            # Ranks advanced from starting position.
                            advance = pawn_rank - 1
                            if advance >= 2:
                                score -= advance * PAWN_ADVANCE_PENALTY

        # --- Black King ---
        b_king_sq = board.king(chess.BLACK)
        if b_king_sq is not None:
            king_file = chess.square_file(b_king_sq)
            king_rank = chess.square_rank(b_king_sq)

            # Only relevant when king is near the back rank.
            if king_rank >= 6:
                for f in range(max(0, king_file - 1), min(8, king_file + 2)):
                    file_mask = FILES[f]
                    pawns_on_file = file_mask & black_pawns
                    if pawns_on_file:
                        for sq in chess.SquareSet(pawns_on_file):
                            pawn_rank = chess.square_rank(sq)
                            # Black pawns advance downward from rank 6.
                            advance = 6 - pawn_rank
                            if advance >= 2:
                                score += advance * PAWN_ADVANCE_PENALTY

        return score

    def _eval_knight_outposts(
        self, board: chess.Board, white_pawns: int, black_pawns: int
    ) -> int:
        """Bonus for knights on outpost squares; penalty for unsupported rim knights."""
        score = 0
        OUTPOST_BONUS = 25
        RIM_UNSUPPORTED_PENALTY = 15

        for sq in board.pieces(chess.KNIGHT, chess.WHITE):
            f = chess.square_file(sq)
            r = chess.square_rank(sq)

            # Outpost: in opponent's territory (ranks 4-6).
            if 3 <= r <= 5:
                # Friendly pawn support?
                supported = False
                if f > 0 and r > 0:
                    support_sq = chess.square(f - 1, r - 1)
                    p = board.piece_at(support_sq)
                    if p and p.piece_type == chess.PAWN and p.color == chess.WHITE:
                        supported = True
                if f < 7 and r > 0:
                    support_sq = chess.square(f + 1, r - 1)
                    p = board.piece_at(support_sq)
                    if p and p.piece_type == chess.PAWN and p.color == chess.WHITE:
                        supported = True

                # No enemy pawn can attack this square?
                safe = True
                for rr in range(r + 1, 8):  # Check ranks above.
                    if f > 0:
                        atk_sq = chess.square(f - 1, rr)
                        if (1 << atk_sq) & black_pawns:
                            safe = False
                            break
                    if f < 7:
                        atk_sq = chess.square(f + 1, rr)
                        if (1 << atk_sq) & black_pawns:
                            safe = False
                            break

                if supported and safe:
                    score += OUTPOST_BONUS

            # Rim penalty: unsupported knight on a/h file.
            if f == 0 or f == 7:
                has_support = False
                if r > 0:
                    for df in [-1, 1]:
                        sf = f + df
                        if 0 <= sf <= 7:
                            support_sq = chess.square(sf, r - 1)
                            p = board.piece_at(support_sq)
                            if (
                                p
                                and p.piece_type == chess.PAWN
                                and p.color == chess.WHITE
                            ):
                                has_support = True
                if not has_support:
                    score -= RIM_UNSUPPORTED_PENALTY

        for sq in board.pieces(chess.KNIGHT, chess.BLACK):
            f = chess.square_file(sq)
            r = chess.square_rank(sq)

            # Outpost: in opponent's territory (ranks 3-5 for Black).
            if 2 <= r <= 4:
                supported = False
                if f > 0 and r < 7:
                    support_sq = chess.square(f - 1, r + 1)
                    p = board.piece_at(support_sq)
                    if p and p.piece_type == chess.PAWN and p.color == chess.BLACK:
                        supported = True
                if f < 7 and r < 7:
                    support_sq = chess.square(f + 1, r + 1)
                    p = board.piece_at(support_sq)
                    if p and p.piece_type == chess.PAWN and p.color == chess.BLACK:
                        supported = True

                safe = True
                for rr in range(0, r):  # Stop below knight's rank
                    if f > 0:
                        atk_sq = chess.square(f - 1, rr)
                        if (1 << atk_sq) & white_pawns:
                            safe = False
                            break
                    if f < 7:
                        atk_sq = chess.square(f + 1, rr)
                        if (1 << atk_sq) & white_pawns:
                            safe = False
                            break

                if supported and safe:
                    score -= OUTPOST_BONUS

            # Rim penalty for Black.
            if f == 0 or f == 7:
                has_support = False
                if r < 7:
                    for df in [-1, 1]:
                        sf = f + df
                        if 0 <= sf <= 7:
                            support_sq = chess.square(sf, r + 1)
                            p = board.piece_at(support_sq)
                            if (
                                p
                                and p.piece_type == chess.PAWN
                                and p.color == chess.BLACK
                            ):
                                has_support = True
                if not has_support:
                    score += RIM_UNSUPPORTED_PENALTY

        return score

    def _eval_queen_activity(self, board: chess.Board, phase: int) -> int:
        """Penalize passive queen placement; reward central positioning."""
        if phase < 4:  # Endgame: skip.
            return 0

        score = 0

        # Bonus by distance from board center.
        QUEEN_POSITION_BONUS = [8, 5, 2, 0]

        BACK_RANK_PENALTY = 15

        for sq in board.pieces(chess.QUEEN, chess.WHITE):
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            # Distance from center of board
            file_dist = min(abs(f - 3), abs(f - 4))
            rank_dist = min(abs(r - 3), abs(r - 4))
            center_dist = max(file_dist, rank_dist)
            center_dist = min(center_dist, 3)
            score += QUEEN_POSITION_BONUS[center_dist]

            # Back-rank penalty (mid-game only).
            if r == 0 and phase < 22:
                score -= BACK_RANK_PENALTY

        for sq in board.pieces(chess.QUEEN, chess.BLACK):
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            file_dist = min(abs(f - 3), abs(f - 4))
            rank_dist = min(abs(r - 3), abs(r - 4))
            center_dist = max(file_dist, rank_dist)
            center_dist = min(center_dist, 3)
            score -= QUEEN_POSITION_BONUS[center_dist]

            # Back-rank penalty.
            if r == 7 and phase < 22:
                score += BACK_RANK_PENALTY

        return score

    def _eval_rook_7th(self, board: chess.Board) -> tuple:
        """Bonus for rooks on 7th rank (2nd for Black). Especially strong
        when enemy king is on 8th rank or there are enemy pawns on 7th."""
        ROOK_7TH_MG = 20
        ROOK_7TH_EG = 40
        ROOK_7TH_KING_BONUS = 10  # Extra when enemy king on 8th.
        mg = 0
        eg = 0

        b_king_sq = board.king(chess.BLACK)
        w_king_sq = board.king(chess.WHITE)

        for sq in board.pieces(chess.ROOK, chess.WHITE):
            if chess.square_rank(sq) == 6:  # 7th rank (0-indexed).
                mg += ROOK_7TH_MG
                eg += ROOK_7TH_EG
                if b_king_sq is not None and chess.square_rank(b_king_sq) == 7:
                    mg += ROOK_7TH_KING_BONUS
                    eg += ROOK_7TH_KING_BONUS

        for sq in board.pieces(chess.ROOK, chess.BLACK):
            if chess.square_rank(sq) == 1:  # 2nd rank (7th for Black).
                mg -= ROOK_7TH_MG
                eg -= ROOK_7TH_EG
                if w_king_sq is not None and chess.square_rank(w_king_sq) == 0:
                    mg -= ROOK_7TH_KING_BONUS
                    eg -= ROOK_7TH_KING_BONUS

        return mg, eg

    def _eval_connected_rooks(self, board: chess.Board) -> int:
        """Bonus when two rooks can see each other (same rank or file, no pieces between)."""
        CONNECTED_ROOK_BONUS = 15
        score = 0

        for color in (chess.WHITE, chess.BLACK):
            rooks = list(board.pieces(chess.ROOK, color))
            sign = 1 if color == chess.WHITE else -1
            if len(rooks) >= 2:
                r1, r2 = rooks[0], rooks[1]
                # Check if they attack each other (line of sight).
                if r2 in board.attacks(r1):
                    score += sign * CONNECTED_ROOK_BONUS

        return score

    def _eval_space(
        self, board: chess.Board, white_pawns: int, black_pawns: int
    ) -> int:
        """Space advantage: reward pawns advanced into center ranks (fast bitboard)."""
        SPACE_WEIGHT = 2
        # Mask for ranks 3-6 (0-indexed ranks 2-5).
        RANK_3_6_MASK = 0x0000FFFFFFFF0000
        w_space = chess.popcount(white_pawns & RANK_3_6_MASK)
        b_space = chess.popcount(black_pawns & RANK_3_6_MASK)
        return (w_space - b_space) * SPACE_WEIGHT

    def _eval_bishop_knight_imbalance(
        self, board: chess.Board, white_pawns: int, black_pawns: int
    ) -> int:
        """Bishop vs Knight imbalance: bishops are better in open positions,
        knights are better in closed positions with many pawns.

        Metric: total pawn count. Many pawns = closed = knight bonus.
        Few pawns = open = bishop bonus.
        """
        total_pawns = chess.popcount(white_pawns | black_pawns)
        # threshold: 10 pawns = neutral, <10 = open (bishop bonus), >10 = closed (knight bonus)
        openness = 10 - total_pawns  # positive = open, negative = closed
        BONUS_PER_PAWN = 5  # 5cp per pawn difference from threshold

        score = 0
        w_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
        w_knights = len(board.pieces(chess.KNIGHT, chess.WHITE))
        b_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
        b_knights = len(board.pieces(chess.KNIGHT, chess.BLACK))

        # White: bonus for bishops in open, knights in closed
        if w_bishops > w_knights:
            score += openness * BONUS_PER_PAWN  # open = positive bonus for bishop side
        elif w_knights > w_bishops:
            score -= openness * BONUS_PER_PAWN  # open = penalty for knight side

        # Black: same logic, inverted
        if b_bishops > b_knights:
            score -= openness * BONUS_PER_PAWN
        elif b_knights > b_bishops:
            score += openness * BONUS_PER_PAWN

        return score
