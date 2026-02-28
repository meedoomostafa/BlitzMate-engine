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
        mg_score += int(w_pawn_struct * 0.5)
        mg_score -= int(b_pawn_struct * 0.5)
        eg_score += int(w_pawn_struct * 1.0)
        eg_score -= int(b_pawn_struct * 1.0)

        # 4.1 Bishop pair bonus.
        if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
            mg_score += self.cfg.BISHOP_PAIR_BONUS
            eg_score += self.cfg.BISHOP_PAIR_BONUS
        if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
            mg_score -= self.cfg.BISHOP_PAIR_BONUS
            eg_score -= self.cfg.BISHOP_PAIR_BONUS

        # 4.2 Mobility.
        mob_score = self._eval_mobility(board, white_pawns, black_pawns)
        mg_score += mob_score
        eg_score += mob_score

        # 4.3 King safety (critical in middlegame).
        ks_score = self._eval_king_safety(board, white_pawns, black_pawns)
        mg_score += ks_score
        eg_score += int(ks_score * 0.15)

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

        # Tapered eval: blend MG/EG scores by remaining material.
        phase = min(phase, 24)
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

    def _eval_mobility(
        self, board: chess.Board, white_pawns: int = 0, black_pawns: int = 0
    ) -> int:
        """Safe mobility score: attack squares minus enemy pawn-controlled squares."""
        score = 0
        MOBILITY_WEIGHTS = {
            chess.KNIGHT: 8,
            chess.BISHOP: 8,
            chess.ROOK: 5,
            chess.QUEEN: 3,
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
                score += mob * MOBILITY_WEIGHTS[pt]

            for sq in board.pieces(pt, chess.BLACK):
                attacks = board.attacks_mask(sq)
                safe_squares = attacks & ~w_pawn_attacks
                mob = chess.popcount(safe_squares)
                score -= mob * MOBILITY_WEIGHTS[pt]

        return score

    def _eval_king_safety(
        self,
        board: chess.Board,
        white_pawns: int,
        black_pawns: int,
    ) -> int:
        """King safety: castling rights, pawn shield, open files, and attacker pressure."""
        score = 0
        MISSING_SHIELD_PENALTY = 35
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

            # Pawn shield
            shield_mask = self.king_shield_masks[w_king_sq] if w_king_sq < 56 else 0
            if shield_mask:
                pawns_in_shield = (
                    shield_mask & board.pieces_mask(chess.PAWN, chess.WHITE)
                ).bit_count()
                if pawns_in_shield < 3:
                    score -= (3 - pawns_in_shield) * MISSING_SHIELD_PENALTY

            # Open/semi-open files adjacent to king.
            for f in range(max(0, w_king_file - 1), min(8, w_king_file + 2)):
                file_mask = FILES[f]
                own_pawns_on_file = file_mask & white_pawns
                opp_pawns_on_file = file_mask & black_pawns
                if not own_pawns_on_file and not opp_pawns_on_file:
                    score -= OPEN_FILE_PENALTY
                elif not own_pawns_on_file:
                    score -= SEMI_OPEN_FILE_PENALTY

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

            # Pawn shield (rank below for Black).
            f, r = b_king_file, b_king_rank
            missing = 0
            if r > 0:
                for df in [-1, 0, 1]:
                    if 0 <= f + df <= 7:
                        sq = chess.square(f + df, r - 1)
                        piece = board.piece_at(sq)
                        if not (
                            piece
                            and piece.piece_type == chess.PAWN
                            and piece.color == chess.BLACK
                        ):
                            missing += 1
            if missing > 0:
                score += missing * MISSING_SHIELD_PENALTY

            # Open/semi-open files near black king.
            for f_idx in range(max(0, b_king_file - 1), min(8, b_king_file + 2)):
                file_mask = FILES[f_idx]
                own_pawns_on_file = file_mask & black_pawns
                opp_pawns_on_file = file_mask & white_pawns
                if not own_pawns_on_file and not opp_pawns_on_file:
                    score += OPEN_FILE_PENALTY
                elif not own_pawns_on_file:
                    score += SEMI_OPEN_FILE_PENALTY

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
        if phase < 10:  # Endgame: skip.
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
