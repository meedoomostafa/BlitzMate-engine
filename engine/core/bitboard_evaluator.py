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
        """
        Generates bitmasks representing the "front span" of a pawn.

        A pawn is passed if no enemy pawns exist in its front span.
        The span includes the file ahead of the pawn and both adjacent files.
        """
        for sq in range(64):
            f, r = chess.square_file(sq), chess.square_rank(sq)

            # White: Spans from rank+1 to Rank 8
            w_front = 0
            for rr in range(r + 1, 8):
                w_front |= 1 << chess.square(f, rr)
                if f > 0:
                    w_front |= 1 << chess.square(f - 1, rr)
                if f < 7:
                    w_front |= 1 << chess.square(f + 1, rr)
            self.white_passed_masks[sq] = w_front

            # Black: Spans from rank-1 to Rank 1
            b_front = 0
            for rr in range(0, r):
                b_front |= 1 << chess.square(f, rr)
                if f > 0:
                    b_front |= 1 << chess.square(f - 1, rr)
                if f < 7:
                    b_front |= 1 << chess.square(f + 1, rr)
            self.black_passed_masks[sq] = b_front

    def _init_king_masks(self) -> None:
        """
        Generates bitmasks for the squares directly guarding the king.
        Used to detect missing pawn shields (King Safety).
        """
        for sq in range(64):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            w_shield = 0
            # For White, the shield is ideally on the rank above.
            if r < 7:
                w_shield |= 1 << chess.square(f, r + 1)
                if f > 0:
                    w_shield |= 1 << chess.square(f - 1, r + 1)
                if f < 7:
                    w_shield |= 1 << chess.square(f + 1, r + 1)
            self.king_shield_masks[sq] = w_shield

    def evaluate(self, board: chess.Board) -> int:
        """
        Calculates the static value of the current board position.

        Args:
            board (chess.Board): The position to evaluate.

        Returns:
            int: The score in centipawns (100 cp = 1 Pawn).
                 Positive values favor the side to move.
        """
        if board.is_insufficient_material():
            return 0
        if board.is_stalemate():
            return 0

        # --- 1. Global State Retrieval ---
        # Retrieve bitboards once to avoid repeated calls
        white_pawns = board.pieces_mask(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces_mask(chess.PAWN, chess.BLACK)

        mg_score = 0  # Middle Game Score
        eg_score = 0  # End Game Score
        phase = 0  # Game Phase (0=Endgame, 24=Opening)

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
                # Mirror the square index for Black (A1 becomes A8)
                for sq in squares:
                    mg_score -= table_mg[chess.square_mirror(sq)]
            if table_eg:
                for sq in squares:
                    eg_score -= table_eg[chess.square_mirror(sq)]

        # 3.1 Base Material & PST
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

        # 3.2 Advanced Structure (Doubled, Isolated, Passed)
        w_pawn_struct = self._eval_pawns_bitwise(white_pawns, black_pawns, chess.WHITE)
        b_pawn_struct = self._eval_pawns_bitwise(black_pawns, white_pawns, chess.BLACK)

        # Pawn structure is weighted differently in MG vs EG
        mg_score += int(w_pawn_struct * 0.5)
        mg_score -= int(b_pawn_struct * 0.5)
        eg_score += int(w_pawn_struct * 1.0)
        eg_score -= int(b_pawn_struct * 1.0)

        # 4.1 Bishop Pair Bonus
        if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
            mg_score += self.cfg.BISHOP_PAIR_BONUS
            eg_score += self.cfg.BISHOP_PAIR_BONUS
        if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
            mg_score -= self.cfg.BISHOP_PAIR_BONUS
            eg_score -= self.cfg.BISHOP_PAIR_BONUS

        # 4.2 Mobility (Encourage Development)
        mob_score = self._eval_mobility(board, white_pawns, black_pawns)
        mg_score += mob_score
        eg_score += mob_score

        # 4.3 King Safety (Critical in Middle Game)
        ks_score = self._eval_king_safety(board, white_pawns, black_pawns)
        mg_score += ks_score
        eg_score += int(ks_score * 0.15)

        # 4.4 Threat Detection
        threat_score = self._eval_threats(board)
        mg_score += threat_score
        eg_score += threat_score

        # 4.5 Development (penalize undeveloped pieces in MG)
        dev_score = self._eval_development(board, phase)
        mg_score += dev_score

        # 4.6 Rook on open/semi-open files
        rook_file_score = self._eval_rook_files(board, white_pawns, black_pawns)
        mg_score += rook_file_score
        eg_score += rook_file_score

        # 4.7 Pawn advance near king penalty (prevent self-weakening)
        pawn_storm_score = self._eval_pawn_king_proximity(
            board, white_pawns, black_pawns
        )
        mg_score += pawn_storm_score

        # 4.8 Knight outpost evaluation
        outpost_score = self._eval_knight_outposts(board, white_pawns, black_pawns)
        mg_score += outpost_score
        eg_score += outpost_score

        # 4.9 Queen centrality (discourage passive queen placement)
        queen_score = self._eval_queen_activity(board, phase)
        mg_score += queen_score

        # Blends MG and EG scores based on remaining material on the board.
        phase = min(phase, 24)
        final_score = (mg_score * phase + eg_score * (24 - phase)) // 24

        # Return score relative to the side to move (Negamax requirement)
        if board.turn == chess.BLACK:
            return -final_score
        return final_score

    def _eval_pawns_bitwise(self, my_pawns: int, opp_pawns: int, color: bool) -> int:
        """
        Calculates score for pawn structure features using bitwise operations.

        Features Evaluated:
            - Doubled Pawns: Penalty for two pawns on the same file.
            - Isolated Pawns: Penalty for having no friendly pawns on adjacent files.
            - Passed Pawns: Bonus for having no enemy pawns blocking the path to promotion.
        """
        score = 0

        # 1. Doubled Pawns Check — count pawns per file, penalize extras
        for f in range(8):
            pawns_on_file = (my_pawns & FILES[f]).bit_count()
            if pawns_on_file > 1:
                score += (pawns_on_file - 1) * self.cfg.DOUBLED_PAWN_PENALTY

        # 2. File-based Checks
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
        """
        Calculates a safe mobility score.

        Uses attack squares minus squares controlled by enemy pawns
        for a more accurate assessment of piece activity.
        """
        score = 0
        MOBILITY_WEIGHTS = {
            chess.KNIGHT: 8,
            chess.BISHOP: 8,
            chess.ROOK: 5,
            chess.QUEEN: 3,
        }

        # Use cached pawn masks if provided, otherwise compute
        white_pawns_mask = white_pawns or board.pieces_mask(chess.PAWN, chess.WHITE)
        black_pawns_mask = black_pawns or board.pieces_mask(chess.PAWN, chess.BLACK)

        # White pawn attacks: shift NE and NW
        w_pawn_attacks = ((white_pawns_mask & ~FILES[7]) << 9) | (
            (white_pawns_mask & ~FILES[0]) << 7
        )
        # Black pawn attacks: shift SE and SW
        b_pawn_attacks = ((black_pawns_mask & ~FILES[0]) >> 9) | (
            (black_pawns_mask & ~FILES[7]) >> 7
        )

        for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for sq in board.pieces(pt, chess.WHITE):
                attacks = board.attacks_mask(sq)
                # Safe mobility: exclude squares attacked by enemy pawns
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
        """
        Enhanced king safety evaluation.

        Evaluates:
            - Lost castling rights (major penalty)
            - King drifting from safe squares (with queen on board)
            - Missing pawn shield
            - Open/semi-open files near king
            - Attacker pressure on king zone
        """
        score = 0
        MISSING_SHIELD_PENALTY = 35
        LOST_CASTLING_PENALTY = 100
        KING_DRIFT_PENALTY = 50
        OPEN_FILE_PENALTY = self.cfg.KING_OPEN_FILE_PENALTY
        SEMI_OPEN_FILE_PENALTY = self.cfg.KING_SEMI_OPEN_FILE_PENALTY

        # Attacker weights for king zone pressure
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

            # Castling rights check
            if w_king_sq == chess.E1:
                can_castle = (board.castling_rights & chess.BB_H1) or (
                    board.castling_rights & chess.BB_A1
                )
                if not can_castle:
                    score -= LOST_CASTLING_PENALTY
            # Also penalize if king moved but NOT to a castled position
            elif w_king_sq not in [chess.G1, chess.C1, chess.B1]:
                # Check if the opponent has heavy pieces that can attack
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

            # Open/semi-open files near king
            for f in range(max(0, w_king_file - 1), min(8, w_king_file + 2)):
                file_mask = FILES[f]
                own_pawns_on_file = file_mask & white_pawns
                opp_pawns_on_file = file_mask & black_pawns
                if not own_pawns_on_file and not opp_pawns_on_file:
                    score -= OPEN_FILE_PENALTY
                elif not own_pawns_on_file:
                    score -= SEMI_OPEN_FILE_PENALTY

            # King zone attackers
            if board.pieces(chess.QUEEN, chess.BLACK):
                attacker_score = 0
                attacker_count = 0
                # King zone: squares around the king
                king_zone = board.attacks(w_king_sq) | chess.SquareSet(
                    chess.BB_SQUARES[w_king_sq]
                )
                for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    for sq in board.pieces(pt, chess.BLACK):
                        piece_attacks = board.attacks_mask(sq)
                        if piece_attacks & int(king_zone):
                            attacker_count += 1
                            attacker_score += ATTACKER_WEIGHTS[pt]

                # Scale attack score non-linearly (more attackers = disproportionately dangerous)
                if attacker_count >= 2:
                    score -= attacker_score * attacker_count // 2

        # --- Black King ---
        b_king_sq = board.king(chess.BLACK)
        if b_king_sq is not None:
            b_king_file = chess.square_file(b_king_sq)
            b_king_rank = chess.square_rank(b_king_sq)

            # Castling rights check
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

            # Pawn shield (for Black, check rank below)
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

            # Open/semi-open files near black king
            for f_idx in range(max(0, b_king_file - 1), min(8, b_king_file + 2)):
                file_mask = FILES[f_idx]
                own_pawns_on_file = file_mask & black_pawns
                opp_pawns_on_file = file_mask & white_pawns
                if not own_pawns_on_file and not opp_pawns_on_file:
                    score += OPEN_FILE_PENALTY
                elif not own_pawns_on_file:
                    score += SEMI_OPEN_FILE_PENALTY

            # King zone attackers
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
        """
        Evaluates immediate tactical threats using bitboard operations.

        Detects:
            - Pieces attacked by enemy pawns (fast bitboard check).
            - Hanging pieces (undefended or underdefended).
        Optimized to minimize expensive board.attackers() calls.
        """
        score = 0
        THREAT_WEIGHTS_LIST = [0, 40, 80, 85, 120, 200, 0]  # Indexed by piece_type

        # Precompute pawn attack masks (very cheap bitboard ops)
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

        # Hanging piece detection — only for pieces NOT on pawn-attacked squares
        # (already counted above with full weight)
        # Use a combined occupied mask to iterate efficiently
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
        """
        Penalizes undeveloped minor pieces in the opening/middlegame.

        Pieces on their starting squares in the opening phase get penalized
        to encourage development. This prevents moves like Ng8 (retreating
        a developed knight to its starting square).
        """
        if phase < 14:  # Deep endgame, development doesn't matter
            return 0

        score = 0
        penalty = self.cfg.UNDEVELOPED_PENALTY

        # Scale penalty by game phase (stronger in opening)
        phase_factor = min(phase, 24) / 24.0

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
        """
        Evaluates rook placement on open and semi-open files.

        Rooks are strongest on files with no pawns (open) or only enemy
        pawns (semi-open).
        """
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
        """
        Penalizes pawns that have advanced forward near the friendly king.

        A pawn pushed from g7→g5 or h7→h5 while the king is on g8 weakens
        the kingside shelter and opens lines for the opponent's pieces.
        This penalty scales with how far the pawn has advanced.
        """
        score = 0
        PAWN_ADVANCE_PENALTY = 15  # Per rank of advancement near king

        # --- White King ---
        w_king_sq = board.king(chess.WHITE)
        if w_king_sq is not None:
            king_file = chess.square_file(w_king_sq)
            king_rank = chess.square_rank(w_king_sq)

            # Only care about king on ranks 0-1 (near back rank, i.e. castled)
            if king_rank <= 1:
                for f in range(max(0, king_file - 1), min(8, king_file + 2)):
                    file_mask = FILES[f]
                    pawns_on_file = file_mask & white_pawns
                    if pawns_on_file:
                        for sq in chess.SquareSet(pawns_on_file):
                            pawn_rank = chess.square_rank(sq)
                            # Normal starting position for White pawns is rank 1
                            # Pawn on rank 2 = not advanced, rank 3+ = advanced
                            advance = pawn_rank - 1  # How far from starting rank
                            if advance >= 2:
                                score -= advance * PAWN_ADVANCE_PENALTY

        # --- Black King ---
        b_king_sq = board.king(chess.BLACK)
        if b_king_sq is not None:
            king_file = chess.square_file(b_king_sq)
            king_rank = chess.square_rank(b_king_sq)

            # Only care about king on ranks 6-7 (near back rank, castled)
            if king_rank >= 6:
                for f in range(max(0, king_file - 1), min(8, king_file + 2)):
                    file_mask = FILES[f]
                    pawns_on_file = file_mask & black_pawns
                    if pawns_on_file:
                        for sq in chess.SquareSet(pawns_on_file):
                            pawn_rank = chess.square_rank(sq)
                            # Black pawns start on rank 6, advance downward
                            advance = 6 - pawn_rank
                            if advance >= 2:
                                score += advance * PAWN_ADVANCE_PENALTY

        return score

    def _eval_knight_outposts(
        self, board: chess.Board, white_pawns: int, black_pawns: int
    ) -> int:
        """
        Evaluates knight placement on outpost squares.

        A knight outpost is a square that:
        - Is protected by a friendly pawn
        - Cannot be attacked by an enemy pawn
        - Is in the opponent's half of the board

        Also penalizes knights on the rim (a/h files) that lack support.
        """
        score = 0
        OUTPOST_BONUS = 25
        RIM_UNSUPPORTED_PENALTY = 15

        for sq in board.pieces(chess.KNIGHT, chess.WHITE):
            f = chess.square_file(sq)
            r = chess.square_rank(sq)

            # Outpost check: in opponent's half (rank 4-6)
            if 3 <= r <= 5:
                # Protected by own pawn?
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

                # Can't be attacked by enemy pawn?
                safe = True
                for rr in range(r + 1, 8):  # Start above knight's rank
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

            # Rim penalty: knight on a/h file without pawn support
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

            # Outpost check: in opponent's half (rank 2-4 for Black = rank index 2-4)
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

            # Rim penalty for Black
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
                    score += RIM_UNSUPPORTED_PENALTY  # Penalty for Black = positive for White

        return score

    def _eval_queen_activity(self, board: chess.Board, phase: int) -> int:
        """
        Evaluates queen placement and activity in the middlegame.

        Penalizes queens on the back rank (d1/d8) when pieces are developed.
        Rewards queens in central/active positions.
        """
        if phase < 10:  # Endgame: queen placement matters less
            return 0

        score = 0

        # Queen centrality table (bonus for central squares)
        # Indexed by (file, rank) distance from center
        QUEEN_POSITION_BONUS = [
            # Distance from center (d4/d5/e4/e5):
            # 0 = on center, 1 = adjacent, 2 = far, 3 = edge
            8,
            5,
            2,
            0,
        ]

        BACK_RANK_PENALTY = 15  # Queen sitting on d1/d8 passively

        for sq in board.pieces(chess.QUEEN, chess.WHITE):
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            # Distance from center of board
            file_dist = min(abs(f - 3), abs(f - 4))
            rank_dist = min(abs(r - 3), abs(r - 4))
            center_dist = max(file_dist, rank_dist)
            center_dist = min(center_dist, 3)
            score += QUEEN_POSITION_BONUS[center_dist]

            # Penalty for queen on back rank (not counting early game)
            if r == 0 and phase < 22:  # Not full opening
                score -= BACK_RANK_PENALTY

        for sq in board.pieces(chess.QUEEN, chess.BLACK):
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            file_dist = min(abs(f - 3), abs(f - 4))
            rank_dist = min(abs(r - 3), abs(r - 4))
            center_dist = max(file_dist, rank_dist)
            center_dist = min(center_dist, 3)
            score -= QUEEN_POSITION_BONUS[center_dist]

            # Penalty for queen on back rank
            if r == 7 and phase < 22:
                score += BACK_RANK_PENALTY

        return score
