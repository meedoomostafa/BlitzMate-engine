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

Author: Medo
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
        if board.is_fivefold_repetition():
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

        # Iterate over piece types (excluding pawns)
        phase_weights = {
            chess.KNIGHT: 1,
            chess.BISHOP: 1,
            chess.ROOK: 2,
            chess.QUEEN: 4,
            chess.KING: 0,
        }

        for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            p_name = chess.piece_name(pt).upper()
            mg_val = self.cfg.MATERIAL_MG.get(p_name, 0)
            eg_val = self.cfg.MATERIAL_EG.get(p_name, 0)

            table_mg = getattr(self.cfg, f"PST_{p_name}_MG", None)
            table_eg = getattr(self.cfg, f"PST_{p_name}_EG", None)

            # Evaluate White Pieces
            squares = board.pieces(pt, chess.WHITE)
            count = len(squares)
            phase += count * phase_weights[pt]

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
            phase += count * phase_weights[pt]

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
        mob_score = self._eval_mobility(board)
        mg_score += mob_score
        eg_score += mob_score

        # 4.3 King Safety (Critical in Middle Game)
        ks_score = self._eval_king_safety(board)
        mg_score += ks_score
        eg_score += int(ks_score * 0.2)

        # 4.4 Threat Detection
        threat_score = self._eval_threats(board)
        mg_score += threat_score
        eg_score += threat_score

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

    def _eval_mobility(self, board: chess.Board) -> int:
        """
        Calculates a mobility score based on the number of legal moves (attacks).

        Higher mobility encourages the engine to develop pieces to active squares
        and discourages passive play ("Bunker Mentality").
        """
        score = 0
        MOBILITY_WEIGHTS = {
            chess.KNIGHT: 10,
            chess.BISHOP: 10,
            chess.ROOK: 6,
            chess.QUEEN: 3,
        }

        for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for sq in board.pieces(pt, chess.WHITE):
                mob = len(board.attacks(sq))
                score += mob * MOBILITY_WEIGHTS[pt]

            for sq in board.pieces(pt, chess.BLACK):
                mob = len(board.attacks(sq))
                score -= mob * MOBILITY_WEIGHTS[pt]

        return score

    def _eval_king_safety(self, board: chess.Board) -> int:
        """
        Evaluates the safety of the King.

        Penalizes:
            - Loss of castling rights manually (stuck in center).
            - King wandering away from safety zones (G1/C1) in the middle game.
            - Missing pawn shield (open files in front of king).
        """
        score = 0
        MISSING_SHIELD_PENALTY = 20
        LOST_CASTLING_PENALTY = 40
        KING_DRIFT_PENALTY = 15

        w_king_sq = board.king(chess.WHITE)
        if w_king_sq is not None:
            if w_king_sq == chess.E1:
                can_castle = (board.castling_rights & chess.BB_H1) or (
                    board.castling_rights & chess.BB_A1
                )
                if not can_castle:
                    score -= LOST_CASTLING_PENALTY

            if w_king_sq not in [chess.E1, chess.G1, chess.C1]:
                if board.pieces(chess.QUEEN, chess.BLACK):
                    score -= KING_DRIFT_PENALTY

            shield_mask = self.king_shield_masks[w_king_sq] if w_king_sq < 56 else 0
            if shield_mask:
                pawns_in_shield = (
                    shield_mask & board.pieces_mask(chess.PAWN, chess.WHITE)
                ).bit_count()
                if pawns_in_shield < 2:
                    score -= (2 - pawns_in_shield) * MISSING_SHIELD_PENALTY

        b_king_sq = board.king(chess.BLACK)
        if b_king_sq is not None:
            if b_king_sq == chess.E8:
                can_castle = (board.castling_rights & chess.BB_H8) or (
                    board.castling_rights & chess.BB_A8
                )
                if not can_castle:
                    score += LOST_CASTLING_PENALTY

            if b_king_sq not in [chess.E8, chess.G8, chess.C8]:
                if board.pieces(chess.QUEEN, chess.WHITE):
                    score += KING_DRIFT_PENALTY

            f, r = chess.square_file(b_king_sq), chess.square_rank(b_king_sq)
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
            if missing > 1:
                score += (missing - 1) * MISSING_SHIELD_PENALTY

        return score

    def _eval_threats(self, board: chess.Board) -> int:
        """
        Evaluates immediate tactical threats.

        Detects:
            - Hanging pieces (undefended or attacked by lower value piece).
            - Valuable pieces attacked by pawns.
        """
        score = 0
        THREAT_WEIGHTS = {
            chess.PAWN: 10,
            chess.KNIGHT: 30,
            chess.BISHOP: 30,
            chess.ROOK: 50,
            chess.QUEEN: 80,
            chess.KING: 0,
        }

        piece_map = board.piece_map()

        for sq, piece in piece_map.items():
            if piece.piece_type == chess.KING:
                continue

            attackers = board.attackers(not piece.color, sq)
            if not attackers:
                continue

            # 1. Attacked by a Pawn? (High danger)
            if piece.piece_type > chess.PAWN:
                attacked_by_pawn = False
                for atk_sq in attackers:
                    if board.piece_at(atk_sq).piece_type == chess.PAWN:
                        attacked_by_pawn = True
                        break

                if attacked_by_pawn:
                    penalty = 40
                    score += penalty if piece.color == chess.BLACK else -penalty

            # 2. Hanging Piece Logic
            defenders = board.attackers(piece.color, sq)
            if len(attackers) > len(defenders) or len(defenders) == 0:
                penalty = THREAT_WEIGHTS[piece.piece_type]
                score += penalty if piece.color == chess.BLACK else -penalty

        return score
