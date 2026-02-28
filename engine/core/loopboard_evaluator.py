"""Loop-based static evaluator, functionally equivalent to BitboardEvaluator but more readable."""

import chess
from engine.config import CONFIG


class Evaluator:
    def __init__(self):
        self.cfg = CONFIG.eval

    def evaluate(self, board: chess.Board) -> int:
        """Return static eval in centipawns, positive favors side to move."""
        if board.is_insufficient_material():
            return 0
        if board.is_fivefold_repetition():
            return 0
        if board.is_stalemate():
            return 0

        # Phase and score accumulators.
        mg_score = 0
        eg_score = 0
        phase = 0

        # Cache piece data.
        piece_map = board.piece_map()
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)

        for sq, piece in piece_map.items():
            pt = piece.piece_type
            color = piece.color

            # Phase calculation (non-pawn material).
            if pt == chess.KNIGHT:
                phase += 1
            elif pt == chess.BISHOP:
                phase += 1
            elif pt == chess.ROOK:
                phase += 2
            elif pt == chess.QUEEN:
                phase += 4

            # Config values.
            p_name = chess.piece_name(pt).upper()
            mg_val = self.cfg.MATERIAL_MG.get(p_name, 0)
            eg_val = self.cfg.MATERIAL_EG.get(p_name, 0)

            # PST lookup.
            pst_sq = sq if color == chess.WHITE else chess.square_mirror(sq)

            # Safe attribute access.
            table_mg = getattr(self.cfg, f"PST_{p_name}_MG", None)
            table_eg = getattr(self.cfg, f"PST_{p_name}_EG", None)

            pst_mg = table_mg[pst_sq] if table_mg else 0
            pst_eg = table_eg[pst_sq] if table_eg else 0

            # Add to scores
            if color == chess.WHITE:
                mg_score += mg_val + pst_mg
                eg_score += eg_val + pst_eg
            else:
                mg_score -= mg_val + pst_mg
                eg_score -= eg_val + pst_eg

        # Bishop pair.
        if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
            mg_score += self.cfg.BISHOP_PAIR_BONUS
            eg_score += self.cfg.BISHOP_PAIR_BONUS
        if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
            mg_score -= self.cfg.BISHOP_PAIR_BONUS
            eg_score -= self.cfg.BISHOP_PAIR_BONUS

        # Pawn structure.
        w_pawn_score = self._eval_pawns(board, white_pawns, chess.WHITE)
        b_pawn_score = self._eval_pawns(board, black_pawns, chess.BLACK)

        # Distribute: 50% in MG, 100% in EG.
        mg_score += int(w_pawn_score * 0.5)
        mg_score -= int(b_pawn_score * 0.5)
        eg_score += int(w_pawn_score * 1.0)
        eg_score -= int(b_pawn_score * 1.0)

        # Mobility.
        mob_score = self._eval_mobility(board)
        mg_score += mob_score
        eg_score += mob_score

        # King safety (primarily middlegame).
        ks_score = self._eval_king_safety(board)
        mg_score += ks_score
        eg_score += int(ks_score * 0.2)

        # Threats.
        threat_score = self._eval_threats(board, piece_map)
        mg_score += threat_score
        eg_score += threat_score

        # Tapered eval.
        phase = min(phase, 24)
        final_score = (mg_score * phase + eg_score * (24 - phase)) // 24

        if board.turn == chess.BLACK:
            return -final_score

        return final_score

    def _eval_pawns(self, board, pawns, color):
        """Score isolated and passed pawns."""
        score = 0
        for sq in pawns:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)

            # Isolated pawn check.
            is_isolated = True
            if file > 0:
                for r in range(8):
                    p = board.piece_at(chess.square(file - 1, r))
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        is_isolated = False
                        break
            if is_isolated and file < 7:
                for r in range(8):
                    p = board.piece_at(chess.square(file + 1, r))
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        is_isolated = False
                        break

            if is_isolated:
                score += self.cfg.ISOLATED_PAWN_PENALTY

            # Passed pawn check.
            if self._is_passed(board, sq, color):
                relative_rank = rank if color == chess.WHITE else 7 - rank
                if 0 <= relative_rank < 8:
                    score += self.cfg.PASSED_PAWN_BONUS[relative_rank]

        return score

    def _is_passed(self, board, sq, color):
        """True if no enemy pawns block promotion path."""
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        enemy_pawn = chess.Piece(chess.PAWN, not color)

        start_rank = rank + 1 if color == chess.WHITE else 0
        end_rank = 8 if color == chess.WHITE else rank

        # Check current file and adjacent files
        for f in range(max(0, file - 1), min(7, file + 1) + 1):
            for r in range(start_rank, end_rank):
                if board.piece_at(chess.square(f, r)) == enemy_pawn:
                    return False
        return True

    def _eval_mobility(self, board: chess.Board) -> int:
        """Reward pieces for having many attack squares."""
        score = 0
        MOBILITY_WEIGHTS = {
            chess.KNIGHT: 10,
            chess.BISHOP: 10,
            chess.ROOK: 6,
            chess.QUEEN: 3,
        }

        # Non-pawn, non-king pieces.
        for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            # White Mobility
            for sq in board.pieces(pt, chess.WHITE):
                # board.attacks(sq) returns a SquareSet of attacked squares
                mob = len(board.attacks(sq))
                score += mob * MOBILITY_WEIGHTS[pt]

            # Black Mobility
            for sq in board.pieces(pt, chess.BLACK):
                mob = len(board.attacks(sq))
                score -= mob * MOBILITY_WEIGHTS[pt]

        return score

    def _eval_king_safety(self, board: chess.Board) -> int:
        """Penalize lost castling rights, king drift, and missing pawn shield."""
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

            shield_sqs = self._get_shield_squares(w_king_sq, chess.WHITE)
            pawns_in_shield = 0
            for sq in shield_sqs:
                p = board.piece_at(sq)
                if p and p.piece_type == chess.PAWN and p.color == chess.WHITE:
                    pawns_in_shield += 1

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

            shield_sqs = self._get_shield_squares(b_king_sq, chess.BLACK)
            pawns_in_shield = 0
            for sq in shield_sqs:
                p = board.piece_at(sq)
                if p and p.piece_type == chess.PAWN and p.color == chess.BLACK:
                    pawns_in_shield += 1

            if pawns_in_shield < 2:
                score += (2 - pawns_in_shield) * MISSING_SHIELD_PENALTY

        return score

    def _get_shield_squares(self, king_sq, color):
        """Return the three squares directly in front of the king."""
        squares = []
        f, r = chess.square_file(king_sq), chess.square_rank(king_sq)

        # Determine rank in front
        front_r = r + 1 if color == chess.WHITE else r - 1

        if 0 <= front_r <= 7:
            for df in [-1, 0, 1]:
                if 0 <= f + df <= 7:
                    squares.append(chess.square(f + df, front_r))
        return squares

    def _eval_threats(self, board: chess.Board, piece_map: dict) -> int:
        """Detect hanging pieces and pawn-attack threats."""
        score = 0

        # Reduced threat weights to prevent evaluation explosion.
        THREAT_WEIGHTS = {
            chess.PAWN: 10,
            chess.KNIGHT: 30,
            chess.BISHOP: 30,
            chess.ROOK: 50,
            chess.QUEEN: 80,
            chess.KING: 0,
        }

        for sq, piece in piece_map.items():
            if piece.piece_type == chess.KING:
                continue

            attackers = board.attackers(not piece.color, sq)

            if attackers:
                # 1. Attacked by pawn.
                if piece.piece_type > chess.PAWN:
                    attacked_by_pawn = False
                    for atk_sq in attackers:
                        attacker_piece = board.piece_at(atk_sq)
                        if attacker_piece and attacker_piece.piece_type == chess.PAWN:
                            attacked_by_pawn = True
                            break

                    if attacked_by_pawn:
                        penalty = 40
                        score += penalty if piece.color == chess.BLACK else -penalty

                # 2. Hanging piece.
                defenders = board.attackers(piece.color, sq)
                if len(attackers) > len(defenders) or len(defenders) == 0:
                    val = THREAT_WEIGHTS[piece.piece_type]
                    score += val if piece.color == chess.BLACK else -val

        return score
