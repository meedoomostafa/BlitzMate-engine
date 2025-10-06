# core/search.py
import chess
from core.evaluator import Evaluator
from collections import defaultdict

INF = 10**9

class SearchEngine:
    def __init__(self, evaluator=None, depth=3):
        """
        evaluator.evaluate_board(board) must return an int (centipawns),
        positive if White is better, negative if Black is better.
        depth = search depth in plies (half-moves).
        """
        self.evaluator = evaluator or Evaluator()
        self.depth = depth
        # simple history heuristic table (sq_from, sq_to) -> score
        self.history = defaultdict(int)

    # Public API
    def search_best_move(self, board: chess.Board):
        """
        Returns (best_move, best_score_in_centipawns)
        """
        color = 1 if board.turn == chess.WHITE else -1
        best_move = None
        alpha = -INF
        beta = INF

        # order root moves (captures first then history)
        moves = list(board.legal_moves)
        moves = self._order_moves(board, moves)

        for move in moves:
            board.push(move)
            score = -self._negamax(board, self.depth - 1, -beta, -alpha, -color)
            board.pop()
            if score > alpha:
                alpha = score
                best_move = move
            # small optimization: if we found mate or very high score, early stop
            if alpha >= INF - 1000:
                break

        return best_move, alpha

    # -------------------------
    # Core negamax (alpha-beta)
    # -------------------------
    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, color: int) -> int:
        """
        color: +1 if the side to move is treated as maximizing for White, -1 for Black.
        We always return centipawns from the viewpoint of the side to move multiplied
        by color convention in the caller (so root uses color=1 for White to move).
        """
        if depth <= 0 or board.is_game_over():
            # at leaf, use quiescence to extend captures
            return self._quiescence(board, alpha, beta, color)

        max_eval = -INF

        moves = list(board.legal_moves)
        moves = self._order_moves(board, moves)

        for move in moves:
            board.push(move)
            val = -self._negamax(board, depth - 1, -beta, -alpha, -color)
            board.pop()

            if val > max_eval:
                max_eval = val
            if val > alpha:
                alpha = val
                # history heuristic update for non-captures
                if not board.is_capture(move):
                    self.history[(move.from_square, move.to_square)] += 2 ** depth
            if alpha >= beta:
                # Beta cutoff
                # update killer/history heuristics could be done here
                return alpha

        return max_eval

    # -------------------------
    # Quiescence search (captures only)
    # -------------------------
    def _quiescence(self, board: chess.Board, alpha: int, beta: int, color: int) -> int:
        """
        Stand pat evaluation first, then search capture moves to avoid horizon effect.
        """
        stand_pat = color * self.evaluator.evaluate_board(board)
        if stand_pat >= beta:
            return stand_pat
        if alpha < stand_pat:
            alpha = stand_pat

        # consider captures only
        captures = [m for m in board.legal_moves if board.is_capture(m)]
        # order captures by MVV-LVA approximate (victim value - attacker value)
        captures.sort(key=lambda m: self._capture_score(board, m), reverse=True)

        for move in captures:
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, -color)
            board.pop()

            if score >= beta:
                return score
            if score > alpha:
                alpha = score

        return alpha

    # -------------------------
    # Move ordering helpers
    # -------------------------
    def _order_moves(self, board: chess.Board, moves):
        """
        Return moves ordered: captures by MVV-LVA, then history heuristic.
        """
        # build a key for each move
        def key(move):
            if board.is_capture(move):
                return (2, self._capture_score(board, move))
            # history heuristic score
            h = self.history.get((move.from_square, move.to_square), 0)
            return (1, h)
        # sort descending
        return sorted(moves, key=key, reverse=True)

    def _capture_score(self, board: chess.Board, move):
        """
        Rough MVV-LVA: victim_value - attacker_value (centipawns)
        Larger is better.
        """
        from engine.config import PIECE_VALUES
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        vic_val = 0 if victim is None else PIECE_VALUES.get(victim.piece_type, 0)
        att_val = 0 if attacker is None else PIECE_VALUES.get(attacker.piece_type, 0)
        return (vic_val - att_val)

