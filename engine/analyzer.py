"""Move classification and full-game analysis."""

import chess
from typing import List, Dict, Any

# Classification thresholds (centipawns, delta vs engine best).
TH_BRILLIANT = -50
TH_BEST = 50
TH_GOOD = 150
TH_INACCURACY = 300
TH_MISTAKE = 600


class Analyzer:
    def __init__(self, search_engine):
        self.search_engine = search_engine

    def _from_player_pov(self, value: int, player_is_white: bool) -> int:
        """Convert engine eval (positive = White) to the moving player's perspective."""
        return value if player_is_white else -value

    def classify_move(
        self,
        board: chess.Board,
        move: chess.Move,
        best_move: chess.Move,
        best_score: int,
    ) -> Dict[str, Any]:
        """Classify a single move against the engine's best. Board is not mutated."""
        player_is_white = board.turn == chess.WHITE

        # Baseline eval before move.
        old_eval = self.search_engine.evaluator.evaluate(board)

        # Eval after the player's move.
        board.push(move)
        new_eval = self.search_engine.evaluator.evaluate(board)
        is_checkmate = board.is_checkmate()
        board.pop()

        # Use search-reported best_score; fall back to static eval of best_move.
        best_eval = best_score
        if best_move is not None:
            board.push(best_move)
            best_eval_alt = self.search_engine.evaluator.evaluate(board)
            board.pop()
            # Prefer deeper search score when available.
            best_eval = best_score if best_score is not None else best_eval_alt

        # Convert to moving player's POV.
        old_pov = self._from_player_pov(old_eval, player_is_white)
        new_pov = self._from_player_pov(new_eval, player_is_white)
        best_pov = self._from_player_pov(best_eval, player_is_white)

        # Positive delta means best move was superior to the chosen one.
        delta_vs_best = best_pov - new_pov
        improvement_vs_old = new_pov - old_pov

        # Classify the move.
        if is_checkmate:
            label = "Checkmate (winning move)"
        elif best_move is not None and move == best_move:
            # Player matched engine's best move.
            if improvement_vs_old > TH_BEST:
                label = "Brilliant (best)"
            else:
                label = "Best move"
        else:
            # Threshold-based classification.
            d = delta_vs_best
            if d <= TH_BRILLIANT:
                label = "Brilliant"
            elif d <= TH_BEST:
                label = "Excellent"
            elif d <= TH_GOOD:
                label = "Good"
            elif d <= TH_INACCURACY:
                label = "Inaccuracy"
            elif d <= TH_MISTAKE:
                label = "Mistake"
            else:
                label = "Blunder"

        info = {
            "move_uci": move.uci(),
            "old_eval": old_eval,
            "new_eval": new_eval,
            "best_move": best_move.uci() if best_move else None,
            "best_score": best_score,
            "delta_vs_best": int(delta_vs_best),
            "improvement_vs_old": int(improvement_vs_old),
            "label": label,
            "player_is_white": player_is_white,
        }
        return info

    def analyze_game(self, moves: List[str]) -> List[Dict[str, Any]]:
        """Analyze a full game from UCI move strings. Returns a per-move report."""
        board = chess.Board()
        report = []
        for mv_uci in moves:
            # Engine's best at the current position.
            best_move, _ponder, best_score = self.search_engine.search_best_move(board)
            # Parse and validate the player's move.
            move = chess.Move.from_uci(mv_uci)
            if move not in board.legal_moves:
                report.append({"move_uci": mv_uci, "label": "Illegal", "error": True})
                break
            # Classify and advance.
            info = self.classify_move(board, move, best_move, best_score)
            report.append(info)
            board.push(move)
        return report
