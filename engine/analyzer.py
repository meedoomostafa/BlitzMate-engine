# engine/analyzer.py
import chess
from typing import List, Dict, Any

# thresholds in centipawns (integers). Tune these later.
TH_BRILLIANT = -50    # if your move is better than best by >= 50cp => "Brilliant"
TH_BEST = 50          # if within 50cp of best => "Best/Excellent"
TH_GOOD = 150         # within 150cp => "Good"
TH_INACCURACY = 300   # 150..300 => "Inaccuracy"
TH_MISTAKE = 600      # 300..600 => "Mistake"
# > TH_MISTAKE => "Blunder"

class Analyzer:
    def __init__(self, search_engine):
        self.search_engine = search_engine

    def _from_player_pov(self, value: int, player_is_white: bool) -> int:
        """
        Convert a raw evaluation (centipawns, positive = White better)
        into a value from the moving player's POV:
          - For White to move: pov = value
          - For Black to move: pov = -value
        """
        return value if player_is_white else -value

    def classify_move(self, board: chess.Board, move: chess.Move, best_move: chess.Move, best_score: int) -> Dict[str, Any]:
        """
        Classify a single move. 
        - board: current board BEFORE the player's move (unchanged by this function).
        - move: the player's chosen move (chess.Move object).
        - best_move: engine's best move at this position (chess.Move or None).
        - best_score: engine's reported score for best_move in centipawns (positive=White better).
        Returns a dict with label, deltas and diagnostics.
        """
        player_is_white = (board.turn == chess.WHITE)

        # baseline eval before the move
        old_eval = self.search_engine.evaluator.evaluate_board(board)  # centipawns

        # get evaluation after the player's move
        board.push(move)
        new_eval = self.search_engine.evaluator.evaluate_board(board)
        is_checkmate = board.is_checkmate()
        board.pop()

        # get engine's best eval: prefer search-provided best_score,
        # but also try to evaluate the resulting position after best_move for accuracy
        best_eval = best_score
        if best_move is not None:
            board.push(best_move)
            best_eval_alt = self.search_engine.evaluator.evaluate_board(board)
            board.pop()
            # choose the search-reported best_score if it exists, otherwise fallback to evaluator after best_move
            # sometimes search returns deeper estimate; keep best_score (expected centipawns)
            best_eval = best_score if best_score is not None else best_eval_alt

        # convert to moving player's POV
        old_pov = self._from_player_pov(old_eval, player_is_white)
        new_pov = self._from_player_pov(new_eval, player_is_white)
        best_pov = self._from_player_pov(best_eval, player_is_white)

        # delta: how much worse (positive means best is better than chosen move)
        delta_vs_best = best_pov - new_pov
        improvement_vs_old = new_pov - old_pov

        # SEE (static exchange evaluation) to detect bad captures
        see_value = None
        try:
            see_value = board.see(move)  # returns material gain for side to move (centipawns-ish)
        except Exception:
            see_value = None

        # special cases
        if is_checkmate:
            label = "Checkmate (winning move)"
        elif best_move is not None and move == best_move:
            # If the player's move equals engine best move
            # check if it's strictly better than previous position
            if improvement_vs_old > TH_BEST:
                label = "Brilliant (best)"
            else:
                label = "Best move"
        else:
            # classify using delta_vs_best thresholds
            d = delta_vs_best
            # If player's move is actually better than engine best (rare), call it brilliant
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
            "see": int(see_value) if see_value is not None else None,
            "label": label,
            "player_is_white": player_is_white
        }
        return info

    def analyze_game(self, moves: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze a list of moves (UCI strings). Returns a detailed report list.
        """
        board = chess.Board()
        report = []
        for mv_uci in moves:
            # compute engine best move at current position
            best_move, best_score = self.search_engine.search_best_move(board)
            # parse the player's move
            move = chess.Move.from_uci(mv_uci)
            # classify
            info = self.classify_move(board, move, best_move, best_score)
            report.append(info)
            # play the player's move on the board
            board.push(move)
        return report
