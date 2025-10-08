import chess
from engine.core.evaluator import Evaluator
from collections import defaultdict
from typing import Optional, Tuple, Callable
import threading
from engine.core.transposition import TranspositionTable

INF = 10 ** 9

# Transposition table flags
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2


class SearchEngine:
    """
    Negamax search engine with alpha-beta, quiescence, iterative deepening
    and a transposition table. Designed to be usable from a GUI by running
    searches in a background thread (start_search) while keeping a blocking
    synchronous API (search_best_move) if you prefer.

    Important notes for GUI integration:
    - The engine never mutates the Board object you pass in; it works on a
      local copy. That makes it safe to call from another thread while the
      GUI thread keeps using the displayed board.
    - Prefer using start_search(board, depth, callback) from the GUI and
      perform any board.push(...) calls on the main thread (callback runs in
      the engine thread). To apply the engine move safely in the GUI, post a
      pygame event or schedule the push to run on the main thread.
    """

    def __init__(self, evaluator: Optional[Evaluator] = None, depth: int = 3):
        self.evaluator = evaluator or Evaluator()
        self.max_depth = depth
        # history heuristic
        self.history = defaultdict(int)
        # transposition table: fen -> (entry_depth, value, flag, best_move)
        self.tt = TranspositionTable()  
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._result_move: Optional[chess.Move] = None

    # -------------------------
    # Public APIs
    # -------------------------
    def search_best_move(self, board: chess.Board) -> Optional[chess.Move]:
            """
            Blocking full search (no time limit). Returns a chess.Move or None.
            Performs iterative deepening up to self.max_depth.
            """
            # perform the search on a copy so we don't mutate caller's board
            board_copy = board.copy()
            self._stop_event.clear()
            self._result_move = None

            best_move = None
            for depth in range(1, self.max_depth + 1):
                # synchronous call to root search
                _ = self._negamax_root(board_copy, depth, -INF, INF)
                with self._lock:
                    entry = self.tt.get(board_copy)
                    if entry is not None:
                        # Access attributes directly (best practice)
                        if isinstance(entry.best_move, chess.Move):
                            best_move = entry.best_move
                # continue till max_depth (no time limit)
                if self._stop_event.is_set():
                    break

            self._result_move = best_move
            return best_move

    def start_search(self, board: chess.Board, depth: Optional[int] = None,
                     callback: Optional[Callable[[Optional[chess.Move], int], None]] = None) -> None:
        """
        Starts the search in a background thread. The callback, if provided,
        will be called after each completed iterative-deepening depth with
        arguments (best_move, depth_value). IMPORTANT: the callback is invoked
        in the engine thread — do NOT mutate GUI objects there. Instead post a
        message/event to the GUI thread.
        """
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Search already running")

        self._stop_event.clear()
        self._result_move = None
        target_depth = depth or self.max_depth

        def worker():
            board_copy = board.copy()
            best_move_local = None
            for d in range(1, target_depth + 1):
                if self._stop_event.is_set():
                    break
                _ = self._negamax_root(board_copy, d, -INF, INF)
                with self._lock:
                    entry = self.tt.get(board_copy)
                    if entry is not None:
                        _, _, _, tt_move = entry
                        if isinstance(tt_move, chess.Move):
                            best_move_local = tt_move
                # call callback with best move so far and depth
                if callback:
                    try:
                        callback(best_move_local, d)
                    except Exception:
                        # callback must not crash engine; ignore exceptions
                        pass
                if self._stop_event.is_set():
                    break

            # finished
            self._result_move = best_move_local
            if callback:
                try:
                    callback(best_move_local, -1)  # -1 indicates final
                except Exception:
                    pass

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Request the running search to stop (cooperative)."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.1)

    def is_searching(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_result(self) -> Optional[chess.Move]:
        return self._result_move

    # -------------------------
    # Root helper
    # -------------------------
    def _negamax_root(self, board: chess.Board, depth: int, alpha: int, beta: int) -> int:
        # operate on the board object passed (it should be a copy created by caller)
        color = 1 if board.turn == chess.WHITE else -1
        max_eval = -INF

        moves = list(board.legal_moves)

        # try to use TT best move first
        with self._lock:
            entry = self.tt.get(board)
            tt_move = entry.best_move if entry is not None else None

        if tt_move and tt_move in moves:
            moves.remove(tt_move)
            moves.insert(0, tt_move)

        moves = self._order_moves(board, moves)

        for move in moves:
            if self._stop_event.is_set():
                break
            board.push(move)
            val = -self._negamax(board, depth - 1, -beta, -alpha, -color)
            board.pop()

            if val > max_eval:
                max_eval = val
                alpha = max(alpha, val)
                with self._lock:
                    self.tt.store(board, depth, max_eval, EXACT, move)

            if alpha >= beta:
                break

        return max_eval

    # -------------------------
    # Negamax with alpha-beta and TT
    # -------------------------
    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, color: int) -> int:
        if depth <= 0 or board.is_game_over() or self._stop_event.is_set():
            return self._quiescence(board, alpha, beta, color)

        with self._lock:
            entry = self.tt.get(board)
        original_alpha = alpha

        if entry is not None:
            if entry.depth >= depth:
                if entry.flag == EXACT:
                    return entry.value
                elif entry.flag == LOWERBOUND:
                    alpha = max(alpha, entry.value)
                elif entry.flag == UPPERBOUND:
                    beta = min(beta, entry.value)
                if alpha >= beta:
                    return entry.value

        max_eval = -INF
        moves = list(board.legal_moves)

        # use TT move as first if present
        if entry is not None and isinstance(entry.best_move, chess.Move):
            try:
                moves.remove(entry.best_move)
                moves.insert(0, entry.best_move)
            except ValueError:
                pass

        moves = self._order_moves(board, moves)

        for move in moves:
            if self._stop_event.is_set():
                break
            board.push(move)
            val = -self._negamax(board, depth - 1, -beta, -alpha, -color)
            board.pop()

            if val > max_eval:
                max_eval = val

            alpha = max(alpha, val)

            if val > original_alpha and not board.is_capture(move):
                # history heuristic update
                self.history[(move.from_square, move.to_square)] += 2 ** depth

            if alpha >= beta:
                with self._lock:
                    self.tt.store(board, depth, alpha, LOWERBOUND, move)
                return alpha

        # store result in TT
        flag = EXACT
        if max_eval <= original_alpha:
            flag = UPPERBOUND
        elif max_eval >= beta:
            flag = LOWERBOUND

        with self._lock:
            self.tt.store(board, depth, max_eval, flag, moves[0] if moves else None)

        return max_eval


    # -------------------------
    # Quiescence search (captures only)
    # -------------------------
    def _quiescence(self, board: chess.Board, alpha: int, beta: int, color: int) -> int:
        if self._stop_event.is_set():
            return 0

        stand_pat = color * self.evaluator.evaluate_board(board)
        if stand_pat >= beta:
            return stand_pat
        if alpha < stand_pat:
            alpha = stand_pat

        captures = [m for m in board.legal_moves if board.is_capture(m)]
        captures.sort(key=lambda m: self._capture_score(board, m), reverse=True)

        for move in captures:
            if self._stop_event.is_set():
                break
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
        def key(move):
            if board.is_capture(move):
                return (2, self._capture_score(board, move))
            h = self.history.get((move.from_square, move.to_square), 0)
            return (1, h)

        return sorted(moves, key=key, reverse=True)

    def _capture_score(self, board: chess.Board, move):
        from engine.config import PIECE_VALUES
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        vic_val = 0 if victim is None else PIECE_VALUES.get(victim.piece_type, 0)
        att_val = 0 if attacker is None else PIECE_VALUES.get(attacker.piece_type, 0)
        return (vic_val - att_val)
