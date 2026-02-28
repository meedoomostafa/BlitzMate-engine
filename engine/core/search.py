"""Negamax search engine with iterative deepening, LMR, null-move pruning, and quiescence."""

import chess
from chess import polyglot
from chess import syzygy
import math
import time
import threading
import os
from collections import defaultdict
from typing import Optional, Callable, List, Tuple

from engine.core.bitboard_evaluator import BitboardEvaluator as Evaluator
from engine.core.transposition import TranspositionTable, TT_EXACT, TT_ALPHA, TT_BETA
from engine.core.utils import print_info

INF = 1000000
MATE_SCORE = 900000
TB_WIN_SCORE = 800000

# SEE piece values indexed by chess.piece_type (PAWN=1 .. KING=6).
SEE_PIECE_VALUES = [0, 100, 320, 330, 500, 900, 20000]

# Precomputed LMR reduction table: LMR_TABLE[depth][move_index].
MAX_LMR_DEPTH = 64
MAX_LMR_MOVES = 64

LMR_TABLE = [[0] * MAX_LMR_MOVES for _ in range(MAX_LMR_DEPTH)]
for _d in range(1, MAX_LMR_DEPTH):
    for _m in range(1, MAX_LMR_MOVES):
        LMR_TABLE[_d][_m] = max(0, int(0.5 + math.log(_d) * math.log(_m) / 2.0))


class SearchEngine:
    def __init__(self, evaluator: Optional[Evaluator] = None, depth: int = 6):
        self.evaluator = evaluator or Evaluator()
        self.max_depth = depth
        self.tt = TranspositionTable()
        self.history = defaultdict(lambda: defaultdict(int))

        self.killers = defaultdict(lambda: [None, None])

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.nodes = 0

        base_dir = os.path.dirname(os.path.abspath(__file__))
        books_path = os.path.join(base_dir, "../assets/openings")
        self.book_paths = [
            os.path.join(books_path, "Titans.bin"),
            os.path.join(books_path, "gm2600.bin"),
            os.path.join(books_path, "komodo.bin"),
            os.path.join(books_path, "rodent.bin"),
            os.path.join(books_path, "Human.bin"),
        ]

        syzygy_path = os.path.join(base_dir, "../assets/syzygy")
        self.tablebase = None

        if os.path.exists(syzygy_path):
            try:
                self.tablebase = chess.syzygy.open_tablebase(syzygy_path)
            except Exception:
                pass

    def get_book_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Probe opening books for a weighted random move. Returns None on miss."""
        for book_path in self.book_paths:
            if os.path.exists(book_path):
                try:
                    with polyglot.open_reader(book_path) as reader:
                        entry = reader.weighted_choice(board)
                        print(
                            f"[{os.path.basename(book_path)}] Book Move: {entry.move.uci()}"
                        )
                        return entry.move
                except Exception:
                    continue
        return None

    def get_syzygy_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Probe Syzygy tablebases (<=5 pieces, no castling). Returns best DTZ move or None."""
        if (
            not self.tablebase
            or chess.popcount(board.occupied) > 5
            or board.castling_rights != 0
        ):
            return None

        try:
            root_wdl = self.tablebase.probe_wdl(board)
        except Exception:
            return None

        candidates = []

        for move in board.legal_moves:
            board.push(move)
            try:
                outcome = self.tablebase.probe_wdl(board)

                is_valid_transition = False
                if root_wdl > 0:
                    is_valid_transition = outcome < 0
                elif root_wdl < 0:
                    is_valid_transition = (
                        True  # Already losing; maximize DTZ to delay.
                    )
                else:
                    is_valid_transition = outcome == 0

                if is_valid_transition:
                    dtz = self.tablebase.probe_dtz(board)
                    candidates.append((move, dtz))

            except Exception:
                pass

            finally:
                board.pop()

        if not candidates:
            return None

        if root_wdl > 0:  # Winning: fastest conversion.
            candidates.sort(key=lambda x: x[1], reverse=True)
        elif root_wdl < 0:  # Losing: slowest loss.
            candidates.sort(key=lambda x: x[1], reverse=True)
        else:  # Drawn: stay near zero.
            candidates.sort(key=lambda x: abs(x[1]))

        best_move = candidates[0][0]
        print(
            f"[Syzygy] WDL: {root_wdl}, Best Move: {best_move}, DTZ(opp): {candidates[0][1]}"
        )
        return best_move

    def search_best_move(
        self, board: chess.Board
    ) -> Tuple[Optional[chess.Move], Optional[chess.Move], int]:
        self._stop_event.clear()

        book_move = self.get_book_move(board)
        if book_move:
            return book_move, None, 0

        tb_move = self.get_syzygy_move(board)
        if tb_move:
            return tb_move, None, 0

        self.nodes = 0
        search_board = board.copy()
        best_move = None
        ponder_move = None
        best_score = 0

        # Age history scores to prevent stale move ordering.
        for from_sq in self.history:
            for to_sq in self.history[from_sq]:
                self.history[from_sq][to_sq] //= 2

        self.killers = defaultdict(lambda: [None, None])

        start_time = time.time()

        # Iterative deepening with aspiration windows.
        ASPIRATION_WINDOW = 50
        prev_score = 0

        for d in range(1, self.max_depth + 1):
            if self._stop_event.is_set():
                break

            # Aspiration windows from depth 4+.
            if d >= 4:
                asp_alpha = prev_score - ASPIRATION_WINDOW
                asp_beta = prev_score + ASPIRATION_WINDOW
                score = self._negamax(search_board, d, asp_alpha, asp_beta, 0)

                # Re-search with full window on fail.
                if not self._stop_event.is_set() and (
                    score <= asp_alpha or score >= asp_beta
                ):
                    score = self._negamax(search_board, d, -INF, INF, 0)
            else:
                score = self._negamax(search_board, d, -INF, INF, 0)
            if self._stop_event.is_set():
                break
            best_score = score
            prev_score = score

            # Retrieve best move from TT.
            entry = self.tt.get(search_board)
            if entry and entry.best_move:
                best_move = entry.best_move

            pv_moves = self._get_pv_line(search_board, d)
            if len(pv_moves) > 0:
                best_move = pv_moves[0]
            if len(pv_moves) > 1:
                ponder_move = pv_moves[1]

            elapsed = time.time() - start_time
            print_info(d, score, self.nodes, elapsed, pv_moves, ponder_move, MATE_SCORE)

        return best_move, ponder_move, best_score

    def start_search(
        self,
        board: chess.Board,
        depth: Optional[int] = None,
        callback: Optional[Callable] = None,
    ):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        target_depth = depth or self.max_depth

        def worker():
            book_move = self.get_book_move(board)
            if book_move:
                if callback:
                    callback(book_move, None, 0, 0)
                return

            tb_move = self.get_syzygy_move(board)
            if tb_move:
                if callback:
                    callback(tb_move, None, 0, 0)
                return

            search_board = board.copy()
            best_move = None
            ponder_move = None
            self.nodes = 0
            start_time = time.time()

            for d in range(1, target_depth + 1):
                if self._stop_event.is_set():
                    break
                score = self._negamax(search_board, d, -INF, INF, 0)

                entry = self.tt.get(search_board)
                if entry:
                    best_move = entry.best_move

                # Print to stdout for debugging.
                pv_moves = self._get_pv_line(search_board, d)
                if len(pv_moves) > 0:
                    best_move = pv_moves[0]
                if len(pv_moves) > 1:
                    ponder_move = pv_moves[1]

                elapsed = time.time() - start_time
                print_info(
                    d, score, self.nodes, elapsed, pv_moves, ponder_move, MATE_SCORE
                )

                if callback:
                    callback(best_move, ponder_move, d, score)

            if callback:
                callback(best_move, ponder_move, -1, 0)

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.2)

    def _get_pv_line(self, board: chess.Board, depth: int) -> List[chess.Move]:
        """Extract the principal variation from the TT."""
        pv_moves = []
        curr_board = board.copy()
        seen_hashes = set()  # Cycle detection.

        for _ in range(depth):
            entry = self.tt.get(curr_board)
            if not entry or not entry.best_move:
                break

            move = entry.best_move
            if move not in curr_board.legal_moves:
                break

            pv_moves.append(move)
            curr_board.push(move)

            # Cycle detection via Zobrist hash.
            h = polyglot.zobrist_hash(curr_board)
            if h in seen_hashes:
                break
            seen_hashes.add(h)

        return pv_moves

    def _negamax(
        self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int
    ) -> int:
        self.nodes += 1
        if board.is_repetition(2) or board.halfmove_clock >= 100:
            return 0
        if self.nodes % 2048 == 0 and self._stop_event.is_set():
            return 0

        if self.tablebase and board.castling_rights == 0:
            if chess.popcount(board.occupied) <= 5:
                try:
                    wdl = self.tablebase.probe_wdl(board)
                    if wdl > 0:
                        return 800000 - ply
                    if wdl < 0:
                        return -800000 + ply
                    return 0
                except Exception:
                    pass

        # Check extension: extend depth when in check.
        in_check = board.is_check()
        if in_check:
            depth += 1

        alpha_orig = alpha

        # TT Lookup
        tt_entry = self.tt.get(board)
        tt_move = None

        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == TT_EXACT:
                return tt_entry.value
            elif tt_entry.flag == TT_ALPHA:
                beta = min(beta, tt_entry.value)
            elif tt_entry.flag == TT_BETA:
                alpha = max(alpha, tt_entry.value)
            if alpha >= beta:
                return tt_entry.value

        if tt_entry:
            tt_move = tt_entry.best_move

        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply=ply)

        if board.is_game_over():
            if board.is_checkmate():
                return -MATE_SCORE + ply
            return 0

        is_pv_node = beta - alpha > 1

        big_piece_count = sum(
            len(board.pieces(pt, board.turn))
            for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        )

        static_eval = self.evaluator.evaluate(board)

        # Reverse futility pruning.
        if (
            depth <= 3
            and not in_check
            and not is_pv_node
            and abs(alpha) < MATE_SCORE - 100
            and static_eval - 120 * depth >= beta
        ):
            return static_eval

        # Null-move pruning with adaptive reduction.
        if (
            depth >= 3
            and not in_check
            and ply > 0
            and big_piece_count >= 2
            and static_eval >= beta
        ):
            # Adaptive R = 3 + depth/6, capped at depth-1.
            R = 3 + depth // 6
            R = min(R, depth - 1)
            board.push(chess.Move.null())
            score = -self._negamax(board, depth - R, -beta, -beta + 1, ply + 1)
            board.pop()
            if score >= beta:
                # Verification search at shallow depths.
                if depth <= 6:
                    # Verify with reduced depth to avoid null-move re-trigger.
                    v_depth = max(1, depth - R - 1)
                    v_score = self._negamax(board, v_depth, alpha, beta, ply)
                    if v_score >= beta:
                        return beta
                else:
                    return beta

        # Futility pruning margins.
        futility_margin = [0, 200, 350, 500]
        can_futility_prune = (
            depth <= 3
            and not in_check
            and not is_pv_node
            and abs(alpha) < MATE_SCORE - 100
            and static_eval + futility_margin[depth] <= alpha
        )

        moves = self._order_moves(board, tt_move, ply)
        best_score = -INF
        best_move_found = None

        moves_searched = 0

        for move in moves:
            is_cap = board.is_capture(move)
            # Lazy gives_check: only compute when actually needed (saves ~11% time)
            gives_check = None

            # Futility pruning: skip quiet moves unlikely to raise alpha.
            if (
                can_futility_prune
                and moves_searched > 0
                and not is_cap
                and not move.promotion
            ):
                if gives_check is None:
                    gives_check = board.gives_check(move)
                if not gives_check:
                    continue

            # Compute gives_check before push (board.gives_check requires move is legal)
            if gives_check is None and depth >= 3 and not in_check:
                gives_check = board.gives_check(move)

            board.push(move)
            moves_searched += 1
            needs_full_search = True
            is_killer = move == self.killers[ply][0] or move == self.killers[ply][1]

            # Late move reductions (table-based).
            if (
                depth >= 3
                and moves_searched > 3
                and not is_cap
                and not move.promotion
                and not is_killer
                and not in_check
            ):
                if gives_check:
                    pass  # Never reduce checking moves.
                else:
                    # Use precomputed LMR table
                    r = LMR_TABLE[min(depth, MAX_LMR_DEPTH - 1)][
                        min(moves_searched, MAX_LMR_MOVES - 1)
                    ]
                    # Reduce less in PV nodes.
                    if is_pv_node:
                        r = max(0, r - 1)
                    # Reduce less for high-history moves.
                    hist_score = self.history[move.from_square][move.to_square]
                    if hist_score > 1000:
                        r = max(0, r - 1)

                    r = max(1, r)  # At least reduce by 1
                    new_depth = max(0, depth - 1 - r)

                    score = -self._negamax(
                        board, new_depth, -alpha - 1, -alpha, ply + 1
                    )
                    needs_full_search = score > alpha
            elif not is_pv_node and moves_searched > 1:
                # PVS: null-window search for non-PV moves.
                score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                needs_full_search = score > alpha

            if needs_full_search:
                score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)

            board.pop()

            if self._stop_event.is_set():
                return 0

            if score > best_score:
                best_score = score
                best_move_found = move

            if score > alpha:
                alpha = score
                if not is_cap:
                    self.history[move.from_square][move.to_square] += depth * depth
                    if move != self.killers[ply][0]:
                        self.killers[ply][1] = self.killers[ply][0]
                        self.killers[ply][0] = move

                if alpha >= beta:
                    if not self._stop_event.is_set():
                        self.tt.store(board, depth, best_score, TT_BETA, move)
                    return best_score

        if moves_searched == 0:
            if in_check:
                return -MATE_SCORE + ply
            return 0

        if self._stop_event.is_set():
            return best_score

        if best_score <= alpha_orig:
            flag = TT_ALPHA
        elif best_score >= beta:
            flag = TT_BETA
        else:
            flag = TT_EXACT

        if not self._stop_event.is_set():
            self.tt.store(board, depth, best_score, flag, best_move_found)
        return best_score

    def _quiescence(
        self, board: chess.Board, alpha: int, beta: int, qs_depth: int = 0, ply: int = 0
    ) -> int:
        """Quiescence search: resolve captures and checks before returning static eval."""
        self.nodes += 1
        if self._stop_event.is_set():
            return 0
        if qs_depth > 30:
            return self.evaluator.evaluate(board)

        # TT probe in quiescence
        tt_entry = self.tt.get(board)
        if tt_entry and tt_entry.depth >= 0:
            if tt_entry.flag == TT_EXACT:
                return tt_entry.value
            elif tt_entry.flag == TT_ALPHA:
                beta = min(beta, tt_entry.value)
            elif tt_entry.flag == TT_BETA:
                alpha = max(alpha, tt_entry.value)
            if alpha >= beta:
                return tt_entry.value

        alpha_orig = alpha
        in_check = board.is_check()

        if in_check:
            # Must search all evasions when in check.
            stand_pat = -INF
            moves = list(board.legal_moves)
            if not moves:
                return -MATE_SCORE + ply
        else:
            stand_pat = self.evaluator.evaluate(board)
            if stand_pat >= beta:
                return stand_pat
            # Delta pruning.
            if stand_pat < alpha - 975:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat

            # Generate captures; include non-capture checks at first QS ply.
            captures = list(board.generate_legal_captures())

            if qs_depth == 0:
                # Include non-capture checking moves at first QS ply.
                check_moves = []
                for move in board.legal_moves:
                    if not board.is_capture(move) and board.gives_check(move):
                        check_moves.append(move)
                moves = captures + check_moves
            else:
                moves = captures

        # Sort moves: captures by MVV-LVA (fast), checks lower priority
        def qs_move_score(m):
            if board.is_capture(m):
                return self._mvv_lva(board, m) + 100000
            else:
                return 50000  # Non-capture checks get medium priority

        moves.sort(key=qs_move_score, reverse=True)

        best_score = stand_pat if not in_check else -INF
        best_move = None

        for move in moves:
            # SEE pruning: skip clearly losing captures.
            if not in_check and board.is_capture(move):
                if self._see(board, move) < -50:
                    continue

            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, qs_depth + 1, ply + 1)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if score >= beta:
                if not self._stop_event.is_set():
                    self.tt.store(board, 0, best_score, TT_BETA, move)
                return best_score
            if score > alpha:
                alpha = score

        # Store quiescence result in TT
        if not self._stop_event.is_set():
            if best_score <= alpha_orig:
                flag = TT_ALPHA
            elif best_score >= beta:
                flag = TT_BETA
            else:
                flag = TT_EXACT
            self.tt.store(board, 0, best_score, flag, best_move)

        return best_score

    def _see(self, board: chess.Board, move: chess.Move) -> int:
        """Static Exchange Evaluation: estimate material gain/loss of a capture sequence."""
        to_sq = move.to_square
        from_sq = move.from_square

        # Determine the initial captured value
        if board.is_en_passant(move):
            captured_val = SEE_PIECE_VALUES[chess.PAWN]
        else:
            victim = board.piece_at(to_sq)
            if victim is None:
                return 0
            captured_val = SEE_PIECE_VALUES[victim.piece_type]

        attacker = board.piece_at(from_sq)
        if attacker is None:
            return 0

        attacker_val = SEE_PIECE_VALUES[attacker.piece_type]

        # Quick exit: capturing with equal or lesser value is always favorable.
        if attacker_val <= captured_val:
            return captured_val - attacker_val

        # Iterative SEE with gain array.
        gain = [0] * 32
        gain[0] = captured_val

        current_attacker_val = attacker_val
        side = not attacker.color  # Opponent moves next
        d = 0

        # Track consumed squares.
        used = chess.SquareSet()
        used.add(from_sq)

        while True:
            d += 1
            gain[d] = current_attacker_val - gain[d - 1]

            # If continuing can't improve, stop.
            if max(-gain[d - 1], gain[d]) < 0:
                break

            # Find least valuable attacker for the side to move.
            attackers = board.attackers(side, to_sq)
            best_atk_sq = None
            min_val = 20001

            for sq in attackers:
                if sq in used:
                    continue
                p = board.piece_at(sq)
                if p is not None and SEE_PIECE_VALUES[p.piece_type] < min_val:
                    min_val = SEE_PIECE_VALUES[p.piece_type]
                    best_atk_sq = sq

            if best_atk_sq is None:
                break

            used.add(best_atk_sq)
            current_attacker_val = min_val
            side = not side

        # Negamax the gain array.
        while d > 0:
            gain[d - 1] = -max(-gain[d - 1], gain[d])
            d -= 1

        return gain[0]

    def _order_moves(self, board: chess.Board, tt_move: Optional[chess.Move], ply: int):
        """Sort legal moves: TT move > good captures > killers > history heuristic."""
        def score_move(move):
            if move == tt_move:
                return 2000000
            elif board.is_capture(move):
                # MVV-LVA for fast ordering; SEE sign for good/bad split
                mvv_lva = self._mvv_lva(board, move)
                if mvv_lva >= 0:
                    return mvv_lva + 200000  # Good captures.
                else:
                    return mvv_lva + 50000  # Likely losing captures.
            elif move == self.killers[ply][0]:
                return 110000
            elif move == self.killers[ply][1]:
                return 100000
            else:
                return self.history[move.from_square][move.to_square]

        moves = list(board.legal_moves)
        moves.sort(key=score_move, reverse=True)
        return moves

    def _mvv_lva(self, board, move):
        """Most-Valuable-Victim / Least-Valuable-Attacker heuristic."""
        attacker = board.piece_at(move.from_square)
        if board.is_en_passant(move):
            victim_type = chess.PAWN
        else:
            victim = board.piece_at(move.to_square)
            victim_type = victim.piece_type if victim else chess.PAWN
        attacker_type = attacker.piece_type if attacker else chess.PAWN
        return (victim_type * 10) - attacker_type

    def close(self):
        if self.tablebase:
            self.tablebase.close()
