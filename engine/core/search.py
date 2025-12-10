import chess
from chess import polyglot
import time
import threading
import os
from collections import defaultdict
from typing import Optional, Callable, List, Tuple
# from engine.core.loopboard_evaluator import Evaluator
from engine.core.bitboard_evaluator import BitboardEvaluator as Evaluator
from engine.core.transposition import TranspositionTable
from engine.core.utils import print_info

INF = 1000000
MATE_SCORE = 900000

TT_EXACT = 0
TT_ALPHA = 1
TT_BETA = 2

class SearchEngine:
    def __init__(self, evaluator: Optional[Evaluator] = None, depth: int = 4):
        self.evaluator = evaluator or Evaluator()
        self.max_depth = depth
        self.tt = TranspositionTable()
        self.history = defaultdict(lambda: defaultdict(int)) 
        
        self.killers = [[None] * 2 for _ in range(128)]
        
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.nodes = 0
         
        base_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(base_dir, "../../gui/assets")
        self.book_paths = [
            os.path.join(assets_dir, "Titans.bin"),
            os.path.join(assets_dir, "gm2600.bin"),
            os.path.join(assets_dir, "komodo.bin"),
            os.path.join(assets_dir, "rodent.bin"),
            os.path.join(assets_dir, "Human.bin")
        ]
        
        
    def get_book_move(self, board:chess.Board)-> Optional[chess.Move]:
        """Check if the current position is in the opening book."""
        for book_path in self.book_paths:
            if os.path.exists(book_path):
                try:
                    with polyglot.open_reader(book_path) as reader:
                        entry = reader.weighted_choice(board)
                        print(f"[{os.path.basename(book_path)}] Book Move: {entry.move.uci()}")
                        return entry.move
                except:
                    continue
        return None

    def search_best_move(self, board: chess.Board) ->Tuple[Optional[chess.Move],Optional[chess.Move]]:
        self._stop_event.clear()
        
        book_move = self.get_book_move(board)
        if book_move:
            return book_move,None
        
        self.nodes = 0
        search_board = board.copy()
        best_move = None
        ponder_move = None
        
        start_time = time.time()

        # Iterative Deepening
        for d in range(1, self.max_depth + 1):
            if self._stop_event.is_set(): break
            
            # Run Search
            score = self._negamax(search_board, d, -INF, INF, 0)
            
            # Fetch Best Move
            entry = self.tt.get(search_board)
            if entry and entry.best_move:
                best_move = entry.best_move
            
            pv_moves = self._get_pv_line(search_board, d)
            if len(pv_moves) > 0: best_move = pv_moves[0]
            if len(pv_moves) > 1: ponder_move = pv_moves[1]

            elapsed = time.time() - start_time
            print_info(d, score, self.nodes, elapsed, pv_moves, ponder_move, MATE_SCORE)

        return best_move, ponder_move

    def start_search(self, board: chess.Board, depth: Optional[int] = None, callback: Optional[Callable] = None):
        if self._thread and self._thread.is_alive(): return 
        self._stop_event.clear()
        target_depth = depth or self.max_depth

        def worker():
            book_move = self.get_book_move(board)
            if book_move: 
                if callback: callback(book_move, None, 0, 0)
                return
            
            search_board = board.copy()
            best_move = None
            ponder_move = None
            self.nodes = 0
            start_time = time.time()

            for d in range(1, target_depth + 1):
                if self._stop_event.is_set(): break
                score = self._negamax(search_board, d, -INF, INF, 0)
                
                entry = self.tt.get(search_board)
                if entry: best_move = entry.best_move
                
                # Print to terminal for debugging
                pv_moves = self._get_pv_line(search_board, d)
                if len(pv_moves) > 0: best_move = pv_moves[0]
                if len(pv_moves) > 1: ponder_move = pv_moves[1]
                
                elapsed = time.time() - start_time
                print_info(d, score, self.nodes, elapsed, pv_moves, ponder_move, MATE_SCORE)                
                
                if callback: callback(best_move, ponder_move, d, score)
            
            if callback: callback(best_move, ponder_move, -1, 0)

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.2)

    def _get_pv_line(self, board: chess.Board, depth: int) -> List[chess.Move]:
        pv_moves = []
        curr_board = board.copy()
        
        # Avoid infinite loops (max depth or repetition)
        seen_hashes = set()

        for _ in range(depth):
            entry = self.tt.get(curr_board)
            if not entry or not entry.best_move:
                break
            
            move = entry.best_move
            if move not in curr_board.legal_moves:
                break
                
            pv_moves.append(move)
            curr_board.push(move)
            
            # cycle detection
            fen = curr_board.fen()
            if fen in seen_hashes:
                break
            seen_hashes.add(fen)
            
        return pv_moves

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        if board.can_claim_draw():
            current_eval = self.evaluator.evaluate(board)
            if current_eval > 50:
                return -50 
            return 0 
        if self.nodes % 2048 == 0 and self._stop_event.is_set(): return 0
        if board.is_fivefold_repetition(): return 0

        alpha_orig = alpha

        # TT Lookup
        tt_entry = self.tt.get(board)
        tt_move = None
        
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == TT_EXACT: return tt_entry.value
            elif tt_entry.flag == TT_ALPHA: alpha = max(alpha, tt_entry.value)
            elif tt_entry.flag == TT_BETA: beta = min(beta, tt_entry.value)
            if alpha >= beta: return tt_entry.value
        
        if tt_entry: tt_move = tt_entry.best_move

        if depth <= 0:
            return self._quiescence(board, alpha, beta)
        
        if board.is_game_over():
            if board.is_checkmate(): return -MATE_SCORE + ply
            return 0
        
        has_big_pieces = any(board.pieces(pt,board.turn)
                             for pt in [chess.KNIGHT, chess.BISHOP
                                        , chess.ROOK, chess.QUEEN])

        # Null Move Pruning
        if depth >= 3 and not board.is_check() and ply > 0:
            board.push(chess.Move.null())
            score = -self._negamax(board, depth - 3, -beta, -beta + 1, ply + 1)
            board.pop()
            if score >= beta: return beta

        moves = self._order_moves(board, tt_move, ply)
        best_score = -INF
        best_move_found = None
        
        moves_searched = 0
        
        for move in moves:
            board.push(move)
            moves_searched += 1
            needs_full_search = True
            if depth >= 3 and moves_searched > 4 and not board.is_capture(move) and not board.is_check():
                score = -self._negamax(board, depth - 2, -alpha-1, -alpha, ply + 1)
                needs_full_search = score > alpha
                
            if needs_full_search:
                score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
                
            board.pop()
            
            if self._stop_event.is_set(): return 0

            if score > best_score:
                best_score = score
                best_move_found = move

            if score > alpha:
                alpha = score
                if not board.is_capture(move):
                    self.history[move.from_square][move.to_square] += depth * depth
                    if move != self.killers[ply][0]:
                        self.killers[ply][1] = self.killers[ply][0]
                        self.killers[ply][0] = move
                        
                if alpha >= beta:
                    self.tt.store(board, depth, beta, TT_BETA, move)
                    return beta

        if best_score <= alpha_orig:
            flag = TT_ALPHA  
        elif best_score >= beta:
            flag = TT_BETA  
        else:
            flag = TT_EXACT  

        self.tt.store(board, depth, best_score, flag, best_move_found)
        return best_score

    def _quiescence(self, board: chess.Board, alpha: int, beta: int) -> int:
        if self._stop_event.is_set(): return 0
        
        stand_pat = self.evaluator.evaluate(board)
        if stand_pat >= beta: return beta
        if stand_pat < alpha - 900: return alpha
        if stand_pat > alpha: alpha = stand_pat

        moves = list(board.generate_legal_captures())
        moves.sort(key=lambda m: self._mvv_lva(board, m), reverse=True)

        for move in moves:
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha)
            board.pop()
            if score >= beta: return beta
            if score > alpha: alpha = score
            
        return alpha

    def _order_moves(self, board: chess.Board, tt_move: Optional[chess.Move], ply: int):
        moves = list(board.legal_moves)
        scores = []
        for move in moves:
            if move == tt_move:
                scores.append(2000000)
            elif board.is_capture(move):
                scores.append(self._mvv_lva(board, move) + 100000)
            elif move == self.killers[ply][0]:
                scores.append(90000)
            elif move == self.killers[ply][1]:
                scores.append(80000)
            else:
                scores.append(self.history[move.from_square][move.to_square])
        
        return [m for _, m in sorted(zip(scores, moves), key=lambda x: x[0], reverse=True)]

    def _mvv_lva(self, board, move):
        attacker = board.piece_at(move.from_square)
        if board.is_en_passant(move):
            victim_type = chess.PAWN
        else:
            victim = board.piece_at(move.to_square)
            victim_type = victim.piece_type if victim else None
        return (victim_type * 10) - attacker.piece_type