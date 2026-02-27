import chess
import chess.polyglot

# Constants matching search
TT_EXACT = 0
TT_ALPHA = 1
TT_BETA = 2


class TTEntry:
    __slots__ = ("depth", "value", "flag", "best_move")

    def __init__(self, depth, value, flag, best_move):
        self.depth = depth
        self.value = value
        self.flag = flag
        self.best_move = best_move


class TranspositionTable:
    def __init__(self, size_mb=64):
        # ~48 bytes per entry (object overhead + slots)
        entry_size = 48
        self.max_entries = max(1, (size_mb * 1024 * 1024) // entry_size)
        self.table = {}
    
    def get(self, board):
        key = chess.polyglot.zobrist_hash(board)
        return self.table.get(key % self.max_entries)
        
    def store(self, board, depth, value, flag, best_move):
        key = chess.polyglot.zobrist_hash(board)
        idx = key % self.max_entries
        existing = self.table.get(idx)
        # Depth-preferred replacement: only overwrite with equal or deeper search
        if existing is None or depth >= existing.depth:
            self.table[idx] = TTEntry(depth, value, flag, best_move)
    
    def clear(self):
        self.table.clear()