import chess
import chess.polyglot

# Constants matching search
TT_EXACT = 0
TT_ALPHA = 1
TT_BETA = 2


class TTEntry:
    __slots__ = ('key', 'depth', 'value', 'flag', 'best_move')
    def __init__(self, key, depth, value, flag, best_move):
        self.key = key
        self.depth = depth
        self.value = value
        self.flag = flag
        self.best_move = best_move


class TranspositionTable:
    def __init__(self, size_mb=64):
        # ~56 bytes per entry (object overhead + slots + key)
        entry_size = 56
        self.max_entries = max(1, (size_mb * 1024 * 1024) // entry_size)
        self.table = {}
    
    def get(self, board):
        key = chess.polyglot.zobrist_hash(board)
        idx = key % self.max_entries
        entry = self.table.get(idx)
        if entry and entry.key == key:
            return entry
        return None
        
    def store(self, board, depth, value, flag, best_move):
        key = chess.polyglot.zobrist_hash(board)
        idx = key % self.max_entries
        existing = self.table.get(idx)
        # Depth-preferred replacement: only overwrite with equal or deeper search
        if existing is None or depth >= existing.depth or existing.key != key:
            self.table[idx] = TTEntry(key, depth, value, flag, best_move)
    
    def clear(self):
        self.table.clear()