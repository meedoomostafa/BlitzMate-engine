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
        self.table = {}

    def get(self, board):
        key = chess.polyglot.zobrist_hash(board)
        return self.table.get(key)

    def store(self, board, depth, value, flag, best_move):
        key = chess.polyglot.zobrist_hash(board)
        self.table[key] = TTEntry(depth, value, flag, best_move)
