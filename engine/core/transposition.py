"""Zobrist-hashed transposition table with depth-preferred replacement."""

import chess
import chess.polyglot

# Node type flags.
TT_EXACT = 0
TT_ALPHA = 1
TT_BETA = 2


class TTEntry:
    """Single transposition table entry."""

    __slots__ = ("key", "depth", "value", "flag", "best_move", "age")

    def __init__(self, key, depth, value, flag, best_move, age=0):
        self.key = key
        self.depth = depth
        self.value = value
        self.flag = flag
        self.best_move = best_move
        self.age = age


class TranspositionTable:
    """Hash map of TTEntry objects, capped by memory budget."""

    def __init__(self, size_mb=64):
        entry_size = 56  # Approximate bytes per Python TTEntry object.
        self.max_entries = max(1, (size_mb * 1024 * 1024) // entry_size)
        self.table = {}
        self.generation = 0

    def new_search(self):
        """Increment generation counter between moves to age stale entries."""
        self.generation += 1

    def get(self, board):
        """Probe the table. Returns TTEntry or None."""
        key = chess.polyglot.zobrist_hash(board)
        idx = key % self.max_entries
        entry = self.table.get(idx)
        if entry and entry.key == key:
            return entry
        return None

    def store(self, board, depth, value, flag, best_move):
        """Store or replace using depth-preferred + age-based strategy.

        An entry is replaced if:
          - The slot is empty, OR
          - The new entry has >= depth, OR
          - The existing entry is from an older generation (stale), OR
          - The key doesn't match (hash collision).
        """
        key = chess.polyglot.zobrist_hash(board)
        idx = key % self.max_entries
        existing = self.table.get(idx)
        if (
            existing is None
            or depth >= existing.depth
            or existing.key != key
            or existing.age != self.generation
        ):
            self.table[idx] = TTEntry(
                key, depth, value, flag, best_move, self.generation
            )

    def clear(self):
        """Flush all entries."""
        self.table.clear()
