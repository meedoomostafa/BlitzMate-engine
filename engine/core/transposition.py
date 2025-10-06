"""Zobrist hashing and a thread-safe transposition table.

This module provides two main classes:

- Zobrist: builds random zobrist keys and computes a 64-bit key for any
  python-chess Board. Implementation computes from-scratch (no incremental
  maintenance) which is simpler and safe; you can extend it later to keep
  an incremental key updated on push/pop for more speed.

- TranspositionTable: a small thread-safe wrapper around a dict keyed by
  zobrist keys. Each entry stores the FEN for collision-detection, the
  search depth, stored value and flag, plus the best move.

Usage (example):

    from engine.core.transposition import TranspositionTable

    tt = TranspositionTable()
    entry = tt.get(board)
    if entry is not None:
        # entry is a TTEntry dataclass
        print(entry.fen, entry.depth, entry.value, entry.flag, entry.best_move)

    # store
    tt.store(board, depth=3, value=123, flag=0, best_move=some_move)

"""
from __future__ import annotations

import random
import threading
from dataclasses import dataclass
from typing import Optional, Dict

import chess

# pieces symbols as returned by piece.symbol() in python-chess
PIECE_SYMBOLS = ["P", "N", "B", "R", "Q", "K",
                 "p", "n", "b", "r", "q", "k"]


def _rand64() -> int:
    return random.getrandbits(64)


def make_zobrist_table() -> Dict[str, object]:
    """Create a fresh zobrist table.

    Structure returned:
      {
        "piece": { 'P': [64 ints], 'N': [64 ints], ... },
        "side": int,
        "castling": [16 ints],
        "ep": [8 ints]
      }
    """
    piece_table = {sym: [_rand64() for _ in range(64)] for sym in PIECE_SYMBOLS}
    side_key = _rand64()
    castling_table = [_rand64() for _ in range(16)]
    ep_table = [_rand64() for _ in range(8)]
    return {"piece": piece_table, "side": side_key, "castling": castling_table, "ep": ep_table}


@dataclass
class TTEntry:
    fen: str
    depth: int
    value: int
    flag: int
    best_move: Optional[chess.Move]

    def __iter__(self):
        return iter((self.fen, self.depth, self.value, self.flag, self.best_move))



class Zobrist:
    """Zobrist hash utilities.

    This implementation computes the key from scratch for a given Board.
    It's simple and reasonably fast for many use-cases; if you need maximum
    performance later, implement incremental updates on push/pop.
    """

    def __init__(self):
        self.table = make_zobrist_table()

    def hash(self, board: chess.Board) -> int:
        t = self.table
        h = 0
        # pieces
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p is not None:
                sym = p.symbol()
                arr = t["piece"].get(sym)
                if arr is not None:
                    h ^= arr[sq]
        # side: xor when black to move (convention)
        if board.turn == chess.BLACK:
            h ^= t["side"]
        # castling rights into a small index (0..15)
        cr = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            cr |= 1
        if board.has_queenside_castling_rights(chess.WHITE):
            cr |= 2
        if board.has_kingside_castling_rights(chess.BLACK):
            cr |= 4
        if board.has_queenside_castling_rights(chess.BLACK):
            cr |= 8
        h ^= t["castling"][cr]
        # en-passant file if available
        ep = board.ep_square
        if ep is not None:
            f = chess.square_file(ep)
            h ^= t["ep"][f]
        return h


class TranspositionTable:
    """Thread-safe transposition table keyed by zobrist hash.

    Each stored entry is a TTEntry with fen stored for collision checking.
    Methods:
      - get(board) -> Optional[TTEntry]
      - store(board, depth, value, flag, best_move)
      - clear()
      - key(board) -> int  (zobrist key)
    """

    def __init__(self, seed: Optional[int] = None):
        # optional seed could be used to seed random in make_zobrist_table
        # but we rely on module-level random for simplicity
        self.z = Zobrist()
        self._table: Dict[int, TTEntry] = {}
        self._lock = threading.Lock()

    def key(self, board: chess.Board) -> int:
        return self.z.hash(board)

    def get(self, board: chess.Board) -> Optional[TTEntry]:
        k = self.key(board)
        with self._lock:
            entry = self._table.get(k)
        if entry is None:
            return None
        # verify fen to avoid rare collisions
        if entry.fen != board.fen():
            return None
        return entry

    def store(self, board: chess.Board, depth: int, value: int, flag: int, best_move: Optional[chess.Move]):
        k = self.key(board)
        entry = TTEntry(board.fen(), depth, value, flag, best_move)
        with self._lock:
            self._table[k] = entry

    def clear(self):
        with self._lock:
            self._table.clear()


# small self-test when run directly
if __name__ == "__main__":
    b = chess.Board()
    tt = TranspositionTable()
    k = tt.key(b)
    print("Key:", k)
    tt.store(b, 3, 100, 0, None)
    e = tt.get(b)
    print(e)
