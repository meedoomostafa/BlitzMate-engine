# chess_engine/core/board.py

import chess


class ChessBoard:
    def __init__(self, fen: str = None):
        """
        Initialize a new chess board.
        If fen is None, start from the standard opening position.
        """
        self.board = chess.Board(fen) if fen else chess.Board()
        self.move_history = []

    def reset(self):
        """Reset to initial position."""
        self.board.reset()
        self.move_history.clear()

    def set_fen(self, fen: str):
        """Load a board position from FEN string."""
        self.board.set_fen(fen)

    def get_fen(self) -> str:
        """Return current FEN representation."""
        return self.board.fen()

    def make_move(self, move_str: str) -> bool:
        """
        Try to make a move (e.g., 'e2e4').
        Return True if valid, False if illegal.
        """
        try:
            move = chess.Move.from_uci(move_str)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.move_history.append(move_str)
                return True
            return False
        except Exception:
            return False

    def undo_move(self):
        """Undo the last move if exists."""
        if self.move_history:
            self.board.pop()
            self.move_history.pop()

    def get_legal_moves(self):
        """Return all legal moves in UCI format."""
        return [m.uci() for m in self.board.legal_moves]

    def is_game_over(self):
        """Check if game has ended."""
        return self.board.is_game_over()

    def print_board(self):
        """Print board in ASCII form."""
        print(self.board)
