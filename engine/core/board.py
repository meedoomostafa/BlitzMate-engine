"""Board wrapper over python-chess providing move history tracking."""

import chess


class ChessBoard:
    def __init__(self, fen: str = None):
        """Initialize from FEN or the standard starting position."""
        self.board = chess.Board(fen) if fen else chess.Board()
        self.move_history = []

    def reset(self):
        """Reset to the initial position."""
        self.board.reset()
        self.move_history.clear()

    def set_fen(self, fen: str):
        """Set board state from a FEN string."""
        self.board.set_fen(fen)

    def get_fen(self) -> str:
        """Return the current FEN."""
        return self.board.fen()

    def make_move(self, move_str: str) -> bool:
        """Push a UCI move (e.g. 'e2e4'). Returns True if legal."""
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
        """Pop the last move."""
        if self.move_history:
            self.board.pop()
            self.move_history.pop()

    def get_legal_moves(self):
        """Return legal moves as UCI strings."""
        return [m.uci() for m in self.board.legal_moves]

    def is_game_over(self):
        """Check if the game has ended."""
        return self.board.is_game_over()

    def print_board(self):
        """Print ASCII representation."""
        print(self.board)
