from engine.core.board import ChessBoard
from engine.core.search import SearchEngine
from engine.core.bitboard_evaluator import BitboardEvaluator


class Engine:
    def __init__(self, depth=6):
        self.board = ChessBoard()
        self.search = SearchEngine(BitboardEvaluator(), depth=depth)

    def get_best_move(self):
        move, ponder, score = self.search.search_best_move(self.board.board)
        return move.uci() if move else None, ponder.uci() if ponder else None, score

    def make_move(self, move_uci: str):
        return self.board.make_move(move_uci)

    def print_board(self):
        self.board.print_board()
