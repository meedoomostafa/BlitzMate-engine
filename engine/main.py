from engine.core.board import ChessBoard
from engine.core.search import SearchEngine
from engine.core.loopboard_evaluator import Evaluator

class Engine:
    def __init__(self, depth=3):
        self.board = ChessBoard()
        self.search = SearchEngine(Evaluator(), max_depth=depth)

    def get_best_move(self):
        move, value = self.search.find_best_move(self.board.board)
        return move.uci(), value

    def make_move(self, move_uci: str):
        return self.board.make_move(move_uci)

    def print_board(self):
        self.board.print_board()
