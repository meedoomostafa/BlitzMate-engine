import sys
import chess
from engine.main import Engine

class UCI:
    def __init__(self):
        self.engine = Engine()

    def run(self):
        while True:
            command = input()
            if command == "uci":
                print("id name MedoEngine")
                print("id author Medo")
                print("uciok")
            elif command.startswith("position"):
                # parse FEN or moves list
                ...
            elif command == "isready":
                print("readyok")
            elif command.startswith("go"):
                move, value = self.engine.get_best_move()
                print(f"bestmove {move}")
            elif command == "quit":
                break
