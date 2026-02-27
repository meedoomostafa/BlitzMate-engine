import sys
import chess
import threading
from engine.config import CONFIG
from engine.core.search import SearchEngine
from engine.core.bitboard_evaluator import BitboardEvaluator


class UCI:
    def __init__(self):
        self.board = chess.Board()
        self.engine = SearchEngine(BitboardEvaluator(), depth=CONFIG.search.depth)
        self._movetime_timer: threading.Timer | None = None

    def run(self):
        while True:
            try:
                line = input().strip()
            except EOFError:
                break
            if not line:
                continue

            tokens = line.split()
            cmd = tokens[0]

            if cmd == "uci":
                print(f"id name {CONFIG.ui.engine_name}")
                print(f"id author {CONFIG.ui.engine_author}")
                print(
                    f"option name Hash type spin default {CONFIG.search.hash_size_mb} min 1 max 4096"
                )
                print(
                    f"option name Threads type spin default {CONFIG.search.threads} min 1 max 128"
                )
                print("uciok")

            elif cmd == "isready":
                print("readyok")

            elif cmd == "ucinewgame":
                self.engine.stop()
                self.board = chess.Board()
                self.engine = SearchEngine(
                    BitboardEvaluator(), depth=CONFIG.search.depth
                )

            elif cmd == "position":
                self._parse_position(tokens[1:])

            elif cmd == "go":
                self._parse_go(tokens[1:])

            elif cmd == "stop":
                if self._movetime_timer is not None:
                    self._movetime_timer.cancel()
                self.engine.stop()

            elif cmd == "quit":
                self.engine.stop()
                break

            elif cmd == "setoption":
                self._parse_setoption(tokens[1:])

            sys.stdout.flush()

    def _parse_position(self, tokens):
        idx = 0
        if not tokens:
            return

        if tokens[0] == "startpos":
            self.board = chess.Board()
            idx = 1
        elif tokens[0] == "fen":
            # Collect FEN parts (up to 6 tokens after "fen")
            fen_parts = []
            idx = 1
            while idx < len(tokens) and tokens[idx] != "moves":
                fen_parts.append(tokens[idx])
                idx += 1
            try:
                self.board = chess.Board(" ".join(fen_parts))
            except ValueError:
                return

        # Apply moves
        if idx < len(tokens) and tokens[idx] == "moves":
            for move_str in tokens[idx + 1 :]:
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                except ValueError:
                    break

    def _parse_go(self, tokens):
        depth = CONFIG.search.depth
        movetime = None
        wtime = None
        btime = None
        winc = 0
        binc = 0

        i = 0
        while i < len(tokens):
            if tokens[i] == "depth" and i + 1 < len(tokens):
                try:
                    depth = int(tokens[i + 1])
                except ValueError:
                    pass
                i += 2
            elif tokens[i] == "movetime" and i + 1 < len(tokens):
                try:
                    movetime = int(tokens[i + 1])
                except ValueError:
                    pass
                i += 2
            elif tokens[i] == "wtime" and i + 1 < len(tokens):
                try:
                    wtime = int(tokens[i + 1])
                except ValueError:
                    pass
                i += 2
            elif tokens[i] == "btime" and i + 1 < len(tokens):
                try:
                    btime = int(tokens[i + 1])
                except ValueError:
                    pass
                i += 2
            elif tokens[i] == "winc" and i + 1 < len(tokens):
                try:
                    winc = int(tokens[i + 1])
                except ValueError:
                    pass
                i += 2
            elif tokens[i] == "binc" and i + 1 < len(tokens):
                try:
                    binc = int(tokens[i + 1])
                except ValueError:
                    pass
                i += 2
            elif tokens[i] == "infinite":
                depth = 64
                i += 1
            else:
                i += 1

        # Time management: convert wtime/btime into a movetime budget
        if movetime is None and (wtime is not None or btime is not None):
            time_left = wtime if self.board.turn == chess.WHITE else btime
            inc = winc if self.board.turn == chess.WHITE else binc
            if time_left is not None and time_left > 0:
                # Estimate ~30 moves remaining, allocate time_left/30 + increment
                moves_to_go = max(20, 40 - self.board.fullmove_number)
                movetime = max(100, time_left // moves_to_go + inc * 3 // 4)
                # Never use more than 80% of remaining time
                movetime = min(movetime, int(time_left * 0.8))
                depth = 64  # search until time runs out

        def on_search_done(best_move, ponder_move, d, score):
            if d <= 0:  # Search finished
                # Cancel movetime timer if search ended naturally
                if self._movetime_timer is not None:
                    self._movetime_timer.cancel()
                    self._movetime_timer = None
                result = f"bestmove {best_move.uci()}" if best_move else "bestmove 0000"
                if ponder_move:
                    result += f" ponder {ponder_move.uci()}"
                print(result)
                sys.stdout.flush()

        self.engine.start_search(
            self.board.copy(), depth=depth, callback=on_search_done
        )

        # Start a timer to stop the search after movetime
        if movetime is not None:
            if self._movetime_timer is not None:
                self._movetime_timer.cancel()
            self._movetime_timer = threading.Timer(movetime / 1000.0, self.engine.stop)
            self._movetime_timer.daemon = True
            self._movetime_timer.start()

    def _parse_setoption(self, tokens):
        # setoption name <name> value <value>
        try:
            name_idx = tokens.index("name") + 1
            value_idx = tokens.index("value") + 1
            name = tokens[name_idx].lower()
            value = tokens[value_idx]

            if name == "hash":
                pass  # TT resize not implemented yet
            elif name == "threads":
                pass  # Single-threaded for now
        except (ValueError, IndexError):
            pass


if __name__ == "__main__":
    uci = UCI()
    uci.run()
