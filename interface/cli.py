import chess
from engine.core.search import SearchEngine

# initialize board and engine
board = chess.Board()
engine = SearchEngine(depth=6)

while not board.is_game_over():
    print(board)
    print("----------------------------")

    if board.turn == chess.WHITE:
        user_move = input("Enter your move (uci format, e.g. e2e4): ")
        try:
            move = chess.Move.from_uci(user_move)
        except chess.InvalidMoveError:
            print("Invalid move format, try again.")
            continue
        if move in board.legal_moves:
            board.push(move)
        else:
            print("Illegal move, try again.")
            continue
    else:
        move, _ponder, score = engine.search_best_move(board)
        if move is None:
            print("Engine has no legal moves.")
            break
        print(f"Engine plays: {move} | Eval: {score / 100:.2f}")
        board.push(move)

print("Game Over")
print(f"Result: {board.result()}")
