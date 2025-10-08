import chess
from core.search import SearchEngine

# initialize board and engine
board = chess.Board()
engine = SearchEngine(depth=6)

while not board.is_game_over():
    print(board)
    print("----------------------------")

    if board.turn == chess.WHITE:  # الإنسان يلعب الأبيض
        user_move = input("Enter your move (uci format, e2e4): ")
        move = chess.Move.from_uci(user_move)
        if move in board.legal_moves:
            board.push(move)
        else:
            print("Illegal move, try again.")
            continue
    else:
        move, score = engine.search_best_move(board)
        print(f"Engine plays: {move} | Eval: {score:.2f}")
        board.push(move)

print("Game Over")
print(f"Result: {board.result()}")


    # Optional: pause to follow moves easily
    # input("Press Enter for next move...")

