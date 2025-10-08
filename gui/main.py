import pygame
import chess
import threading
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine.core.search import SearchEngine  

pygame.init()

SQUARE_SIZE = 80
BOARD_SIZE = 8 * SQUARE_SIZE
screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
pygame.display.set_caption("Python Chess Engine GUI")

light = (240, 217, 181)
dark = (181, 136, 99)
highlight = (186, 202, 68)

pieces = {}
for piece in ["r", "n", "b", "q", "k", "p"]:
    img_black = pygame.image.load(f"assets/b{piece.upper()}.png")
    img_white = pygame.image.load(f"assets/w{piece.upper()}.png")
    pieces[piece] = pygame.transform.smoothscale(img_black, (SQUARE_SIZE, SQUARE_SIZE))
    pieces[piece.upper()] = pygame.transform.smoothscale(img_white, (SQUARE_SIZE, SQUARE_SIZE))

board = chess.Board()
engine = SearchEngine(depth=6)  # use smaller depth first (6 will be slow)
selected_square = None
engine_thread = None

# Custom event to deliver engine move to main thread
ENGINE_MOVE_EVENT = pygame.USEREVENT + 1


def draw_board():
    for row in range(8):
        for col in range(8):
            square = chess.square(col, 7 - row)
            color = light if (row + col) % 2 == 0 else dark
            if selected_square == square:
                color = highlight
            pygame.draw.rect(screen, color, (col*SQUARE_SIZE, row*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            piece = board.piece_at(square)
            if piece:
                screen.blit(pieces[piece.symbol()], (col*SQUARE_SIZE, row*SQUARE_SIZE))


def run_engine():
    """Run engine in background thread and send move event."""
    best_move = engine.search_best_move(board.copy())
    # if your engine returns (move, score), comment the above and uncomment below:
    # best_move, score = engine.search_best_move(board.copy())

    if isinstance(best_move, tuple):
        # In case it returns (move, score)
        best_move = best_move[0]

    if best_move is not None:
        pygame.event.post(pygame.event.Event(ENGINE_MOVE_EVENT, {"move": best_move}))


running = True
while running:
    draw_board()
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            col = x // SQUARE_SIZE
            row = 7 - (y // SQUARE_SIZE)
            square = chess.square(col, row)
            piece = board.piece_at(square)

            if selected_square is None:
                # Select a piece if it exists
                if piece:
                    selected_square = square
            else:
                # Try to move
                move = chess.Move(selected_square, square)
                if move in board.legal_moves:
                    board.push(move)
                    selected_square = None
                    
                    if board.is_game_over():
                        print("Game Over", board.result())

                    # Run engine in background after player's move
                    else:
                        if not (engine_thread and engine_thread.is_alive()):
                            engine_thread = threading.Thread(target=run_engine, daemon=True)
                            engine_thread.start()
                else:
                    # If clicked square has your own piece, re-select it
                    if piece and piece.color == board.turn:
                        selected_square = square
                    else:
                        # Invalid move, deselect
                        selected_square = None


        elif event.type == ENGINE_MOVE_EVENT:
            move = event.move
            if isinstance(move, chess.Move) and move in board.legal_moves:
                board.push(move)
                print("ENGINE MOVE:", move.uci())

pygame.quit()