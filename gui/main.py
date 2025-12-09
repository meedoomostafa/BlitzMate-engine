import pygame
import chess
import threading
import sys, os

# To ensure we can import from the engine folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine.config import CONFIG
from engine.core.search import SearchEngine  

pygame.init()

# Board
SQUARE_SIZE = 110
BOARD_SIZE = 8 * SQUARE_SIZE
screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
pygame.display.set_caption(f"BlitzMate GUI - {CONFIG.ui.engine_name}")

# Colors
LIGHT_COLOR = (240, 217, 181)
DARK_COLOR = (181, 136, 99)
HIGHLIGHT_COLOR = (186, 202, 68)
LAST_MOVE_COLOR = (205, 210, 106)
PREMOVE_COLOR = (200, 100, 100) 
PREMOVE_HIGHLIGHT = (220, 120, 120)
PROMOTION_BG_COLOR = (255, 255, 255)

# Load Assets
pieces = {}
base_folder = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(base_folder, "assets")

def load_assets():
    for piece in ["r", "n", "b", "q", "k", "p"]:
        try:
            img_black = pygame.image.load(os.path.join(assets_path, f"b{piece.upper()}.png"))
            img_white = pygame.image.load(os.path.join(assets_path, f"w{piece.upper()}.png"))
            pieces[piece] = pygame.transform.smoothscale(img_black, (SQUARE_SIZE, SQUARE_SIZE))
            pieces[piece.upper()] = pygame.transform.smoothscale(img_white, (SQUARE_SIZE, SQUARE_SIZE))
        except FileNotFoundError:
            print(f"Warning: Asset {piece} not found in {assets_path}")

load_assets()

# Initial Part
board = chess.Board()
engine = SearchEngine(depth=CONFIG.search.depth) 
engine_thread = None
game_over = False

# Interaction State
selected_square = None
dragging_square = None
dragging_piece = None
mouse_pos = (0, 0)

# Promotion State
promotion_pending = False
promotion_move_start = None
promotion_move_end = None
promotion_color = None

# Premove State
premove = None

# Custom Events
ENGINE_MOVE_EVENT = pygame.USEREVENT + 1


# #-----------------------#
# | Helper Functions Part |
# #-----------------------#

def get_square_under_mouse(pos):
    col = pos[0] // SQUARE_SIZE
    row = 7 - (pos[1] // SQUARE_SIZE)
    if 0 <= col <= 7 and 0 <= row <= 7:
        return chess.square(col, row)
    return None

def draw_board():
    for row in range(8):
        for col in range(8):
            square = chess.square(col, 7 - row)
            color = LIGHT_COLOR if (row + col) % 2 == 0 else DARK_COLOR
            
            if square == selected_square:
                color = HIGHLIGHT_COLOR
            elif premove and (square == premove.from_square or square == premove.to_square):
                color = PREMOVE_HIGHLIGHT
            elif board.move_stack and (square == board.move_stack[-1].from_square or square == board.move_stack[-1].to_square):
                color = LAST_MOVE_COLOR

            pygame.draw.rect(screen, color, (col*SQUARE_SIZE, row*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

            # Draw Valid Move Hints (Dots) - Only show if it is HUMAN turn
            if board.turn == chess.WHITE and not premove:
                if selected_square is not None or dragging_square is not None:
                    source = dragging_square if dragging_square is not None else selected_square
                    candidate = chess.Move(source, square)
                    promo = chess.Move(source, square, promotion=chess.QUEEN)
                    if candidate in board.legal_moves or promo in board.legal_moves:
                        center = (col*SQUARE_SIZE + SQUARE_SIZE//2, row*SQUARE_SIZE + SQUARE_SIZE//2)
                        pygame.draw.circle(screen, (100, 100, 100, 100), center, SQUARE_SIZE//6)

            # Draw Pieces
            piece = board.piece_at(square)
            
            # If there is a premove, show the board AS IF the move happened
            if premove and square == premove.to_square:
                # Show the piece that moved here
                p_moved = board.piece_at(premove.from_square)
                if p_moved:
                    screen.blit(pieces[p_moved.symbol()], (col*SQUARE_SIZE, row*SQUARE_SIZE))
            elif premove and square == premove.from_square:
                # Show nothing here (it moved away)
                pass
            elif piece and square != dragging_square:
                screen.blit(pieces[piece.symbol()], (col*SQUARE_SIZE, row*SQUARE_SIZE))

    # Draw Dragging Piece
    if dragging_square is not None and dragging_piece:
        x, y = mouse_pos
        screen.blit(pieces[dragging_piece.symbol()], (x - SQUARE_SIZE//2, y - SQUARE_SIZE//2))

    # Draw Promotion Menu
    if promotion_pending:
        draw_promotion_menu()

def draw_promotion_menu():
    target_sq = promotion_move_end
    col = chess.square_file(target_sq)
    row = 7 - chess.square_rank(target_sq)
    menu_rect = pygame.Rect(col*SQUARE_SIZE, row*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE*4)
    if row > 4: menu_rect.y -= SQUARE_SIZE * 3
        
    pygame.draw.rect(screen, PROMOTION_BG_COLOR, menu_rect)
    pygame.draw.rect(screen, (0,0,0), menu_rect, 2)

    opts = ['q', 'r', 'b', 'n'] if promotion_color == chess.BLACK else ['Q', 'R', 'B', 'N']
    for i, p_char in enumerate(opts):
        y_offset = menu_rect.y + (i * SQUARE_SIZE)
        screen.blit(pieces[p_char], (menu_rect.x, y_offset))
        if i < 3:
            pygame.draw.line(screen, (0,0,0), (menu_rect.x, y_offset + SQUARE_SIZE), (menu_rect.right, y_offset + SQUARE_SIZE))

def handle_promotion_click(pos):
    global promotion_pending, promotion_move_start, promotion_move_end
    target_sq = promotion_move_end
    col = chess.square_file(target_sq)
    row = 7 - chess.square_rank(target_sq)
    base_x = col * SQUARE_SIZE
    base_y = row * SQUARE_SIZE
    if row > 4: base_y -= SQUARE_SIZE * 3
    rel_y = pos[1] - base_y
    idx = rel_y // SQUARE_SIZE
    
    if 0 <= idx <= 3 and base_x <= pos[0] <= base_x + SQUARE_SIZE:
        choices = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        final_move = chess.Move(promotion_move_start, promotion_move_end, promotion=choices[int(idx)])
        handle_move_input(final_move)
        promotion_pending = False
        promotion_move_start = None
        promotion_move_end = None
        return True
    return False

def handle_move_input(move):
    """
    Decides if the move is played immediately (My Turn)
    or stored as a Premove (Opponent's Turn).
    """
    global premove, selected_square
    
    # 1. If it is human(you|me) turn (White):
    if board.turn == chess.WHITE:
        if move in board.legal_moves:
            board.push(move)
            print(f"You played:   {move.uci()}")
            selected_square = None
            premove = None # Clear any old premove
            if not board.is_game_over():
                trigger_engine()
        else:
            print("Illegal move!")
            selected_square = None

    # If it is ENGINE'S turn (Black): premove Logic
    else:
        piece = board.piece_at(move.from_square)
        if piece and piece.color == chess.WHITE:
            print(f"Premove stored: {move.uci()}")
            premove = move
            selected_square = None

def trigger_engine():
    global engine_thread
    if not (engine_thread and engine_thread.is_alive()):
        engine_thread = threading.Thread(target=run_engine_task, args=(board.copy(),), daemon=True)
        engine_thread.start()

def run_engine_task(current_board):
    try:
        best_move = engine.search_best_move(current_board)
        if isinstance(best_move, tuple): best_move = best_move[0]
        if best_move:
            pygame.event.post(pygame.event.Event(ENGINE_MOVE_EVENT, {"move": best_move}))
    except Exception as e:
        print(f"Engine Error: {e}")

# #----------------#
# | Main Game Loop |
# #----------------#
running = True
clock = pygame.time.Clock()

while running:
    mouse_pos = pygame.mouse.get_pos()
    draw_board()
    pygame.display.flip()
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            engine.stop()
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if promotion_pending:
                handle_promotion_click(event.pos)
                continue
            
            if premove:
                premove = None
                selected_square = None
                
            square = get_square_under_mouse(event.pos)
            if square is None: continue

            piece = board.piece_at(square)
            
            # Allow selecting White pieces even if it is Black's turn (for Premove)
            if piece and piece.color == chess.WHITE:
                dragging_square = square
                dragging_piece = piece
                selected_square = square
            
            elif selected_square is not None:
                # Promotion Check logic duplicated for click-click
                is_promo = False
                p = board.piece_at(selected_square)
                if p and p.piece_type == chess.PAWN:
                    if (chess.square_rank(square) == 7):
                        is_promo = True
                
                if is_promo:
                    promotion_pending = True
                    promotion_move_start = selected_square
                    promotion_move_end = square
                    promotion_color = chess.WHITE
                else:
                    move = chess.Move(selected_square, square)
                    handle_move_input(move) 

        elif event.type == pygame.MOUSEBUTTONUP:
            if dragging_square is not None:
                target_square = get_square_under_mouse(event.pos)
                if target_square is not None and target_square != dragging_square:
                    
                    # Drag-Drop Move
                    is_promo = False
                    p = board.piece_at(dragging_square)
                    if p and p.piece_type == chess.PAWN:
                        if (chess.square_rank(target_square) == 7):
                            is_promo = True
                    
                    if is_promo:
                        promotion_pending = True
                        promotion_move_start = dragging_square
                        promotion_move_end = target_square
                        promotion_color = chess.WHITE
                    else:
                        move = chess.Move(dragging_square, target_square)
                        handle_move_input(move)

                dragging_square = None
                dragging_piece = None

        elif event.type == ENGINE_MOVE_EVENT:
            move = event.move
            if move in board.legal_moves:
                board.push(move)
                print(f"Engine plays: {move.uci()}")
                
                # PREMOVE EXECUTION 
                if premove:
                    # Is the premove legal?
                    if premove in board.legal_moves:
                        print(f"Premove EXECUTED: {premove.uci()}")
                        board.push(premove)
                        premove = None
                        if not board.is_game_over():
                            trigger_engine()
                    else:
                        print(f"Premove {premove.uci()} is ILLEGAL now. Cancelled.")
                        premove = None
            
            if board.is_game_over():
                game_over = True
                print("Game Over:", board.result())

pygame.quit()