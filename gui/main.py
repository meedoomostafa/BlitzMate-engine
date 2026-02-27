import pygame
import chess
import queue
import sys, os

# Ensure engine import works
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine.config import CONFIG
from engine.core.search import SearchEngine

pygame.init()
pygame.font.init()
pygame.mixer.init()

SQUARE_SIZE = 90
BOARD_SIZE = 8 * SQUARE_SIZE
PANEL_WIDTH = 250
WINDOW_WIDTH = BOARD_SIZE + PANEL_WIDTH
WINDOW_HEIGHT = BOARD_SIZE

COLOR_LIGHT = (238, 238, 210)
COLOR_DARK = (118, 150, 86)
COLOR_HIGHLIGHT = (186, 202, 68, 200)
COLOR_LAST_MOVE = (246, 246, 105, 180)
COLOR_PREMOVE = (200, 100, 100, 180)
COLOR_CHECK = (255, 50, 50)
COLOR_PANEL_BG = (40, 40, 40)
COLOR_TEXT = (240, 240, 240)
COLOR_TEXT_ACCENT = (150, 150, 150)

# Fonts
FONT_COORD = pygame.font.SysFont("Arial", 14, bold=True)
FONT_UI = pygame.font.SysFont("Verdana", 20)
FONT_HISTORY = pygame.font.SysFont("Consolas", 16)


class SoundManager:
    def __init__(self):
        self.sounds = {}
        base_path = os.path.dirname(os.path.abspath(__file__))
        sound_dir = os.path.join(base_path, "assets", "sounds")

        files = {
            "move": "move.wav",
            "capture": "capture.wav",
            "check": "notify.wav",
            "end": "notify.wav",
        }

        for name, file in files.items():
            path = os.path.join(sound_dir, file)
            if os.path.exists(path):
                self.sounds[name] = pygame.mixer.Sound(path)
            else:
                print(f"Warning: Sound file not found: {path}")

    def play(self, name):
        if name in self.sounds:
            self.sounds[name].play()


class ChessGUI:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(f"BlitzMate - {CONFIG.ui.engine_name}")
        self.clock = pygame.time.Clock()

        # Assets
        self.pieces = self.load_pieces()
        self.sound = SoundManager()

        # Game State
        self.board = chess.Board()
        self.engine = SearchEngine(depth=CONFIG.search.depth)
        self.game_over = False
        self.engine_thinking = False

        # UI State
        self.selected_sq = None
        self.dragging_sq = None
        self.premove = None
        self.promotion_state = None  # {start, end, color}

        self.info_depth = None
        self.info_score = None
        self.ponder_move = None
        
        # Thread-safe queue for engine results
        self._engine_queue = queue.Queue()

    def load_pieces(self):
        pieces = {}
        assets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        for p in ["r", "n", "b", "q", "k", "p"]:
            for color in ["b", "w"]:
                key = p if color == "b" else p.upper()
                try:
                    img = pygame.image.load(
                        os.path.join(assets_path, f"{color}{p.upper()}.png")
                    )
                    pieces[key] = pygame.transform.smoothscale(
                        img, (SQUARE_SIZE, SQUARE_SIZE)
                    )
                except:
                    print(f"Missing asset: {key}")
        return pieces

    def get_square_at(self, pos):
        if pos[0] > BOARD_SIZE:
            return None
        col = pos[0] // SQUARE_SIZE
        row = 7 - (pos[1] // SQUARE_SIZE)
        return chess.square(col, row)

    def draw(self):
        self.screen.fill(COLOR_PANEL_BG)
        self.draw_board()
        self.draw_highlights()
        self.draw_pieces()
        self.draw_drag()
        self.draw_coordinates()
        self.draw_side_panel()
        if self.promotion_state:
            self.draw_promotion_menu()
        pygame.display.flip()

    def draw_board(self):
        for row in range(8):
            for col in range(8):
                color = COLOR_LIGHT if (row + col) % 2 == 0 else COLOR_DARK
                pygame.draw.rect(
                    self.screen,
                    color,
                    (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
                )

    def draw_coordinates(self):
        for r in range(8):
            row_sq_color = (0 + (7 - r)) % 2
            text_color = COLOR_DARK if row_sq_color == 0 else COLOR_LIGHT
            lbl = FONT_COORD.render(str(r + 1), True, text_color)
            self.screen.blit(lbl, (2, (7 - r) * SQUARE_SIZE + 2))

        # Draw Files (a-h) on the bottom
        for c in range(8):
            col_sq_color = (c + 0) % 2
            text_color = COLOR_DARK if col_sq_color == 0 else COLOR_LIGHT
            lbl = FONT_COORD.render(chr(ord("a") + c), True, text_color)
            self.screen.blit(lbl, (c * SQUARE_SIZE + SQUARE_SIZE - 12, BOARD_SIZE - 18))

    def draw_highlights(self):
        s = pygame.Surface(
            (SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA
        )  # Surface for transparency

        if self.board.move_stack:
            last = self.board.move_stack[-1]
            s.fill(COLOR_LAST_MOVE)
            self.screen.blit(s, self.sq_to_rect(last.from_square))
            self.screen.blit(s, self.sq_to_rect(last.to_square))

        if self.selected_sq is not None:
            s.fill(COLOR_HIGHLIGHT)
            self.screen.blit(s, self.sq_to_rect(self.selected_sq))

        if self.premove:
            s.fill(COLOR_PREMOVE)
            self.screen.blit(s, self.sq_to_rect(self.premove.from_square))
            self.screen.blit(s, self.sq_to_rect(self.premove.to_square))

        if self.board.is_check():
            king_sq = self.board.king(self.board.turn)
            if king_sq is not None:
                pygame.draw.circle(
                    self.screen,
                    COLOR_CHECK,
                    self.sq_center(king_sq),
                    SQUARE_SIZE // 2,
                    4,
                )

        if self.board.turn == chess.WHITE and (
            self.selected_sq is not None or self.dragging_sq is not None
        ):
            src = self.dragging_sq if self.dragging_sq is not None else self.selected_sq
            for move in self.board.legal_moves:
                if move.from_square == src:
                    if self.board.is_capture(move):
                        pygame.draw.circle(
                            self.screen,
                            (100, 100, 100, 100),
                            self.sq_center(move.to_square),
                            SQUARE_SIZE // 2,
                            5,
                        )
                    else:
                        pygame.draw.circle(
                            self.screen,
                            (80, 80, 80, 100),
                            self.sq_center(move.to_square),
                            SQUARE_SIZE // 6,
                        )

    def draw_pieces(self):
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)

            if self.premove:
                if sq == self.premove.from_square:
                    piece = None
                elif sq == self.premove.to_square:
                    piece = self.board.piece_at(self.premove.from_square)

            if piece and sq != self.dragging_sq:
                rect = self.sq_to_rect(sq)
                self.screen.blit(self.pieces[piece.symbol()], rect)

    def draw_drag(self):
        if self.dragging_sq is not None:
            piece = self.board.piece_at(self.dragging_sq)
            if piece:
                pos = pygame.mouse.get_pos()
                img = self.pieces[piece.symbol()]
                self.screen.blit(
                    img, (pos[0] - SQUARE_SIZE // 2, pos[1] - SQUARE_SIZE // 2)
                )

    def draw_side_panel(self):
        x = BOARD_SIZE + 10
        y = 20

        turn_text = (
            "White to Move" if self.board.turn == chess.WHITE else "Black (Engine)"
        )
        if self.game_over:
            turn_text = "Game Over"

        label = FONT_UI.render(turn_text, True, COLOR_TEXT)
        self.screen.blit(label, (x, y))
        y += 40

        hist_label = FONT_UI.render("Move History:", True, COLOR_TEXT_ACCENT)
        self.screen.blit(hist_label, (x, y))
        y += 30

        history = [m.uci() for m in self.board.move_stack][-16:]
        for i in range(0, len(history), 2):
            move_num = (len(self.board.move_stack) - len(history) + i) // 2 + 1
            w_move = history[i]
            b_move = history[i + 1] if i + 1 < len(history) else ""
            line = f"{move_num}. {w_move}  {b_move}"
            txt = FONT_HISTORY.render(line, True, COLOR_TEXT)
            self.screen.blit(txt, (x + 10, y))
            y += 20

        if self.game_over:
            y += 40
            res = self.board.result()
            res_lbl = FONT_UI.render(f"Result: {res}", True, (100, 200, 100))
            self.screen.blit(res_lbl, (x, y))

    def draw_promotion_menu(self):
        start, end, color = self.promotion_state

        col = chess.square_file(end)
        row = 0 if chess.square_rank(end) == 7 else 4

        menu_rect = pygame.Rect(
            col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE * 4
        )

        pygame.draw.rect(self.screen, (255, 255, 255), menu_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), menu_rect, 2)

        opts = ["q", "r", "b", "n"] if color == chess.BLACK else ["Q", "R", "B", "N"]
        for i, p_char in enumerate(opts):
            self.screen.blit(
                self.pieces[p_char], (menu_rect.x, menu_rect.y + i * SQUARE_SIZE)
            )

    # #-----------------------#
    # | Helper Functions Part |
    # #-----------------------#

    def sq_to_rect(self, sq):
        col = chess.square_file(sq)
        row = 7 - chess.square_rank(sq)
        return (col * SQUARE_SIZE, row * SQUARE_SIZE)

    def sq_center(self, sq):
        rect = self.sq_to_rect(sq)
        return (rect[0] + SQUARE_SIZE // 2, rect[1] + SQUARE_SIZE // 2)

    def play_sound_for_move(self, move):
        if self.board.is_checkmate():
            self.sound.play("end")
        elif self.board.is_check():
            self.sound.play("check")
        elif self.board.is_capture(move):
            self.sound.play("capture")
        else:
            self.sound.play("move")

    def push_move(self, move):
        if move in self.board.legal_moves:
            is_cap = self.board.is_capture(move)
            self.board.push(move)

            if self.board.is_game_over():
                self.sound.play("end")
            elif self.board.is_check():
                self.sound.play("check")
            elif is_cap:
                self.sound.play("capture")
            else:
                self.sound.play("move")

            return True
        return False

    def handle_human_move(self, move):
        if self.push_move(move):
            print(f"You played: {move.uci()}")
            self.premove = None
            if not self.board.is_game_over():
                self.trigger_engine()
        else:
            print("Illegal Move")

    def handle_premove(self, move):
        p = self.board.piece_at(move.from_square)
        if p and p.color == chess.WHITE:
            self.premove = move
            print(f"Premove stored: {move.uci()}")

    def trigger_engine(self):
        if not self.engine_thinking and not self.game_over:
            self.engine_thinking = True
            self.engine.start_search(self.board.copy(), callback=self.engine_callback)

    def engine_callback(self, best_move, ponder_move, depth, score):
        self._engine_queue.put({
            "best_move": best_move,
            "ponder_move": ponder_move,
            "depth": depth,
            "score": score
        })

    def _handle_engine_event(self, data):
        self.info_depth = data['depth']
        self.ponder_move = data['ponder_move']

        sc = data['score']
        if isinstance(sc, int):
            self.info_score = f"{sc/100:.2f}"
        else:
            self.info_score = str(sc)

        # Check if search is finished
        if data['depth'] <= 0:
            self.engine_thinking = False
            best_move = data['best_move']

            if best_move and self.push_move(best_move):
                print(f"Engine plays: {best_move.uci()}")

                # Handle premove
                if self.premove:
                    if self.push_move(self.premove):
                        print(f"Premove executed: {self.premove.uci()}")
                        self.premove = None
                        if not self.board.is_game_over():
                            self.trigger_engine()
                    else:
                        print("Premove illegal, cancelled.")
                        self.premove = None

            if self.board.is_game_over():
                self.game_over = True
                print("Game Over")

    def run(self):
        running = True
        while running:
            self.clock.tick(60)
            
            # Poll engine results from thread-safe queue
            while not self._engine_queue.empty():
                try:
                    data = self._engine_queue.get_nowait()
                    self._handle_engine_event(data)
                except queue.Empty:
                    break
            
            self.draw()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.engine.stop()
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()

                    if self.promotion_state:
                        start, end, color = self.promotion_state
                        col = chess.square_file(end)
                        base_y = 0 if chess.square_rank(end) == 7 else 4 * SQUARE_SIZE

                        if col * SQUARE_SIZE <= pos[0] <= (col + 1) * SQUARE_SIZE:
                            idx = (pos[1] - base_y) // SQUARE_SIZE
                            if 0 <= idx <= 3:
                                opts = [
                                    chess.QUEEN,
                                    chess.ROOK,
                                    chess.BISHOP,
                                    chess.KNIGHT,
                                ]
                                move = chess.Move(start, end, promotion=opts[int(idx)])
                                self.promotion_state = None
                                self.handle_human_move(move)
                        continue

                    sq = self.get_square_at(pos)
                    if sq is not None:
                        piece = self.board.piece_at(sq)

                        if piece and piece.color == chess.WHITE:
                            self.dragging_sq = sq
                            self.selected_sq = sq
                            if self.board.turn == chess.WHITE:
                                self.premove = None

                        elif self.selected_sq is not None:
                            selected_piece = self.board.piece_at(self.selected_sq)
                            if (
                                selected_piece
                                and selected_piece.piece_type == chess.PAWN
                                and (
                                    chess.square_rank(sq) == 7
                                    or chess.square_rank(sq) == 0
                                )
                            ):
                                pseudo = chess.Move(
                                    self.selected_sq, sq, promotion=chess.QUEEN
                                )
                                if pseudo in self.board.legal_moves or (
                                    self.board.turn == chess.BLACK
                                    and self.board.piece_at(self.selected_sq).color
                                    == chess.WHITE
                                ):
                                    self.promotion_state = (
                                        self.selected_sq,
                                        sq,
                                        chess.WHITE,
                                    )
                                    self.dragging_sq = None
                                    continue

                            move = chess.Move(self.selected_sq, sq)
                            if self.board.turn == chess.WHITE:
                                self.handle_human_move(move)
                            else:
                                self.handle_premove(move)

                            self.selected_sq = None
                            self.dragging_sq = None

                elif event.type == pygame.MOUSEBUTTONUP:
                    if self.dragging_sq is not None:
                        target = self.get_square_at(pygame.mouse.get_pos())
                        if target is not None and target != self.dragging_sq:

                            is_pawn = (
                                self.board.piece_at(self.dragging_sq).piece_type
                                == chess.PAWN
                            )
                            is_promo_rank = chess.square_rank(target) in [0, 7]

                            if is_pawn and is_promo_rank:
                                self.promotion_state = (
                                    self.dragging_sq,
                                    target,
                                    chess.WHITE,
                                )
                            else:
                                move = chess.Move(self.dragging_sq, target)
                                if self.board.turn == chess.WHITE:
                                    self.handle_human_move(move)
                                else:
                                    self.handle_premove(move)

                        self.dragging_sq = None

        pygame.quit()


if __name__ == "__main__":
    gui = ChessGUI()
    gui.run()
