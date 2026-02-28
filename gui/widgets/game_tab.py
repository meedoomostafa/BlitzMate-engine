"""GameTab — independent game session with board, history, eval, and timeline."""

import time
import random
from typing import Optional, List
from concurrent.futures import Future

import chess

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QKeyEvent

from gui.helpers import (
    CONFIG,
    EnginePool,
    SoundManager,
    MoveRecord,
    detect_opening,
)
from gui.widgets.eval_bar import EvalBarWidget
from gui.widgets.eval_graph import EvalGraphWidget
from gui.widgets.captured import CapturedPiecesWidget
from gui.widgets.board import ChessBoardWidget
from gui.widgets.history import MoveHistoryWidget
from gui.widgets.timeline import TimelineWidget


class GameTab(QWidget):
    """Independent game session with board, history, eval, and timeline."""

    title_changed = Signal(str)

    def __init__(self, tab_id: int, pool: EnginePool, sound: SoundManager, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.tab_id = tab_id
        self.pool = pool
        self.sound = sound

        self.board = chess.Board()
        self._records: List[MoveRecord] = []
        self.game_over = False
        self.engine_thinking = False
        self._future: Optional[Future] = None
        self._turn_t0 = time.time()
        self._eval_cp = 0
        self.human_color: bool = chess.WHITE
        self._chosen_color: Optional[bool] = chess.WHITE

        self._view: Optional[int] = None
        self.flipped = False
        self._last_active_white: Optional[bool] = None

        self._build_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(50)

        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._tick_anim)
        self._anim_timer.start(400)

    # ── UI construction ─────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 4)
        root.setSpacing(0)

        body = QHBoxLayout()
        body.setSpacing(0)

        self.eval_bar = EvalBarWidget()
        body.addWidget(self.eval_bar)
        body.addSpacing(4)

        bcol = QVBoxLayout()
        bcol.setSpacing(0)
        self._build_player_card_top(bcol)
        self._build_board(bcol)
        self._build_player_card_bot(bcol)
        body.addLayout(bcol, stretch=3)
        body.addSpacing(6)

        self._build_panel(body)
        root.addLayout(body, stretch=1)

        self.timeline = TimelineWidget()
        self.timeline.position_changed.connect(self._nav)
        root.addWidget(self.timeline)

    def _build_player_card_top(self, parent):
        """Top player card (engine)."""
        self.card_top = QFrame()
        self.card_top.setObjectName("player_card")
        lay = QHBoxLayout(self.card_top)
        lay.setContentsMargins(10, 8, 12, 8)
        lay.setSpacing(10)

        self._avatar_top = QLabel("♚")
        self._avatar_top.setFixedSize(36, 36)
        self._avatar_top.setAlignment(Qt.AlignCenter)
        self._avatar_top.setStyleSheet(
            "background:#555350;color:#c3c1bf;border-radius:18px;font-size:20px;"
        )
        lay.addWidget(self._avatar_top)

        col = QVBoxLayout()
        col.setSpacing(1)
        self.lbl_top = QLabel("Black · Engine")
        self.lbl_top.setStyleSheet(
            "color:#c3c1bf;font-size:14px;font-weight:bold;padding:0;"
        )
        col.addWidget(self.lbl_top)
        self.cap_top = CapturedPiecesWidget(chess.BLACK)
        col.addWidget(self.cap_top)
        lay.addLayout(col, stretch=1)

        self._clock_top = QLabel("0:00")
        self._clock_top.setStyleSheet(
            "background:#262522;color:#c3c1bf;font-size:18px;font-weight:bold;"
            "padding:4px 12px;border-radius:6px;font-family:'Consolas','Courier New',monospace;"
        )
        self._clock_top.setAlignment(Qt.AlignCenter)
        self._clock_top.setFixedWidth(80)
        lay.addWidget(self._clock_top)
        parent.addWidget(self.card_top)

    def _build_board(self, parent):
        self.bw = ChessBoardWidget()
        self.bw.board = self.board
        self.bw.human_color = self.human_color
        self.bw.move_made.connect(self._human_move)
        self.bw.premove_set.connect(self._set_premove)
        parent.addWidget(self.bw, stretch=1)

    def _build_player_card_bot(self, parent):
        """Bottom player card (human)."""
        self.card_bot = QFrame()
        self.card_bot.setObjectName("player_card_active")
        lay = QHBoxLayout(self.card_bot)
        lay.setContentsMargins(10, 8, 12, 8)
        lay.setSpacing(10)

        self._avatar_bot = QLabel("♔")
        self._avatar_bot.setFixedSize(36, 36)
        self._avatar_bot.setAlignment(Qt.AlignCenter)
        self._avatar_bot.setStyleSheet(
            "background:#81b64c;color:#ffffff;border-radius:18px;font-size:20px;"
        )
        lay.addWidget(self._avatar_bot)

        col = QVBoxLayout()
        col.setSpacing(1)
        self.lbl_bot = QLabel("White · You")
        self.lbl_bot.setStyleSheet(
            "color:#ffffff;font-size:14px;font-weight:bold;padding:0;"
        )
        col.addWidget(self.lbl_bot)
        self.cap_bot = CapturedPiecesWidget(chess.WHITE)
        col.addWidget(self.cap_bot)
        lay.addLayout(col, stretch=1)

        self._clock_bot = QLabel("0:00")
        self._clock_bot.setStyleSheet(
            "background:#262522;color:#ffffff;font-size:18px;font-weight:bold;"
            "padding:4px 12px;border-radius:6px;font-family:'Consolas','Courier New',monospace;"
        )
        self._clock_bot.setAlignment(Qt.AlignCenter)
        self._clock_bot.setFixedWidth(80)
        lay.addWidget(self._clock_bot)
        parent.addWidget(self.card_bot)

    def _build_panel(self, parent):
        """Right-side panel: status, opening, eval graph, move history, action buttons, color chooser."""
        panel = QFrame()
        panel.setObjectName("panel")
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(16, 14, 16, 14)
        pl.setSpacing(8)

        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        icon = QLabel("♞")
        icon.setStyleSheet("color:#81b64c;font-size:22px;font-weight:bold;")
        title_row.addWidget(icon)
        title = QLabel(CONFIG.ui.engine_name)
        title.setStyleSheet("color:#ffffff;font-size:16px;font-weight:bold;")
        title_row.addWidget(title)
        title_row.addStretch()
        pl.addLayout(title_row)

        # Status badge
        status_frame = QFrame()
        status_frame.setStyleSheet("background:#3c3a36;border-radius:8px;")
        sf = QHBoxLayout(status_frame)
        sf.setContentsMargins(12, 8, 12, 8)
        self._status_dot = QLabel("●")
        self._status_dot.setStyleSheet("color:#81b64c;font-size:12px;")
        sf.addWidget(self._status_dot)
        self.lbl_status = QLabel("White to Move")
        self.lbl_status.setStyleSheet("color:#fff;font-size:14px;font-weight:bold;")
        sf.addWidget(self.lbl_status)
        sf.addStretch()
        self._thinking_lbl = QLabel()
        self._thinking_lbl.setStyleSheet("color:#f0c15c;font-size:12px;")
        self._thinking_lbl.hide()
        sf.addWidget(self._thinking_lbl)
        pl.addWidget(status_frame)

        self._opening_lbl = QLabel("Starting Position")
        self._opening_lbl.setStyleSheet(
            "color:#9b9892;font-size:11px;font-style:italic;padding:2px 0;"
        )
        pl.addWidget(self._opening_lbl)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background:#48463f;")
        pl.addWidget(sep)

        self.eval_graph = EvalGraphWidget()
        pl.addWidget(self.eval_graph)

        hdr = QLabel("MOVES")
        hdr.setStyleSheet(
            "color:#9b9892;font-size:10px;letter-spacing:2px;font-weight:bold;margin-top:2px;"
        )
        pl.addWidget(hdr)

        self.hist = MoveHistoryWidget()
        self.hist.move_clicked.connect(self._nav)
        pl.addWidget(self.hist, stretch=1)

        self.lbl_result = QLabel("")
        self.lbl_result.setStyleSheet(
            "background:#4a6e28;color:#ffffff;font-size:15px;font-weight:bold;"
            "padding:8px;border-radius:6px;"
        )
        self.lbl_result.setAlignment(Qt.AlignCenter)
        self.lbl_result.hide()
        pl.addWidget(self.lbl_result)

        # Undo / Resign buttons
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        undo = QPushButton("↩  Undo")
        undo.setObjectName("nav_btn")
        undo.clicked.connect(self._undo)
        undo.setFixedHeight(36)
        row1.addWidget(undo, stretch=1)
        resign = QPushButton("⚑  Resign")
        resign.setObjectName("nav_btn")
        resign.setStyleSheet("""
            QPushButton#nav_btn { background:#4a2020; color:#e07a5f; border:1px solid #6a3030; }
            QPushButton#nav_btn:hover { background:#5a2828; border-color:#7a3838; }
        """)
        resign.clicked.connect(self._resign)
        resign.setFixedHeight(36)
        row1.addWidget(resign, stretch=1)
        pl.addLayout(row1)

        # Flip / New Game buttons
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        flip = QPushButton("⟲  Flip Board")
        flip.setObjectName("nav_btn")
        flip.clicked.connect(self._flip)
        flip.setFixedHeight(36)
        row2.addWidget(flip, stretch=1)
        ng = QPushButton("▶  New Game")
        ng.setObjectName("new_game_btn")
        ng.clicked.connect(self._reset)
        ng.setFixedHeight(36)
        row2.addWidget(ng, stretch=1)
        pl.addLayout(row2)

        # Color chooser
        color_row = QHBoxLayout()
        color_row.setSpacing(4)
        color_label = QLabel("Play as:")
        color_label.setStyleSheet("color:#9b9892;font-size:12px;padding:0;")
        color_row.addWidget(color_label)

        self._color_white = QPushButton("♔ White")
        self._color_random = QPushButton("🎲 Random")
        self._color_black = QPushButton("♚ Black")
        COLOR_BTN_CSS = """
            QPushButton { background:#3a3835;color:#9b9892;border:1px solid #4a4845;border-radius:4px;font-size:12px;padding:2px 8px; }
            QPushButton:checked { background:#81b64c;color:#ffffff;border-color:#6fa33d; }
            QPushButton:hover { background:#4a4845; }
            QPushButton:checked:hover { background:#91c65c; }
        """
        for btn in (self._color_white, self._color_random, self._color_black):
            btn.setCheckable(True)
            btn.setFixedHeight(30)
            btn.setStyleSheet(COLOR_BTN_CSS)
            color_row.addWidget(btn)
        self._color_white.setChecked(True)
        self._color_white.clicked.connect(lambda: self._select_color(chess.WHITE))
        self._color_random.clicked.connect(lambda: self._select_color(None))
        self._color_black.clicked.connect(lambda: self._select_color(chess.BLACK))
        color_row.addStretch()
        pl.addLayout(color_row)

        panel.setMinimumWidth(280)
        panel.setMaximumWidth(440)
        parent.addWidget(panel, stretch=1)

    # ── Game logic ──────────────────────────────────────────

    def _human_move(self, mv: chess.Move):
        """Validate and apply a human move."""
        if self.game_over or self.board.turn != self.human_color:
            return
        if self._view is not None:
            return
        if mv not in self.board.legal_moves:
            return
        dt = int((time.time() - self._turn_t0) * 1000)
        self._push(mv, dt)
        if not self.game_over:
            self._start_engine()

    def _set_premove(self, mv: chess.Move):
        """Store a premove (only when viewing the latest position)."""
        if self._view is not None:
            return
        self.bw.premove = mv
        self.bw.update()

    def _push(self, mv: chess.Move, ms: int = 0):
        """Execute a move: record it, animate, play sound, update state."""
        is_cap = self.board.is_capture(mv)
        san = self.board.san(mv)
        gives_check = self.board.gives_check(mv)
        is_castle = self.board.is_castling(mv)

        captured_pt, cap_color = None, None
        if is_cap:
            cap_color = self.board.turn
            if self.board.is_en_passant(mv):
                captured_pt = chess.PAWN
            else:
                cp = self.board.piece_at(mv.to_square)
                if cp:
                    captured_pt = cp.piece_type

        self.bw.animate_move(mv)
        self.board.push(mv)
        self._records.append(
            MoveRecord(
                san=san,
                uci=mv.uci(),
                time_ms=ms,
                is_capture=is_cap,
                is_check=gives_check,
                captured_piece=captured_pt,
                capturing_color=cap_color,
                eval_cp=self._eval_cp,
            )
        )
        self._turn_t0 = time.time()

        if self.board.is_game_over():
            self.sound.play("end")
            self.game_over = True
            self.lbl_result.setText(f"Game Over — {self.board.result()}")
            self.lbl_result.show()
            self.lbl_status.setText("Game Over")
        elif self.board.is_check():
            self.sound.play("check")
        elif is_castle:
            self.sound.play("castle")
        elif is_cap:
            self.sound.play("capture")
        else:
            self.sound.play("move")

        if not self.game_over:
            self.lbl_status.setText(
                "Your Move"
                if self.board.turn == self.human_color
                else "Engine thinking…"
            )

        self._sync()

    def _start_engine(self):
        """Submit FEN to the engine process pool."""
        if self.engine_thinking or self.game_over:
            return
        future = self.pool.submit(self.board.fen())
        if future is None:
            self.lbl_status.setText("Engine unavailable")
            return
        self.engine_thinking = True
        self.lbl_status.setText("Engine thinking…")
        self._future = future

    def _poll(self):
        """Check if the engine future has completed."""
        if self._future and self._future.done():
            try:
                self._on_engine(self._future.result())
            except Exception as exc:
                import traceback

                traceback.print_exc()
                self.lbl_status.setText(f"Engine error: {exc}")
                self.sound.play("notify")
            self._future = None
            self.engine_thinking = False
            self._sync()

    def _tick_anim(self):
        """Refresh thinking dots animation (lightweight, avoids full _sync)."""
        if self.engine_thinking:
            self._thinking_lbl.setText("●" * (1 + int(time.time() * 2) % 3))
            self._thinking_lbl.show()

    def _on_engine(self, res: dict):
        """Process engine result: update eval, apply move, handle premove."""
        sc = res.get("score", 0)
        # Normalize to white's perspective for storage, graph, and bar
        sign = 1 if self.human_color == chess.WHITE else -1
        self._eval_cp = -sign * sc if isinstance(sc, int) else 0
        self.eval_bar.set_eval(self._eval_cp)

        uci = res.get("best_move")
        if not uci:
            return
        mv = chess.Move.from_uci(uci)
        if mv not in self.board.legal_moves:
            return
        self._push(mv, int((time.time() - self._turn_t0) * 1000))

        # Apply premove if still legal after engine move
        pm = self.bw.premove
        if pm and not self.game_over:
            self.bw.premove = None
            if pm in self.board.legal_moves:
                self._turn_t0 = time.time()
                self._push(pm, 0)
                if not self.game_over:
                    self._start_engine()
            else:
                self.sound.play("notify")
                self.bw.update()

    # ── View / navigation ───────────────────────────────────

    def _display_board(self) -> chess.Board:
        """Return board state at the current view position."""
        if self._view is None or self._view >= len(self.board.move_stack):
            return self.board
        b = self.board.copy()
        while len(b.move_stack) > self._view:
            b.pop()
        return b

    def _nav(self, idx: int):
        """Navigate to a specific move index in history."""
        total = len(self.board.move_stack)
        idx = max(0, min(total, idx))
        self._view = None if idx >= total else idx
        # Clear premove ghost when browsing past positions
        if self._view is not None:
            self.bw.premove = None
        self._sync()

    def _sync(self):
        """Synchronize all UI widgets with the current game state."""
        db = self._display_board()
        ci = self._view if self._view is not None else len(self.board.move_stack)

        self.bw.board = db
        self.bw.game_over = self.game_over
        self.bw.engine_thinking = self.engine_thinking
        self.bw.flipped = self.flipped
        self.bw.update()

        self.hist.set_data(self._records, ci)
        self.timeline.sync(len(self.board.move_stack), ci)
        self.cap_top.refresh(self._records, ci, db)
        self.cap_bot.refresh(self._records, ci, db)

        # Update active player card styling (cached to avoid redundant QSS re-parsing)
        human_active = db.turn == self.human_color and not self.game_over
        active_key = (human_active, self.game_over)
        if active_key != self._last_active_white:
            self._last_active_white = active_key
            self.card_bot.setObjectName(
                "player_card_active" if human_active else "player_card"
            )
            self.card_top.setObjectName(
                "player_card_active"
                if not human_active and not self.game_over
                else "player_card"
            )
            for w in (self.card_bot, self.card_top):
                w.style().unpolish(w)
                w.style().polish(w)

            ACTIVE_CSS = (
                "background:#81b64c;color:#ffffff;font-size:18px;font-weight:bold;"
                "padding:4px 12px;border-radius:6px;font-family:'Consolas','Courier New',monospace;"
            )
            INACTIVE_CSS = (
                "background:#262522;color:#9b9892;font-size:18px;font-weight:bold;"
                "padding:4px 12px;border-radius:6px;font-family:'Consolas','Courier New',monospace;"
            )
            if self.game_over:
                self._clock_bot.setStyleSheet(INACTIVE_CSS)
                self._clock_top.setStyleSheet(INACTIVE_CSS)
            elif human_active:
                self._clock_bot.setStyleSheet(ACTIVE_CSS)
                self._clock_top.setStyleSheet(INACTIVE_CSS)
            else:
                self._clock_bot.setStyleSheet(INACTIVE_CSS)
                self._clock_top.setStyleSheet(ACTIVE_CSS)

        # Status dot + thinking animation
        if self.game_over:
            self._status_dot.setStyleSheet("color:#e07a5f;font-size:12px;")
            self._thinking_lbl.hide()
        elif human_active:
            self._status_dot.setStyleSheet("color:#81b64c;font-size:12px;")
            self._thinking_lbl.hide()
        else:
            self._status_dot.setStyleSheet("color:#f0c15c;font-size:12px;")
            if self.engine_thinking:
                self._thinking_lbl.setText("●" * (1 + int(time.time() * 2) % 3))
                self._thinking_lbl.show()
            else:
                self._thinking_lbl.hide()

        # Clock totals
        white_ms = sum(
            r.time_ms for i, r in enumerate(self._records[:ci]) if i % 2 == 0
        )
        black_ms = sum(
            r.time_ms for i, r in enumerate(self._records[:ci]) if i % 2 == 1
        )
        human_ms = white_ms if self.human_color == chess.WHITE else black_ms
        engine_ms = black_ms if self.human_color == chess.WHITE else white_ms
        self._clock_bot.setText(self._fmt_clock(human_ms))
        self._clock_top.setText(self._fmt_clock(engine_ms))

        # Opening detection
        opening = detect_opening(db)
        if opening:
            self._opening_lbl.setText(f"📖  {opening}")
        elif len(db.move_stack) > 10:
            self._opening_lbl.setText("Middlegame")
        else:
            self._opening_lbl.setText("Starting Position")

        # Eval graph
        evals = [r.eval_cp for r in self._records[:ci] if r.eval_cp is not None]
        self.eval_graph.set_evals(evals)

        self.title_changed.emit(
            f"Game {self.tab_id} · Move {len(self.board.move_stack) // 2 + 1}"
        )

    @staticmethod
    def _fmt_clock(ms: int) -> str:
        s = ms // 1000
        return f"{s // 60}:{s % 60:02d}"

    # ── Actions ─────────────────────────────────────────────

    def _flip(self):
        self.flipped = not self.flipped
        self._sync()

    def _undo(self):
        """Take back the last two moves (human + engine)."""
        if self.game_over or self.engine_thinking or len(self.board.move_stack) < 2:
            return
        self.board.pop()
        self.board.pop()
        if len(self._records) >= 2:
            self._records.pop()
            self._records.pop()
        self._view = None
        self.lbl_status.setText("Your Move")
        self._sync()

    def _resign(self):
        if self.game_over:
            return
        if self._future:
            self._future.cancel()
        self.game_over = True
        self.engine_thinking = False
        self._future = None
        self.sound.play("end")
        res = "0-1" if self.human_color == chess.WHITE else "1-0"
        self.lbl_result.setText(f"You resigned — {res}")
        self.lbl_result.show()
        self.lbl_status.setText("Game Over")
        self._sync()

    def _select_color(self, color: Optional[bool]):
        """Update color chooser toggle state."""
        self._chosen_color = color
        self._color_white.setChecked(color == chess.WHITE)
        self._color_random.setChecked(color is None)
        self._color_black.setChecked(color == chess.BLACK)

    def _reset(self):
        """Start a new game with the chosen color."""
        if self._future:
            self._future.cancel()
        self.board = chess.Board()
        self._records.clear()
        self.game_over = False
        self.engine_thinking = False
        self._future = None
        self._view = None
        self._eval_cp = 0
        self._turn_t0 = time.time()

        chosen = getattr(self, "_chosen_color", chess.WHITE)
        if chosen is None:
            chosen = random.choice([chess.WHITE, chess.BLACK])
        self.human_color = chosen

        self.bw.board = self.board
        self.bw.human_color = self.human_color
        self.bw.selected_sq = None
        self.bw.premove = None
        self.bw._promo = None
        self.bw._anim_move = None
        self.eval_bar.set_eval(0)
        self.eval_graph.set_evals([])
        self._opening_lbl.setText("Starting Position")
        self.lbl_result.hide()

        self.flipped = self.human_color == chess.BLACK
        self.lbl_bot.setText(
            "White · You" if self.human_color == chess.WHITE else "Black · You"
        )
        self.lbl_top.setText(
            "Black · Engine" if self.human_color == chess.WHITE else "White · Engine"
        )
        self.cap_bot.capturing_color = self.human_color
        self.cap_bot._mini_cache.clear()
        self.cap_top.capturing_color = not self.human_color
        self.cap_top._mini_cache.clear()

        self._last_active_white = None
        self.lbl_status.setText(
            "Your Move" if self.human_color == chess.WHITE else "Engine thinking…"
        )
        self._sync()

        if self.human_color == chess.BLACK:
            self._start_engine()

    def dispose(self):
        """Stop timers and cancel pending engine work (called on tab close)."""
        self._timer.stop()
        self._anim_timer.stop()
        self.bw._anim_timer.stop()
        self.eval_bar._timer.stop()
        if self._future:
            self._future.cancel()
            self._future = None
        self.engine_thinking = False

    def keyPressEvent(self, ev: QKeyEvent):
        cur = self._view if self._view is not None else len(self.board.move_stack)
        if ev.key() == Qt.Key_Left:
            self._nav(cur - 1)
        elif ev.key() == Qt.Key_Right:
            self._nav(cur + 1)
        elif ev.key() == Qt.Key_Home:
            self._nav(0)
        elif ev.key() == Qt.Key_End:
            self._nav(len(self.board.move_stack))
        elif ev.key() == Qt.Key_F:
            self._flip()
        else:
            super().keyPressEvent(ev)
