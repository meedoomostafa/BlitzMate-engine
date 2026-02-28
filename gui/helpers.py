"""Shared helpers for the BlitzMate GUI — theme, engine pool, sound, openings."""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import dataclass
from typing import Optional, List, Dict

import chess
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QPixmap

try:
    from PySide6.QtMultimedia import QSoundEffect

    _HAS_SOUND = True
except ImportError:
    _HAS_SOUND = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine.config import CONFIG

# ── Board colors (warm brown wood) ──────────────────────────
BOARD_LIGHT = QColor("#D2B48C")
BOARD_DARK = QColor("#8B6B4A")

# ── Highlight colors ────────────────────────────────────────
HL_SELECTED = QColor(246, 246, 105, 180)
HL_LAST_MOVE = QColor(246, 246, 105, 130)
HL_PREMOVE = QColor(220, 90, 90, 150)
HL_CHECK_INNER = QColor(255, 0, 0, 200)
HL_CHECK_OUTER = QColor(255, 0, 0, 0)
HL_LEGAL_DOT = QColor(0, 0, 0, 40)
HL_LEGAL_CAP = QColor(0, 0, 0, 40)

# ── Panel colors (chess.com dark mode) ──────────────────────
BG_DARKEST = QColor("#1b1a18")
BG_DARK = QColor("#262522")
BG_MID = QColor("#302e2b")
BG_CARD = QColor("#3c3a36")
BG_LIGHT = QColor("#48463f")
BG_HOVER = QColor("#555350")
BG_ROW_ALT = QColor("#2a2926")
BG_ACTIVE = QColor("#4a6e28")

# ── Text colors ─────────────────────────────────────────────
TXT = QColor("#c3c1bf")
TXT_DIM = QColor("#9b9892")
TXT_ACCENT = QColor("#81b64c")
TXT_WHITE = QColor("#ffffff")
TXT_GREEN = QColor("#a3d160")

# ── Eval bar colors ─────────────────────────────────────────
EVAL_W = QColor("#f5f5f5")
EVAL_B = QColor("#403d39")

# ── Board frame ─────────────────────────────────────────────
BOARD_FRAME = QColor("#1b1a18")
MIN_BOARD_PX = 360
ANIM_DURATION_MS = 180

# ── QSS Stylesheet ──────────────────────────────────────────
QSS = """
QMainWindow { background: #262522; }

/* Tabs */
QTabWidget::pane { background: #262522; border: none; }
QTabBar { background: #1b1a18; qproperty-drawBase: 0; }
QTabBar::tab {
    background: #302e2b; color: #9b9892;
    padding: 11px 28px; border: none;
    border-bottom: 3px solid transparent;
    font-size: 13px; font-weight: 600; margin-right: 1px;
}
QTabBar::tab:selected {
    background: #3c3a36; color: #ffffff;
    border-bottom: 3px solid #81b64c;
}
QTabBar::tab:hover:!selected { background: #48463f; color: #c3c1bf; }

/* Primary buttons */
QPushButton {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #8ece52, stop:1 #73a83e);
    color: #ffffff; border: none;
    padding: 10px 22px; border-radius: 6px;
    font-size: 13px; font-weight: bold;
}
QPushButton:hover {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #9ddb62, stop:1 #81b64c);
}
QPushButton:pressed { background: #6a9a3c; }

/* Nav buttons (flat) */
QPushButton#nav_btn {
    background: #3c3a36; color: #c3c1bf;
    padding: 7px 14px; border-radius: 5px;
    font-size: 16px; border: 1px solid #48463f;
}
QPushButton#nav_btn:hover { background: #48463f; border-color: #555350; }
QPushButton#nav_btn:pressed { background: #302e2b; }

/* New game button */
QPushButton#new_game_btn {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #8ece52, stop:1 #73a83e);
    padding: 8px 20px; font-size: 13px; border-radius: 6px;
}
QPushButton#new_game_btn:hover {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #9ddb62, stop:1 #81b64c);
}

/* Scrollbar */
QScrollArea { background: transparent; border: none; }
QScrollBar:vertical {
    background: #262522; width: 7px; margin: 0; border-radius: 3px;
}
QScrollBar::handle:vertical {
    background: #555350; border-radius: 3px; min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #6a6864; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }

/* Timeline slider */
QSlider::groove:horizontal { background: #3c3a36; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal {
    background: #81b64c; width: 16px; height: 16px;
    margin: -5px 0; border-radius: 8px; border: 2px solid #262522;
}
QSlider::handle:horizontal:hover { background: #9ddb62; }
QSlider::sub-page:horizontal { background: #81b64c; border-radius: 3px; }

/* Labels */
QLabel { color: #c3c1bf; }

/* Status bar */
QStatusBar {
    background: #1b1a18; color: #9b9892;
    font-size: 12px; padding: 3px 10px;
    border-top: 1px solid #302e2b;
}

/* Side panel */
QFrame#panel {
    background: #302e2b; border-radius: 10px;
    border: 1px solid #3c3a36;
}

/* Player cards */
QFrame#player_card {
    background: #3c3a36; border-radius: 8px;
    border: 1px solid #48463f; padding: 4px;
}
QFrame#player_card_active {
    background: #3c3a36; border-radius: 8px;
    border: 1px solid #81b64c; padding: 4px;
}

/* Toolbar buttons */
QToolButton {
    background: transparent; color: #c3c1bf;
    border: none; padding: 5px; border-radius: 5px;
}
QToolButton:hover { background: #48463f; }

/* Menu bar */
QMenuBar {
    background: #1b1a18; color: #c3c1bf;
    padding: 3px; font-size: 13px;
    border-bottom: 1px solid #302e2b;
}
QMenuBar::item { padding: 6px 14px; border-radius: 4px; }
QMenuBar::item:selected { background: #3c3a36; }
QMenu {
    background: #302e2b; color: #c3c1bf;
    border: 1px solid #48463f; border-radius: 6px; padding: 4px;
}
QMenu::item { padding: 8px 24px; border-radius: 4px; }
QMenu::item:selected { background: #81b64c; color: white; }
QMenu::separator { height: 1px; background: #48463f; margin: 4px 8px; }
"""

# ── Piece pixmaps (loaded after QApplication init) ──────────
_PIECES: Dict[str, QPixmap] = {}


def load_piece_pixmaps():
    """Load piece images from assets. Must be called after QApplication is created."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    for p in "rnbqkp":
        for color in "bw":
            key = p if color == "b" else p.upper()
            path = os.path.join(base, f"{color}{p.upper()}.png")
            if os.path.exists(path):
                _PIECES[key] = QPixmap(path)


# ── Sound manager ───────────────────────────────────────────
class SoundManager:
    """Plays game sound effects (move, capture, check, etc.)."""

    _SOUNDS = {
        "move": "move.wav",
        "capture": "capture.wav",
        "check": "check.wav",
        "end": "game_end.wav",
        "castle": "castle.wav",
        "notify": "notify.wav",
    }

    def __init__(self):
        self._effects: Dict[str, "QSoundEffect"] = {}
        if not _HAS_SOUND:
            return
        base = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets", "sounds"
        )
        for name, fname in self._SOUNDS.items():
            path = os.path.join(base, fname)
            if os.path.exists(path):
                snd = QSoundEffect()
                snd.setSource(QUrl.fromLocalFile(os.path.abspath(path)))
                snd.setVolume(0.5)
                self._effects[name] = snd

    def play(self, name: str):
        if name in self._effects:
            self._effects[name].play()


# ── Move record ─────────────────────────────────────────────
@dataclass
class MoveRecord:
    """One move in the game history."""

    san: str
    uci: str
    time_ms: int
    is_capture: bool
    is_check: bool
    captured_piece: Optional[int] = None
    capturing_color: Optional[bool] = None
    eval_cp: Optional[int] = None


# ── Engine process pool ─────────────────────────────────────
_worker_engine = None


def _init_worker(depth: int, base_path: str):
    """Initialize a SearchEngine in each worker process."""
    global _worker_engine
    if base_path not in sys.path:
        sys.path.insert(0, base_path)
    from engine.core.search import SearchEngine

    _worker_engine = SearchEngine(depth=depth)


def _run_search(fen: str) -> dict:
    """Run a blocking search and return best move + score."""
    import chess as _chess

    board = _chess.Board(fen)
    best, ponder, score = _worker_engine.search_best_move(board)
    return {
        "best_move": best.uci() if best else None,
        "ponder": ponder.uci() if ponder else None,
        "score": score,
    }


class EnginePool:
    """ProcessPoolExecutor sized to (cpu_count - 1) cores."""

    def __init__(self, depth: int):
        self.depth = depth
        cores = os.cpu_count() or 2
        self.max_workers = max(1, cores - 1)
        self._base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.pool: Optional[ProcessPoolExecutor] = None

    def start(self):
        self.pool = ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=_init_worker,
            initargs=(self.depth, self._base),
        )

    def submit(self, fen: str) -> Optional[Future]:
        if self.pool:
            return self.pool.submit(_run_search, fen)
        return None

    def shutdown(self):
        if self.pool:
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None


# ── Opening detection ────────────────────────────────────────
_OPENINGS = [
    (["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"], "Ruy Lopez"),
    (["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"], "Italian Game"),
    (["e2e4", "e7e5", "g1f3", "b8c6", "d2d4"], "Scotch Game"),
    (["e2e4", "e7e5", "g1f3", "b8c6"], "Four Knights Setup"),
    (["e2e4", "e7e5", "g1f3", "g8f6"], "Petrov Defense"),
    (["e2e4", "e7e5", "g1f3"], "King's Knight Opening"),
    (["e2e4", "e7e5", "f2f4"], "King's Gambit"),
    (["e2e4", "e7e5"], "Open Game"),
    (["e2e4", "c7c5", "g1f3", "d7d6", "d2d4"], "Sicilian Najdorf"),
    (["e2e4", "c7c5", "g1f3", "b8c6", "d2d4"], "Sicilian Open"),
    (["e2e4", "c7c5", "g1f3"], "Open Sicilian"),
    (["e2e4", "c7c5", "b1c3"], "Closed Sicilian"),
    (["e2e4", "c7c5"], "Sicilian Defense"),
    (["e2e4", "e7e6", "d2d4", "d7d5"], "French Defense"),
    (["e2e4", "e7e6"], "French Defense"),
    (["e2e4", "c7c6", "d2d4", "d7d5"], "Caro-Kann Defense"),
    (["e2e4", "c7c6"], "Caro-Kann Defense"),
    (["e2e4", "d7d5", "e4d5", "d8d5"], "Scandinavian Defense"),
    (["e2e4", "d7d5"], "Scandinavian Defense"),
    (["e2e4", "g7g6"], "Modern Defense"),
    (["e2e4", "d7d6"], "Pirc Defense"),
    (["e2e4", "g8f6"], "Alekhine's Defense"),
    (["e2e4"], "King's Pawn"),
    (["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"], "QGD Classical"),
    (["d2d4", "d7d5", "c2c4", "e7e6"], "Queen's Gambit Declined"),
    (["d2d4", "d7d5", "c2c4", "d5c4"], "Queen's Gambit Accepted"),
    (["d2d4", "d7d5", "c2c4", "c7c6"], "Slav Defense"),
    (["d2d4", "d7d5", "c2c4"], "Queen's Gambit"),
    (["d2d4", "d7d5", "g1f3", "g8f6", "c2c4"], "Queen's Gambit"),
    (["d2d4", "d7d5"], "Closed Game"),
    (["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5"], "Grünfeld Defense"),
    (["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"], "King's Indian"),
    (["d2d4", "g8f6", "c2c4", "g7g6"], "King's Indian Defense"),
    (["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"], "Nimzo-Indian"),
    (["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6"], "Queen's Indian"),
    (["d2d4", "g8f6", "c2c4", "e7e6"], "Indian Game"),
    (["d2d4", "g8f6", "c2c4", "c7c5"], "Benoni Defense"),
    (["d2d4", "g8f6", "c2c4"], "Indian Systems"),
    (["d2d4", "g8f6", "c1f4"], "London System"),
    (["d2d4", "g8f6"], "Indian Defense"),
    (["d2d4", "f7f5"], "Dutch Defense"),
    (["d2d4"], "Queen's Pawn"),
    (["c2c4", "e7e5"], "English Opening"),
    (["c2c4", "g8f6"], "English Opening"),
    (["c2c4"], "English Opening"),
    (["g1f3", "d7d5", "g2g3"], "Réti Opening"),
    (["g1f3"], "Réti Opening"),
    (["f2f4"], "Bird's Opening"),
    (["b2b3"], "Larsen's Opening"),
    (["g2g3"], "Benko's Opening"),
]


def detect_opening(board: chess.Board) -> str:
    """Match the current move list against known opening prefixes."""
    ucis = [m.uci() for m in board.move_stack]
    if not ucis:
        return ""
    best_name, best_len = "", 0
    for prefix, name in _OPENINGS:
        if len(prefix) <= len(ucis) and ucis[: len(prefix)] == prefix:
            if len(prefix) > best_len:
                best_name, best_len = name, len(prefix)
    return best_name
