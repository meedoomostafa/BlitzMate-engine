"""Row of small captured-piece images with material advantage display."""

from typing import List, Dict

import chess
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QPixmap, QFont, QPaintEvent

from gui.helpers import _PIECES, MoveRecord, TXT


class CapturedPiecesWidget(QWidget):
    """Shows pieces captured by one side, sorted by value, with material diff."""

    _VAL = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    _ORDER = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
    _SORT = {
        chess.QUEEN: 0,
        chess.ROOK: 1,
        chess.BISHOP: 2,
        chess.KNIGHT: 3,
        chess.PAWN: 4,
    }

    def __init__(self, capturing_color: chess.Color, parent=None):
        super().__init__(parent)
        self.capturing_color = capturing_color
        self.setFixedHeight(26)
        self._captured: List[int] = []
        self._advantage = 0
        self._mini_cache: Dict[str, QPixmap] = {}

    def refresh(self, records: List[MoveRecord], up_to: int, board: chess.Board):
        """Recalculate captured pieces and material advantage."""
        self._captured = [
            r.captured_piece
            for r in records[:up_to]
            if r.captured_piece is not None
            and r.capturing_color == self.capturing_color
        ]
        self._captured.sort(key=lambda pt: self._SORT.get(pt, 5))
        my_mat = sum(
            self._VAL.get(pt, 0) * len(board.pieces(pt, self.capturing_color))
            for pt in self._ORDER
        )
        opp_mat = sum(
            self._VAL.get(pt, 0) * len(board.pieces(pt, not self.capturing_color))
            for pt in self._ORDER
        )
        self._advantage = my_mat - opp_mat
        self.update()

    def paintEvent(self, event: QPaintEvent):
        if not self._captured:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        x, opp = 4, not self.capturing_color
        for pt in self._captured:
            sym = chess.Piece(pt, opp).symbol()
            if sym not in self._mini_cache:
                pix = _PIECES.get(sym)
                if pix:
                    self._mini_cache[sym] = pix.scaled(
                        20, 20, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
            sc = self._mini_cache.get(sym)
            if sc:
                p.drawPixmap(x, 3, sc)
            x += 17
        if self._advantage > 0:
            p.setFont(QFont("Segoe UI", 10, QFont.Bold))
            p.setPen(QPen(TXT))
            p.drawText(x + 2, 18, f"+{self._advantage}")
        p.end()
