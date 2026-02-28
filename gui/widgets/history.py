"""Custom-painted scrollable move list with timestamps and zebra striping."""

from typing import List

from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Qt, QRect, QPoint, Signal
from PySide6.QtGui import (
    QPainter,
    QBrush,
    QPen,
    QFont,
    QPaintEvent,
    QMouseEvent,
    QWheelEvent,
)

from gui.helpers import (
    MoveRecord,
    BG_ROW_ALT,
    BG_HOVER,
    BG_ACTIVE,
    TXT,
    TXT_DIM,
    TXT_WHITE,
)


class MoveHistoryWidget(QWidget):
    """Two-column move list with active move highlight and click-to-navigate."""

    move_clicked = Signal(int)
    ROW_H = 34

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self._moves: List[MoveRecord] = []
        self._cur = 0
        self._hover = -1
        self._scroll = 0

    def set_data(self, moves: List[MoveRecord], cur_index: int):
        self._moves = moves
        self._cur = cur_index
        self._clamp_scroll()
        self._auto_scroll(cur_index)
        self.update()

    def _auto_scroll(self, idx: int):
        """Ensure the current move row is visible."""
        row = max(0, (idx - 1)) // 2
        ry = row * self.ROW_H
        if ry < self._scroll:
            self._scroll = ry
        elif ry + self.ROW_H > self._scroll + self.height():
            self._scroll = ry + self.ROW_H - self.height()
        self._clamp_scroll()

    def _clamp_scroll(self):
        total = ((len(self._moves) + 1) // 2) * self.ROW_H
        self._scroll = max(0, min(self._scroll, total - self.height()))

    @staticmethod
    def _fmt_time(ms: int) -> str:
        if ms <= 0:
            return ""
        s = ms / 1000
        return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}:{s % 60:04.1f}"

    def paintEvent(self, event: QPaintEvent):
        pr = QPainter(self)
        pr.setRenderHint(QPainter.Antialiasing)
        w, rh = self.width(), self.ROW_H
        num_w = 38
        col_w = (w - num_w) // 2

        fnt_num = QFont("Segoe UI", 11)
        fnt_move = QFont("Segoe UI", 12, QFont.DemiBold)
        fnt_time = QFont("Segoe UI", 9)

        total_rows = (len(self._moves) + 1) // 2
        v0 = max(0, self._scroll // rh)
        v1 = min(total_rows, (self._scroll + self.height()) // rh + 2)

        for row in range(v0, v1):
            y = row * rh - self._scroll
            wi, bi = row * 2, row * 2 + 1

            if row % 2 == 1:
                pr.fillRect(0, y, w, rh, QBrush(BG_ROW_ALT))
            if self._hover in (wi, bi):
                pr.fillRect(0, y, w, rh, QBrush(BG_HOVER))

            pr.setFont(fnt_num)
            pr.setPen(QPen(TXT_DIM))
            pr.drawText(
                QRect(4, y, num_w - 4, rh),
                Qt.AlignVCenter | Qt.AlignRight,
                f"{row + 1}.",
            )

            for side, idx in enumerate((wi, bi)):
                if idx >= len(self._moves):
                    continue
                m = self._moves[idx]
                x = num_w + 6 + side * col_w
                is_cur = idx == self._cur - 1

                if is_cur:
                    pr.setPen(Qt.NoPen)
                    pr.setBrush(QBrush(BG_ACTIVE))
                    pr.drawRoundedRect(x - 4, y + 3, col_w - 2, rh - 6, 5, 5)

                pr.setFont(fnt_move)
                pr.setPen(QPen(TXT_WHITE if is_cur else TXT))
                pr.drawText(QRect(x, y, col_w - 52, rh), Qt.AlignVCenter, m.san)

                ts = self._fmt_time(m.time_ms)
                if ts:
                    pr.setFont(fnt_time)
                    pr.setPen(QPen(TXT_DIM))
                    pr.drawText(
                        QRect(x + col_w - 56, y, 48, rh),
                        Qt.AlignVCenter | Qt.AlignRight,
                        ts,
                    )
        pr.end()

    def _hit(self, pos: QPoint) -> int:
        """Return the move index at a click position."""
        row = (pos.y() + self._scroll) // self.ROW_H
        num_w = 38
        col_w = (self.width() - num_w) // 2
        if pos.x() < num_w:
            return -1
        side = 0 if pos.x() < num_w + col_w else 1
        return row * 2 + side

    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.LeftButton:
            idx = self._hit(ev.position().toPoint())
            if 0 <= idx < len(self._moves):
                self.move_clicked.emit(idx + 1)

    def mouseMoveEvent(self, ev: QMouseEvent):
        h = self._hit(ev.position().toPoint())
        if h != self._hover:
            self._hover = h
            self.update()

    def leaveEvent(self, ev):
        self._hover = -1
        self.update()

    def wheelEvent(self, ev: QWheelEvent):
        self._scroll -= ev.angleDelta().y() // 3
        self._clamp_scroll()
        self.update()
