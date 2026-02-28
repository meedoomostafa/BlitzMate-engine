"""Animated vertical evaluation bar with smooth interpolation."""

from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Qt, QTimer, QRect
from PySide6.QtGui import QPainter, QBrush, QPen, QFont, QPaintEvent

from gui.helpers import EVAL_W, EVAL_B


class EvalBarWidget(QWidget):
    """Vertical bar showing engine evaluation, smoothly animated."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(30)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self._target = 0.0
        self._display = 0.0
        self._raw_cp = 0  # unclamped value for label (mate display)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

    def set_eval(self, cp: int):
        """Set the target eval in centipawns. Bar is clamped to ±1500; label uses raw value."""
        self._raw_cp = cp
        self._target = max(-1500, min(1500, cp))
        if not self._timer.isActive():
            self._timer.start(16)

    def _tick(self):
        diff = self._target - self._display
        if abs(diff) > 0.5:
            self._display += diff * 0.12
            self.update()
        else:
            self._timer.stop()

    def paintEvent(self, event: QPaintEvent):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Black background
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(EVAL_B))
        p.drawRoundedRect(0, 0, w, h, 5, 5)

        # White portion from bottom
        ratio = max(0.02, min(0.98, (self._display + 1500) / 3000.0))
        wh = int(h * ratio)
        p.setBrush(QBrush(EVAL_W))
        p.drawRoundedRect(0, h - wh, w, wh, 5, 5)
        if 0.02 < ratio < 0.98:
            p.setBrush(QBrush(EVAL_B))
            p.drawRect(0, 0, w, h - wh)

        # Eval text (use raw unclamped value for mate detection)
        raw = self._raw_cp
        cp = self._display
        text = (
            f"M{abs(int(100000 - abs(raw)))}"
            if abs(raw) >= 90000
            else f"{abs(cp / 100):.1f}"
        )
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        if cp >= 0:
            p.setPen(QPen(EVAL_B))
            p.drawText(QRect(0, h - 22, w, 20), Qt.AlignCenter, text)
        else:
            p.setPen(QPen(EVAL_W))
            p.drawText(QRect(0, 2, w, 20), Qt.AlignCenter, text)
        p.end()
