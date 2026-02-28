"""Mini sparkline chart showing evaluation over the game."""

from typing import List

from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import (
    QPainter,
    QColor,
    QBrush,
    QPen,
    QFont,
    QPaintEvent,
    QPainterPath,
)


class EvalGraphWidget(QWidget):
    """Sparkline with green/red fill areas for white/black advantage."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(64)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._evals: List[int] = []

    def set_evals(self, evals: List[int]):
        self._evals = evals
        self.update()

    def paintEvent(self, event: QPaintEvent):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor("#262522")))
        p.drawRoundedRect(0, 0, w, h, 6, 6)

        if not self._evals:
            p.setPen(QPen(QColor("#48463f")))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(self.rect(), Qt.AlignCenter, "Evaluation Graph")
            p.end()
            return

        mid, margin = h / 2.0, 6

        # Center line
        p.setPen(QPen(QColor("#48463f"), 1, Qt.DashLine))
        p.drawLine(margin, int(mid), w - margin, int(mid))

        # Build points (prepend 0 for starting position)
        evals = [0] + self._evals
        n = len(evals)
        dx = (w - margin * 2) / max(1, n - 1)
        points = []
        for i, cp in enumerate(evals):
            x = margin + i * dx
            y = mid - max(-600, min(600, cp)) / 600.0 * (mid - margin)
            points.append((x, y))

        # Green fill (white advantage)
        path_g = QPainterPath()
        path_g.moveTo(points[0][0], mid)
        for x, y in points:
            path_g.lineTo(x, min(y, mid))
        path_g.lineTo(points[-1][0], mid)
        path_g.closeSubpath()
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(129, 182, 76, 50)))
        p.drawPath(path_g)

        # Red fill (black advantage)
        path_r = QPainterPath()
        path_r.moveTo(points[0][0], mid)
        for x, y in points:
            path_r.lineTo(x, max(y, mid))
        path_r.lineTo(points[-1][0], mid)
        path_r.closeSubpath()
        p.setBrush(QBrush(QColor(224, 122, 95, 50)))
        p.drawPath(path_r)

        # Eval line
        line = QPainterPath()
        line.moveTo(points[0][0], points[0][1])
        for x, y in points[1:]:
            line.lineTo(x, y)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor("#c3c1bf"), 1.5))
        p.drawPath(line)

        # Current position dot
        x, y = points[-1]
        color = QColor("#81b64c") if evals[-1] >= 0 else QColor("#e07a5f")
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(color.lighter(130)))
        p.drawEllipse(QPointF(x, y), 5, 5)
        p.setBrush(QBrush(color))
        p.drawEllipse(QPointF(x, y), 3, 3)

        # Eval label
        cp = evals[-1]
        label = (
            f"M{abs(int(100000 - abs(cp)))}"
            if abs(cp) >= 90000
            else f"{'+' if cp > 0 else ''}{cp / 100:.1f}"
        )
        p.setFont(QFont("Segoe UI", 8, QFont.Bold))
        p.setPen(QPen(color))
        p.drawText(QPointF(min(x + 8, w - 40), max(y - 4, 12)), label)
        p.end()
