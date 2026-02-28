"""Horizontal slider with nav buttons for game history browsing."""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSlider, QSizePolicy
from PySide6.QtCore import Qt, Signal


class TimelineWidget(QWidget):
    """Timeline bar with ⏮ ◀ slider ▶ ⏭ navigation."""

    position_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(44)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(6)

        self.btn_start = self._btn("⏮")
        self.btn_prev = self._btn("◀")
        self.btn_next = self._btn("▶")
        self.btn_end = self._btn("⏭")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self._on_slider)

        lay.addWidget(self.btn_start)
        lay.addWidget(self.btn_prev)
        lay.addWidget(self.slider, stretch=1)
        lay.addWidget(self.btn_next)
        lay.addWidget(self.btn_end)

        self.btn_start.clicked.connect(lambda: self.position_changed.emit(0))
        self.btn_prev.clicked.connect(
            lambda: self.position_changed.emit(max(0, self.slider.value() - 1))
        )
        self.btn_next.clicked.connect(
            lambda: self.position_changed.emit(
                min(self.slider.maximum(), self.slider.value() + 1)
            )
        )
        self.btn_end.clicked.connect(
            lambda: self.position_changed.emit(self.slider.maximum())
        )

    @staticmethod
    def _btn(text: str) -> QPushButton:
        b = QPushButton(text)
        b.setObjectName("nav_btn")
        b.setFixedSize(34, 34)
        return b

    def sync(self, total: int, cur: int):
        """Update slider range and position without emitting signals."""
        self.slider.blockSignals(True)
        self.slider.setMaximum(total)
        self.slider.setValue(cur)
        self.slider.blockSignals(False)

    def _on_slider(self, v: int):
        self.position_changed.emit(v)
