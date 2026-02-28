"""BlitzMate — Modern Chess GUI (PySide6, multi-tab, parallel engine).

This module contains only the top-level MainWindow and the application
entry point.  All widgets live in ``gui.widgets`` and shared helpers in
``gui.helpers``.
"""

import os
import sys
import multiprocessing

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QStatusBar,
    QToolButton,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QKeySequence, QShortcut

from gui.helpers import CONFIG, EnginePool, SoundManager, QSS, load_piece_pixmaps
from gui.widgets import GameTab

# ── Main window ─────────────────────────────────────────────


class MainWindow(QMainWindow):
    """Top-level window with tab management and keyboard shortcuts."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"BlitzMate — {CONFIG.ui.engine_name}")
        self.setMinimumSize(960, 680)
        self.resize(1280, 820)

        depth = CONFIG.search.depth
        self.pool = EnginePool(depth)
        self.pool.start()
        self.sound = SoundManager()
        self._n = 0

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        self.setCentralWidget(self.tabs)

        plus = QToolButton()
        plus.setText(" + New Game ")
        plus.setStyleSheet("""
            QToolButton {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #8ece52, stop:1 #73a83e);
                color: white; font: bold 13px; border: none;
                padding: 6px 14px; border-radius: 6px; margin: 3px 6px;
            }
            QToolButton:hover {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #9ddb62, stop:1 #81b64c);
            }
        """)
        plus.clicked.connect(self._add_tab)
        self.tabs.setCornerWidget(plus, Qt.TopRightCorner)

        self._build_menu()

        cores = os.cpu_count() or 2
        sb = QStatusBar()
        sb.showMessage(
            f"Engine: {CONFIG.ui.engine_name}  |  Depth: {depth}"
            f"  |  CPU: {cores} cores  |  Workers: {max(1, cores - 1)}"
        )
        self.setStatusBar(sb)

        QShortcut(QKeySequence("F11"), self, self._toggle_fs)
        QShortcut(QKeySequence("Ctrl+T"), self, self._add_tab)
        QShortcut(QKeySequence("Ctrl+W"), self, self._close_cur)
        QShortcut(QKeySequence("Ctrl+N"), self, self._add_tab)
        QShortcut(QKeySequence("Escape"), self, self._exit_fs)

        self._add_tab()

    def _build_menu(self):
        mb = self.menuBar()
        gm = mb.addMenu("&Game")
        gm.addAction("New Tab", self._add_tab, QKeySequence("Ctrl+T"))
        gm.addAction("Close Tab", self._close_cur, QKeySequence("Ctrl+W"))
        gm.addSeparator()
        gm.addAction("Exit", self.close, QKeySequence("Alt+F4"))
        vm = mb.addMenu("&View")
        vm.addAction("Fullscreen", self._toggle_fs, QKeySequence("F11"))
        vm.addAction("Flip Board", self._flip_cur, QKeySequence("F"))

    def _add_tab(self):
        self._n += 1
        t = GameTab(self._n, self.pool, self.sound)
        t.title_changed.connect(lambda s, tab=t: self._retitle(tab, s))
        idx = self.tabs.addTab(t, f"Game {self._n}")
        self.tabs.setCurrentIndex(idx)
        t.setFocus()

    def _close_tab(self, i: int):
        if self.tabs.count() <= 1:
            return
        w = self.tabs.widget(i)
        self.tabs.removeTab(i)
        if isinstance(w, GameTab):
            w.dispose()
        w.deleteLater()

    def _close_cur(self):
        if self.tabs.count() > 1:
            self._close_tab(self.tabs.currentIndex())

    def _retitle(self, tab, title):
        i = self.tabs.indexOf(tab)
        if i >= 0:
            self.tabs.setTabText(i, title)

    def _toggle_fs(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def _exit_fs(self):
        if self.isFullScreen():
            self.showNormal()

    def _flip_cur(self):
        w = self.tabs.currentWidget()
        if isinstance(w, GameTab):
            w._flip()

    def closeEvent(self, event):
        for i in range(self.tabs.count()):
            w = self.tabs.widget(i)
            if isinstance(w, GameTab):
                w.dispose()
        self.pool.shutdown()
        event.accept()


# ── Entry point ─────────────────────────────────────────────


def main():
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass  # already set
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    font = QFont()
    font.setFamilies(["Segoe UI", "Ubuntu", "Noto Sans", "DejaVu Sans"])
    font.setPointSize(11)
    app.setFont(font)
    load_piece_pixmaps()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
