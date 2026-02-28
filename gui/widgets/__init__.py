"""Widget package — re-exports all GUI widgets for convenient importing."""

from gui.widgets.eval_bar import EvalBarWidget
from gui.widgets.eval_graph import EvalGraphWidget
from gui.widgets.captured import CapturedPiecesWidget
from gui.widgets.board import ChessBoardWidget
from gui.widgets.history import MoveHistoryWidget
from gui.widgets.timeline import TimelineWidget
from gui.widgets.game_tab import GameTab

__all__ = [
    "EvalBarWidget",
    "EvalGraphWidget",
    "CapturedPiecesWidget",
    "ChessBoardWidget",
    "MoveHistoryWidget",
    "TimelineWidget",
    "GameTab",
]
