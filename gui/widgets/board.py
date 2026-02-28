"""Interactive chess board with drag-and-drop, highlights, animation, and promotion overlay."""

from typing import Optional, Dict, Tuple

import chess
from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Qt, QTimer, QRect, QPoint, QPointF, Signal
from PySide6.QtGui import (
    QPainter,
    QColor,
    QBrush,
    QPen,
    QPixmap,
    QFont,
    QMouseEvent,
    QPaintEvent,
    QLinearGradient,
    QRadialGradient,
    QPainterPath,
)

from gui.helpers import (
    _PIECES,
    MIN_BOARD_PX,
    ANIM_DURATION_MS,
    BOARD_LIGHT,
    BOARD_DARK,
    HL_SELECTED,
    HL_LAST_MOVE,
    HL_PREMOVE,
    HL_CHECK_INNER,
    HL_CHECK_OUTER,
    HL_LEGAL_DOT,
    HL_LEGAL_CAP,
)


class ChessBoardWidget(QWidget):
    """QPainter-based board with drag-drop, slide animation, and promotion overlay."""

    move_made = Signal(object)
    premove_set = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(MIN_BOARD_PX, MIN_BOARD_PX)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self.board: chess.Board = chess.Board()
        self.flipped: bool = False
        self.selected_sq: Optional[int] = None
        self.premove: Optional[chess.Move] = None
        self.game_over: bool = False
        self.engine_thinking: bool = False
        self.human_color: bool = chess.WHITE

        # Scaled pixmap caches (invalidated on resize)
        self._cached_sz: int = 0
        self._scaled: Dict[str, QPixmap] = {}
        self._scaled_drag: Dict[str, QPixmap] = {}

        self._drag_sq: Optional[int] = None
        self._drag_pos: Optional[QPoint] = None

        self._promo: Optional[Tuple[int, int]] = None
        self._promo_hover: int = -1

        # Slide animation state
        self._anim_move: Optional[chess.Move] = None
        self._anim_t: float = 1.0
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)
        self._anim_timer.timeout.connect(self._anim_tick)

        self._hover_sq: Optional[int] = None

    # ── Pixmap cache ────────────────────────────────────────

    def _ensure_cache(self):
        """Rebuild scaled pixmaps when square size changes."""
        sz = self.sq_size
        if sz != self._cached_sz and sz > 0:
            self._cached_sz = sz
            self._scaled = {
                k: v.scaled(sz, sz, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                for k, v in _PIECES.items()
            }
            dsz = int(sz * 1.12)
            self._scaled_drag = {
                k: v.scaled(dsz, dsz, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                for k, v in _PIECES.items()
            }

    # ── Animation ───────────────────────────────────────────

    def animate_move(self, mv: chess.Move):
        """Start ease-out cubic slide animation."""
        self._anim_move = mv
        self._anim_t = 0.0
        self._anim_timer.start()

    def _anim_tick(self):
        self._anim_t += 16.0 / ANIM_DURATION_MS
        if self._anim_t >= 1.0:
            self._anim_t = 1.0
            self._anim_timer.stop()
            self._anim_move = None
        self.update()

    # ── Geometry ────────────────────────────────────────────

    @property
    def sq_size(self) -> int:
        return min(self.width(), self.height()) // 8

    @property
    def origin(self) -> QPoint:
        sz = self.sq_size * 8
        return QPoint((self.width() - sz) // 2, (self.height() - sz) // 2)

    def _sq_px(self, sq: int) -> QPoint:
        """Square index → pixel position (top-left corner)."""
        c, r = chess.square_file(sq), chess.square_rank(sq)
        if self.flipped:
            c, r = 7 - c, r
        else:
            r = 7 - r
        o = self.origin
        return QPoint(o.x() + c * self.sq_size, o.y() + r * self.sq_size)

    def _px_sq(self, pos: QPoint) -> Optional[int]:
        """Pixel position → square index (or None if off-board)."""
        o = self.origin
        c = (pos.x() - o.x()) // self.sq_size
        r = (pos.y() - o.y()) // self.sq_size
        if not (0 <= c < 8 and 0 <= r < 8):
            return None
        if self.flipped:
            c, r = 7 - c, r
        else:
            r = 7 - r
        return chess.square(c, r)

    # ── Painting ────────────────────────────────────────────

    def paintEvent(self, event: QPaintEvent):
        self._ensure_cache()
        pr = QPainter(self)
        pr.setRenderHint(QPainter.Antialiasing)
        pr.setRenderHint(QPainter.SmoothPixmapTransform)
        self._paint_board(pr)
        self._paint_highlights(pr)
        self._paint_pieces(pr)
        self._paint_coords(pr)
        self._paint_drag(pr)
        if self._promo:
            self._paint_promo(pr)
        pr.end()

    def _paint_board(self, pr: QPainter):
        """Draw 3D wooden frame and board squares."""
        o, sz = self.origin, self.sq_size
        bpx = sz * 8
        fr = 6

        pr.setPen(Qt.NoPen)
        pr.setBrush(QBrush(QColor(0, 0, 0, 80)))
        pr.drawRoundedRect(
            o.x() - fr - 2, o.y() - fr - 2, bpx + fr * 2 + 4, bpx + fr * 2 + 4, 6, 6
        )

        grad = QLinearGradient(
            o.x() - fr, o.y() - fr, o.x() + bpx + fr, o.y() + bpx + fr
        )
        grad.setColorAt(0.0, QColor("#5a4630"))
        grad.setColorAt(0.3, QColor("#6b5438"))
        grad.setColorAt(0.7, QColor("#4a3828"))
        grad.setColorAt(1.0, QColor("#3a2a1a"))
        pr.setBrush(QBrush(grad))
        pr.drawRoundedRect(o.x() - fr, o.y() - fr, bpx + fr * 2, bpx + fr * 2, 4, 4)

        pr.setPen(QPen(QColor(255, 255, 255, 30), 1))
        pr.setBrush(Qt.NoBrush)
        pr.drawLine(
            o.x() - fr + 2, o.y() - fr + 1, o.x() + bpx + fr - 2, o.y() - fr + 1
        )
        pr.drawLine(
            o.x() - fr + 1, o.y() - fr + 2, o.x() - fr + 1, o.y() + bpx + fr - 2
        )

        pr.setPen(Qt.NoPen)
        for r in range(8):
            for c in range(8):
                clr = BOARD_LIGHT if (r + c) % 2 == 0 else BOARD_DARK
                pr.fillRect(o.x() + c * sz, o.y() + r * sz, sz, sz, QBrush(clr))

    def _paint_highlights(self, pr: QPainter):
        """Draw hover, last move, selection, premove, check, and legal-move indicators."""
        sz = self.sq_size

        if self._hover_sq is not None and not self._promo:
            pt = self._sq_px(self._hover_sq)
            pr.fillRect(pt.x(), pt.y(), sz, sz, QBrush(QColor(255, 255, 255, 20)))

        if self.board.move_stack:
            last = self.board.move_stack[-1]
            for sq in (last.from_square, last.to_square):
                pt = self._sq_px(sq)
                pr.fillRect(pt.x(), pt.y(), sz, sz, QBrush(HL_LAST_MOVE))

        if self.selected_sq is not None:
            pt = self._sq_px(self.selected_sq)
            pr.fillRect(pt.x(), pt.y(), sz, sz, QBrush(HL_SELECTED))

        if self.premove:
            for sq in (self.premove.from_square, self.premove.to_square):
                pt = self._sq_px(sq)
                pr.fillRect(pt.x(), pt.y(), sz, sz, QBrush(HL_PREMOVE))

        if self.board.is_check():
            ksq = self.board.king(self.board.turn)
            if ksq is not None:
                pt = self._sq_px(ksq)
                cx, cy = pt.x() + sz / 2, pt.y() + sz / 2
                g = QRadialGradient(QPointF(cx, cy), sz * 0.55)
                g.setColorAt(0, HL_CHECK_INNER)
                g.setColorAt(1, HL_CHECK_OUTER)
                pr.fillRect(pt.x(), pt.y(), sz, sz, QBrush(g))

        src = self._drag_sq if self._drag_sq is not None else self.selected_sq
        if (
            src is not None
            and self.board.turn == self.human_color
            and not self.game_over
        ):
            for mv in self.board.legal_moves:
                if mv.from_square != src:
                    continue
                pt = self._sq_px(mv.to_square)
                cx, cy = pt.x() + sz // 2, pt.y() + sz // 2
                pr.setPen(Qt.NoPen)
                if self.board.is_capture(mv):
                    path = QPainterPath()
                    path.addEllipse(QPointF(cx, cy), sz * 0.47, sz * 0.47)
                    inner = QPainterPath()
                    inner.addEllipse(QPointF(cx, cy), sz * 0.36, sz * 0.36)
                    path -= inner
                    pr.setBrush(QBrush(HL_LEGAL_CAP))
                    pr.drawPath(path)
                else:
                    pr.setBrush(QBrush(HL_LEGAL_DOT))
                    pr.drawEllipse(QPointF(cx, cy), sz * 0.14, sz * 0.14)

    def _paint_pieces(self, pr: QPainter):
        """Draw pieces with premove ghost and slide animation."""
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if self.premove:
                if sq == self.premove.from_square:
                    piece = None
                elif sq == self.premove.to_square:
                    piece = self.board.piece_at(self.premove.from_square)
            if piece and sq != self._drag_sq:
                sc = self._scaled.get(piece.symbol())
                if sc:
                    if (
                        self._anim_move
                        and sq == self._anim_move.to_square
                        and self._anim_t < 1.0
                    ):
                        from_pt = self._sq_px(self._anim_move.from_square)
                        to_pt = self._sq_px(sq)
                        t = 1.0 - (1.0 - self._anim_t) ** 3
                        ax = from_pt.x() + (to_pt.x() - from_pt.x()) * t
                        ay = from_pt.y() + (to_pt.y() - from_pt.y()) * t
                        pr.drawPixmap(QPoint(int(ax), int(ay)), sc)
                    else:
                        pr.drawPixmap(self._sq_px(sq), sc)

    def _paint_coords(self, pr: QPainter):
        """Draw rank numbers and file letters."""
        sz = self.sq_size
        o = self.origin
        pr.setFont(QFont("Segoe UI", max(8, sz // 8), QFont.Bold))
        for i in range(8):
            rank = i if self.flipped else 7 - i
            pr.setPen(QPen(BOARD_DARK if i % 2 == 0 else BOARD_LIGHT))
            pr.drawText(o.x() + 3, o.y() + i * sz + max(12, sz // 5), str(rank + 1))
            file_i = 7 - i if self.flipped else i
            pr.setPen(QPen(BOARD_DARK if (7 + i) % 2 == 0 else BOARD_LIGHT))
            pr.drawText(
                o.x() + i * sz + sz - max(11, sz // 6),
                o.y() + 8 * sz - 3,
                chr(ord("a") + file_i),
            )

    def _paint_drag(self, pr: QPainter):
        """Draw piece being dragged."""
        if self._drag_sq is None or self._drag_pos is None:
            return
        piece = self.board.piece_at(self._drag_sq)
        if not piece:
            return
        sc = self._scaled_drag.get(piece.symbol())
        if not sc:
            return
        sz = sc.width()
        pr.setOpacity(0.92)
        pr.drawPixmap(self._drag_pos.x() - sz // 2, self._drag_pos.y() - sz // 2, sc)
        pr.setOpacity(1.0)

    def _paint_promo(self, pr: QPainter):
        """Draw promotion piece chooser overlay."""
        from_sq, to_sq = self._promo
        sz = self.sq_size
        col_px = self._sq_px(to_sq).x()
        is_top = (chess.square_rank(to_sq) == 7) != self.flipped
        start_y = self.origin.y() if is_top else self.origin.y() + 4 * sz

        pr.fillRect(self.rect(), QBrush(QColor(0, 0, 0, 140)))

        menu = QRect(col_px - 4, start_y - 4, sz + 8, sz * 4 + 8)
        pr.setPen(Qt.NoPen)
        pr.setBrush(QBrush(QColor(0, 0, 0, 60)))
        pr.drawRoundedRect(menu.adjusted(3, 3, 3, 3), 10, 10)
        grad = QLinearGradient(col_px, start_y, col_px, start_y + sz * 4)
        grad.setColorAt(0, QColor("#3c3a36"))
        grad.setColorAt(1, QColor("#302e2b"))
        pr.setBrush(QBrush(grad))
        pr.setPen(QPen(QColor("#555350"), 1))
        pr.drawRoundedRect(menu, 10, 10)

        pr.setFont(QFont("Segoe UI", 9, QFont.Bold))
        pr.setPen(QPen(QColor("#9b9892")))
        label_y = start_y - 20 if is_top else start_y + sz * 4 + 12
        pr.drawText(QRect(col_px - 20, label_y, sz + 40, 16), Qt.AlignCenter, "PROMOTE")

        # Determine color of promoting pawn
        promo_piece = self.board.piece_at(from_sq)
        is_white_promo = promo_piece.color if promo_piece else True

        for i, base_sym in enumerate("QRBN"):
            y = start_y + i * sz
            if self._promo_hover == i:
                pr.setPen(Qt.NoPen)
                pr.setBrush(QBrush(QColor(129, 182, 76, 60)))
                pr.drawRoundedRect(col_px - 2, y, sz + 4, sz, 6, 6)
            sym = base_sym if is_white_promo else base_sym.lower()
            sc = self._scaled.get(sym)
            if sc:
                psz = sz - 8
                psc = (
                    sc.scaled(psz, psz, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    if sc.width() != psz
                    else sc
                )
                pr.drawPixmap(col_px + 4, y + 4, psc)

    # ── Mouse events ────────────────────────────────────────

    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() != Qt.LeftButton:
            return
        pos = ev.position().toPoint()
        if self._promo:
            self._click_promo(pos)
            return
        sq = self._px_sq(pos)
        if sq is None:
            return
        piece = self.board.piece_at(sq)
        if piece and piece.color == self.human_color:
            self._drag_sq = sq
            self._drag_pos = pos
            self.selected_sq = sq
            if self.board.turn == self.human_color:
                self.premove = None
            self.update()
        elif self.selected_sq is not None:
            self._attempt(self.selected_sq, sq)
            self.selected_sq = None
            self._drag_sq = None
            self.update()

    def mouseMoveEvent(self, ev: QMouseEvent):
        pos = ev.position().toPoint()
        if self._promo:
            from_sq, to_sq = self._promo
            sz = self.sq_size
            col_px = self._sq_px(to_sq).x()
            is_top = (chess.square_rank(to_sq) == 7) != self.flipped
            start_y = self.origin.y() if is_top else self.origin.y() + 4 * sz
            old = self._promo_hover
            if col_px - 4 <= pos.x() <= col_px + sz + 4:
                idx = (pos.y() - start_y) // sz
                self._promo_hover = idx if 0 <= idx < 4 else -1
            else:
                self._promo_hover = -1
            if self._promo_hover != old:
                self.update()
            return
        if self._drag_sq is not None:
            self._drag_pos = pos
            self.update()
        else:
            new_hover = self._px_sq(pos)
            if new_hover != self._hover_sq:
                self._hover_sq = new_hover
                self.update()

    def mouseReleaseEvent(self, ev: QMouseEvent):
        if ev.button() != Qt.LeftButton or self._drag_sq is None:
            return
        tgt = self._px_sq(ev.position().toPoint())
        if tgt is not None and tgt != self._drag_sq:
            self._attempt(self._drag_sq, tgt)
        self._drag_sq = None
        self._drag_pos = None
        self.update()

    def leaveEvent(self, ev):
        self._hover_sq = None
        self.update()

    def _attempt(self, fr: int, to: int):
        """Handle move or premove attempt, including promotion detection."""
        piece = self.board.piece_at(fr)
        if not piece:
            return
        if piece.piece_type == chess.PAWN and chess.square_rank(to) in (0, 7):
            promo = chess.Move(fr, to, promotion=chess.QUEEN)
            if self.board.turn == self.human_color and promo in self.board.legal_moves:
                self._promo = (fr, to)
                self.update()
                return
            elif self.board.turn != self.human_color:
                self.premove_set.emit(chess.Move(fr, to, promotion=chess.QUEEN))
                return
        mv = chess.Move(fr, to)
        if self.board.turn == self.human_color:
            self.move_made.emit(mv)
        else:
            self.premove_set.emit(mv)

    def _click_promo(self, pos: QPoint):
        """Handle click on a promotion piece option."""
        from_sq, to_sq = self._promo
        sz = self.sq_size
        col_px = self._sq_px(to_sq).x()
        is_top = (chess.square_rank(to_sq) == 7) != self.flipped
        start_y = self.origin.y() if is_top else self.origin.y() + 4 * sz
        if col_px <= pos.x() <= col_px + sz:
            idx = (pos.y() - start_y) // sz
            if 0 <= idx < 4:
                promo_pt = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT][idx]
                self._promo = None
                self.move_made.emit(chess.Move(from_sq, to_sq, promotion=promo_pt))
                return
        self._promo = None
        self.update()
