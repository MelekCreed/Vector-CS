"""Transparent, click-through, always-on-top HUD — one window per monitor.

Visual language: dark, minimal, one cyan accent, thin strokes, soft glows.
Only what's useful is drawn; the desktop stays fully visible.

Coordinates: everything arrives in *physical* virtual-desktop pixels. Each
window converts to its own logical space with
``(p - monitor_physical_origin) / devicePixelRatio`` — correct on mixed-DPI
layouts where Qt keeps screen origins but scales sizes.
"""

from __future__ import annotations

import math
import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (QBrush, QColor, QFont, QFontMetricsF, QGuiApplication, QPainter,
                           QPainterPath, QPen, QRadialGradient)
from PySide6.QtWidgets import QWidget

from vector.config import Config
from vector.core.geometry import Rect
from vector.desktop.monitors import Monitor
from vector.gestures.events import HandMode, SystemState
from vector.overlay.canvas import Canvas
from vector.vision.landmarks import CONNECTIONS

FONT_FAMILY = "Segoe UI"


def qc(hex_: str, a: float = 1.0) -> QColor:
    c = QColor(hex_)
    c.setAlphaF(max(0.0, min(1.0, a)))
    return c


class Palette:
    def __init__(self, cfg: Config):
        o = cfg.overlay
        self.accent = o.accent
        self.warn = o.warn
        self.danger = o.danger
        self.text = "#E6F7FB"
        self.panel = "#0B1016"
        self.muted = "#8FA3AD"


class OverlayWindow(QWidget):
    def __init__(self, ctrl: "OverlayController", screen, monitor: Monitor):
        super().__init__(None)
        self.ctrl, self.screen_, self.monitor = ctrl, screen, monitor
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                            | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus
                            | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setGeometry(screen.geometry())
        self._canvas_cache = None
        self._canvas_key = None

    def showEvent(self, e):  # noqa: N802
        super().showEvent(e)
        self._win32_styles()

    def _win32_styles(self) -> None:
        import ctypes
        hwnd = int(self.winId())
        u = ctypes.windll.user32
        GWL_EXSTYLE = -20
        ex = u.GetWindowLongW(hwnd, GWL_EXSTYLE)
        # layered + transparent (click-through) + no-activate + tool window
        u.SetWindowLongW(hwnd, GWL_EXSTYLE, ex | 0x80000 | 0x20 | 0x08000000 | 0x80)
        self.ctrl.own_hwnds.add(hwnd)

    def reassert_topmost(self) -> None:
        import ctypes
        HWND_TOPMOST = -1
        ctypes.windll.user32.SetWindowPos(int(self.winId()), HWND_TOPMOST, 0, 0, 0, 0,
                                          0x1 | 0x2 | 0x10)  # NOSIZE|NOMOVE|NOACTIVATE

    # --------------------------------------------------------- coordinates
    @property
    def dpr(self) -> float:
        return self.screen_.devicePixelRatio()

    def L(self, x: float, y: float) -> QPointF:
        """Physical desktop px -> this window's logical coords."""
        return QPointF((x - self.monitor.rect.x) / self.dpr, (y - self.monitor.rect.y) / self.dpr)

    def LR(self, r: Rect) -> QRectF:
        tl = self.L(r.x, r.y)
        return QRectF(tl.x(), tl.y(), r.w / self.dpr, r.h / self.dpr)

    def on_me(self, x: float, y: float, pad: float = 0) -> bool:
        m = self.monitor.rect
        return m.x - pad <= x < m.right + pad and m.y - pad <= y < m.bottom + pad

    # ---------------------------------------------------------------- paint
    def paintEvent(self, _e):  # noqa: N802
        t0 = time.perf_counter()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        try:
            self.ctrl.painter.paint(p, self)
        finally:
            p.end()
        self.ctrl.metrics.add("overlay_paint", (time.perf_counter() - t0) * 1000)


class HudPainter:
    def __init__(self, ctrl: "OverlayController"):
        self.c = ctrl
        self.pal = Palette(ctrl.cfg)
        self.f_small = QFont(FONT_FAMILY, 8)
        self.f_small.setLetterSpacing(QFont.AbsoluteSpacing, 1.2)
        self.f_small.setWeight(QFont.DemiBold)
        self.f_body = QFont(FONT_FAMILY, 10)
        self.f_toast = QFont(FONT_FAMILY, 13)
        self.f_toast.setWeight(QFont.DemiBold)
        self.f_caption = QFont(FONT_FAMILY, 22)
        self.f_caption.setWeight(QFont.DemiBold)
        self.f_mono = QFont("Cascadia Mono", 8)

    # ---------------------------------------------------------------- main
    def paint(self, p: QPainter, w: OverlayWindow) -> None:
        ui = self.c.ui
        snap = ui.snapshot if ui else None
        now = self.c.clock()
        self._canvas(p, w)
        if ui is None or snap is None:
            if w.monitor.primary:
                self._status_pill(p, w, None, ui, now)
            return
        fb = ui.feedback
        if ui.calibration is not None:
            self._calibration(p, w, ui, now)
        if fb is not None:
            self._hover_and_lock(p, w, fb, snap, now)
            self._throw_trail(p, w, fb, now)
        self._throw_preview(p, w, snap)
        self._resize_guides(p, w, snap)
        self._wake_rings(p, w, ui, snap)
        self._cursor(p, w, ui, snap, now)
        if w.monitor.primary:
            self._status_pill(p, w, snap, ui, now)
            if self.c.cfg.overlay.show_skeleton:
                self._skeleton_panel(p, w, ui, snap)
            if ui.demo is not None:
                self._demo_caption(p, w, ui.demo, now)
            if ui.recording is not None:
                self._recording(p, w, ui.recording)

    # --------------------------------------------------------------- canvas
    def _canvas(self, p: QPainter, w: OverlayWindow) -> None:
        cv: Canvas = self.c.canvas
        strokes = cv.render_list()
        if not strokes:
            return
        p.setBrush(Qt.NoBrush)              # paths must never be filled
        for color, width, pts in strokes:
            if not any(w.on_me(x, y, 50) for x, y in pts[:: max(1, len(pts) // 8)]):
                continue
            path = QPainterPath(w.L(*pts[0]))
            for x, y in pts[1:]:
                path.lineTo(w.L(x, y))
            # soft glow underneath, crisp line on top
            p.setPen(QPen(qc(color, 0.18), (width + 8) / w.dpr, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawPath(path)
            p.setPen(QPen(qc(color, 0.95), width / w.dpr, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawPath(path)

    # ------------------------------------------------------- window frames
    def _brackets(self, p: QPainter, r: QRectF, color: QColor, width: float, arm: float) -> None:
        p.setPen(QPen(color, width, Qt.SolidLine, Qt.RoundCap))
        arm = min(arm, r.width() / 3, r.height() / 3)
        for (x, y, dx, dy) in ((r.left(), r.top(), 1, 1), (r.right(), r.top(), -1, 1),
                               (r.right(), r.bottom(), -1, -1), (r.left(), r.bottom(), 1, -1)):
            p.drawLine(QPointF(x, y), QPointF(x + dx * arm, y))
            p.drawLine(QPointF(x, y), QPointF(x, y + dy * arm))

    def _glow_rect(self, p: QPainter, r: QRectF, color: str, strength: float) -> None:
        for i, a in enumerate((0.06, 0.10, 0.16)):
            grow = (3 - i) * 3
            p.setPen(QPen(qc(color, a * strength), 2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r.adjusted(-grow, -grow, grow, grow), 10 + grow, 10 + grow)
        p.setPen(QPen(qc(color, 0.9 * strength), 1.6))
        p.drawRoundedRect(r, 8, 8)

    def _title_chip(self, p: QPainter, r: QRectF, text: str, color: str, alpha: float) -> None:
        if not text:
            return
        p.setFont(self.f_small)
        fm = QFontMetricsF(self.f_small)
        text = fm.elidedText(text.upper(), Qt.ElideRight, min(360.0, r.width()))
        tw = fm.horizontalAdvance(text) + 16
        chip = QRectF(r.left(), r.top() - 24, tw, 18)
        if chip.top() < 2:                  # window touches the screen top: tuck the chip
            chip.moveTop(r.top() + 8)           # inside, top-right, clear of the app's title
            chip.moveRight(r.right() - 8)
        p.setPen(Qt.NoPen)
        p.setBrush(qc(self.pal.panel, 0.72 * alpha))
        p.drawRoundedRect(chip, 9, 9)
        p.setPen(qc(color, alpha))
        p.drawText(chip, Qt.AlignCenter, text)

    def _hover_and_lock(self, p, w, fb, snap, now) -> None:
        acc = self.pal.accent
        if fb.lock_state:
            frame = self.c.live_frame(fb) or fb.lock_frame
            if frame is None or not w.monitor.rect.intersection_area(frame):
                return
            r = w.LR(frame)
            if fb.lock_state == "pinch":
                self._brackets(p, r.adjusted(-4, -4, 4, 4), qc(acc, 0.9), 2.2, 26)
            elif fb.lock_state in ("grab", "drag"):
                self._glow_rect(p, r, acc, 1.0)
                self._brackets(p, r.adjusted(-6, -6, 6, 6), qc(acc, 1.0), 2.6, 30)
            elif fb.lock_state == "resize":
                p.setPen(QPen(qc(acc, 0.9), 1.6, Qt.DashLine))
                p.setBrush(qc(acc, 0.05))
                p.drawRoundedRect(r, 8, 8)
                self._brackets(p, r.adjusted(-6, -6, 6, 6), qc(acc, 1.0), 2.6, 34)
                size = f"{round(frame.w)} × {round(frame.h)}"
                self._title_chip(p, r, size, acc, 1.0)
                return
            self._title_chip(p, r, fb.lock_title, acc, 1.0)
        elif fb.hover_frame is not None and snap.cursor_visible \
                and w.monitor.rect.intersection_area(fb.hover_frame):
            r = w.LR(fb.hover_frame)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(qc(acc, 0.28), 1.2))
            p.drawRoundedRect(r, 8, 8)
            self._brackets(p, r.adjusted(-3, -3, 3, 3), qc(acc, 0.55), 1.6, 18)
            self._title_chip(p, r, fb.hover_title, acc, 0.75)

    # ---------------------------------------------------------------- throw
    def _throw_trail(self, p, w, fb, now) -> None:
        tr = fb.throw_trail
        if tr is None:
            return
        age = now - tr[0]
        if age > 0.9 or age < 0:
            return
        a = 1.0 - age / 0.9
        (x0, y0), (x1, y1), label = tr[1], tr[2], tr[3]
        target = tr[4] if len(tr) > 4 else None
        acc = self.pal.accent
        # 1) destination: where the window is going, glowing then fading
        if target is not None and w.monitor.rect.intersection_area(target):
            r = w.LR(target).adjusted(6, 6, -6, -6)
            p.setBrush(qc(acc, 0.07 * a))
            self._glow_rect(p, r, acc, a)
            self._brackets(p, r.adjusted(-4, -4, 4, 4), qc(acc, a), 2.6, 40)
        if not (w.on_me(x0, y0, 400) or w.on_me(x1, y1, 400)):
            return
        # 2) the throw itself: a streak that shoots out quickly, then fades
        s = w.L(x0, y0)
        e_full = w.L(x1, y1)
        k = min(1.0, age / 0.15)
        e = QPointF(s.x() + (e_full.x() - s.x()) * k, s.y() + (e_full.y() - s.y()) * k)
        e = QPointF(min(max(e.x(), 24), w.width() - 24), min(max(e.y(), 24), w.height() - 24))
        for width, alpha in ((12, 0.10), (6, 0.25), (2.4, 1.0)):
            p.setPen(QPen(qc(acc, alpha * a), width, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(s, e)
        ang = math.atan2(e.y() - s.y(), e.x() - s.x())
        head = QPainterPath(e)
        for da in (2.55, -2.55):
            head.lineTo(QPointF(e.x() + 16 * math.cos(ang + da), e.y() + 16 * math.sin(ang + da)))
            head.moveTo(e)
        p.setPen(QPen(qc(acc, a), 2.6, Qt.SolidLine, Qt.RoundCap))
        p.drawPath(head)

    def _throw_preview(self, p, w, snap) -> None:
        v = snap.throw_preview
        if v is None or snap.drag_point is None or not w.on_me(*snap.drag_point):
            return
        s = w.L(*snap.drag_point)
        e = w.L(snap.drag_point[0] + v[0] * 0.12, snap.drag_point[1] + v[1] * 0.12)
        p.setPen(QPen(qc(self.pal.accent, 0.35), 1.5, Qt.DotLine, Qt.RoundCap))
        p.drawLine(s, e)

    # --------------------------------------------------------------- resize
    def _resize_guides(self, p, w, snap) -> None:
        rp = snap.resize_points
        if rp is None:
            return
        a, b = rp
        if not (w.on_me(*a, 200) or w.on_me(*b, 200)):
            return
        pa, pb = w.L(*a), w.L(*b)
        p.setPen(QPen(qc(self.pal.accent, 0.55), 1.4, Qt.DashLine))
        p.drawLine(pa, pb)
        for q in (pa, pb):
            p.setPen(Qt.NoPen)
            p.setBrush(qc(self.pal.accent, 0.25))
            p.drawEllipse(q, 14, 14)
            p.setBrush(qc(self.pal.accent, 1.0))
            p.drawEllipse(q, 5, 5)

    # ----------------------------------------------------------- wake rings
    def _wake_rings(self, p, w, ui, snap) -> None:
        prog = snap.wake_progress if snap.system == SystemState.SLEEPING else snap.sleep_progress
        if prog <= 0.02:
            return
        label = "HOLD TO ACTIVATE" if snap.system == SystemState.SLEEPING else "HOLD TO SLEEP"
        for _tid, _primary, px in ui.hand_px:
            if not w.on_me(*px):
                continue
            c = w.L(*px)
            self._ring(p, c, 44, prog, self.pal.accent, 3.0, track=0.18)
            p.setFont(self.f_small)
            p.setPen(qc(self.pal.text, 0.85))
            p.drawText(QRectF(c.x() - 90, c.y() + 54, 180, 18), Qt.AlignCenter, label)

    def _ring(self, p, c: QPointF, r: float, prog: float, color: str, width: float,
              track: float = 0.15) -> None:
        rect = QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(qc(color, track), width))
        p.drawEllipse(rect)
        p.setPen(QPen(qc(color, 0.95), width, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect, 90 * 16, int(-prog * 360 * 16))

    # --------------------------------------------------------------- cursor
    def _cursor(self, p, w, ui, snap, now) -> None:
        if not snap.cursor_visible or snap.cursor is None:
            return
        pos = self.c.cursor_px(snap)
        if pos is None or not w.on_me(*pos):
            return
        c = w.L(*pos)
        acc = self.pal.accent
        mode = snap.primary_mode
        fb = ui.feedback
        hover = fb is not None and fb.hover_frame is not None
        if snap.draw_mode:
            color = self.c.canvas.color
            if mode == HandMode.ERASING:
                r = self.c.cfg.draw.eraser_radius / w.dpr
                p.setPen(QPen(qc(self.pal.text, 0.7), 1.2, Qt.DashLine))
                p.setBrush(qc(self.pal.text, 0.06))
                p.drawEllipse(c, r, r)
                return
            rr = max(3.0, self.c.canvas.width / w.dpr / 2 + 1)
            p.setPen(QPen(qc(color, 0.9), 1.4))
            p.setBrush(qc(color, 0.9 if mode == HandMode.DRAWING else 0.15))
            p.drawEllipse(c, rr, rr)
            return
        if mode in (HandMode.PINCH_STARTED, HandMode.GRABBING, HandMode.DRAGGING, HandMode.RESIZING):
            g = QRadialGradient(c, 22)
            g.setColorAt(0, qc(acc, 0.55))
            g.setColorAt(1, qc(acc, 0.0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(g))
            p.drawEllipse(c, 22, 22)
            p.setBrush(qc(acc, 1.0))
            p.drawEllipse(c, 6, 6)
        else:
            dim = 0.55 if mode == HandMode.HOVERING else 1.0
            r = 14 if hover else 11
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(qc(acc, 0.9 * dim), 1.6))
            p.drawEllipse(c, r, r)
            p.setPen(Qt.NoPen)
            p.setBrush(qc(acc, dim))
            p.drawEllipse(c, 2.6, 2.6)
        for hv in snap.hands:
            if hv.primary and hv.hold_progress > 0.08:
                self._ring(p, c, 22, hv.hold_progress, self.pal.warn, 2.4)

    # ---------------------------------------------------------- status pill
    def _status_pill(self, p, w, snap, ui, now) -> None:
        pal = self.pal
        if snap is None:
            state, color, mode = "STARTING", pal.muted, ""
        else:
            state = snap.system.value
            color = {SystemState.ACTIVE: pal.accent, SystemState.SLEEPING: pal.muted,
                     SystemState.DISABLED: pal.danger}[snap.system]
            mode = "" if snap.system != SystemState.ACTIVE else snap.primary_mode.value.replace("_", " ")
            if snap.neutral_required and snap.system == SystemState.ACTIVE:
                mode = "RELAX HAND"
        if ui is not None and ui.camera_error:
            state, color, mode = "NO CAMERA", pal.danger, ""
        parts = [state] + ([mode] if mode else [])
        if snap is not None and snap.draw_mode:
            parts.append("DRAW")
        text = "   ·   ".join(parts)
        p.setFont(self.f_small)
        fm = QFontMetricsF(self.f_small)
        debug = self.c.cfg.debug
        dbg = self.c.debug_line() if debug else ""
        width = max(fm.horizontalAdvance(text) + 46, fm.horizontalAdvance(dbg) + 30 if dbg else 0)
        h = 30 if not dbg else 48
        margin = 18
        corner = self.c.cfg.overlay.hud_corner
        x = w.width() - width - margin if "right" in corner else margin
        y = margin if "top" in corner else w.height() - h - margin
        box = QRectF(x, y, width, h)
        p.setPen(QPen(qc("#FFFFFF", 0.07), 1))
        p.setBrush(qc(pal.panel, 0.78))
        p.drawRoundedRect(box, 15, 15)
        pulse = 0.6 + 0.4 * math.sin(now * 3.0) if state == "ACTIVE" else 1.0
        p.setPen(Qt.NoPen)
        p.setBrush(qc(color, 0.25 * pulse))
        p.drawEllipse(QPointF(x + 17, y + 15), 7, 7)
        p.setBrush(qc(color, 1.0))
        p.drawEllipse(QPointF(x + 17, y + 15), 3.5, 3.5)
        p.setPen(qc(pal.text, 0.92))
        p.drawText(QRectF(x + 30, y, width - 36, 30), Qt.AlignVCenter | Qt.AlignLeft, text)
        if snap is not None and snap.draw_mode:
            p.setPen(Qt.NoPen)
            p.setBrush(qc(self.c.canvas.color, 1))
            p.drawEllipse(QPointF(x + width - 14, y + 15), 4, 4)
        if dbg:
            p.setFont(self.f_mono)
            p.setPen(qc(pal.muted, 0.95))
            p.drawText(QRectF(x + 14, y + 28, width - 20, 16), Qt.AlignVCenter | Qt.AlignLeft, dbg)
        # toast: last accepted action, fades out
        fbk = ui.feedback if ui else None
        if fbk is not None and fbk.toast is not None and snap is not None:
            age = now - fbk.toast[0]
            if 0 <= age < 1.4:
                a = 1.0 if age < 0.9 else 1.0 - (age - 0.9) / 0.5
                rise = min(1.0, age / 0.12)
                p.setFont(self.f_toast)
                fmt = QFontMetricsF(self.f_toast)
                tw = fmt.horizontalAdvance(fbk.toast[1]) + 32
                tb = QRectF(x + width - tw, y + h + 10 - 6 * (1 - rise), tw, 34)
                p.setPen(Qt.NoPen)
                p.setBrush(qc(pal.panel, 0.66 * a))
                p.drawRoundedRect(tb, 17, 17)
                p.setPen(qc(pal.accent, a))
                p.drawText(tb, Qt.AlignCenter, fbk.toast[1])

    # ------------------------------------------------------ skeleton panel
    def _skeleton_panel(self, p, w, ui, snap) -> None:
        if not snap.hands:
            return
        pw, ph, margin = 196, 150, 18
        box = QRectF(w.width() - pw - margin, w.height() - ph - margin - 44, pw, ph)
        p.setPen(QPen(qc("#FFFFFF", 0.06), 1))
        p.setBrush(qc(self.pal.panel, 0.72))
        p.drawRoundedRect(box, 14, 14)
        # Fit all hands' iso coords into the box.
        pts = np.vstack([hv.iso for hv in snap.hands])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        span = max(hi[0] - lo[0], hi[1] - lo[1], 0.2) * 1.25
        cx, cy = (lo + hi) / 2
        s = min(pw, ph - 22) / span
        op = self.c.cfg.overlay.skeleton_opacity
        for hv in snap.hands:
            color = self.pal.accent if hv.primary else self.pal.text
            alpha = (1.0 if hv.primary else 0.5) * (0.35 if hv.stale else 1.0)
            q = [QPointF(box.center().x() + (x - cx) * s, box.top() + (ph - 22) / 2 + (y - cy) * s)
                 for x, y in hv.iso]
            p.setPen(QPen(qc(color, 0.8 * alpha * op / 0.55), 1.3, Qt.SolidLine, Qt.RoundCap))
            for a, b in CONNECTIONS:
                p.drawLine(q[a], q[b])
            p.setPen(Qt.NoPen)
            p.setBrush(qc(color, alpha))
            for i, pt in enumerate(q):
                p.drawEllipse(pt, 2.2 if i not in (4, 8) else 3.2, 2.2 if i not in (4, 8) else 3.2)
        prim = next((hv for hv in snap.hands if hv.primary), snap.hands[0])
        p.setFont(self.f_small)
        p.setPen(qc(self.pal.muted, 1))
        pose = "PINCH" if prim.pinch else prim.pose.replace("_", " ").upper()
        label = f"{prim.label.upper()} · {pose}"
        p.drawText(QRectF(box.left(), box.bottom() - 22, pw, 18), Qt.AlignCenter, label)

    # -------------------------------------------------------- calibration
    def _calibration(self, p, w, ui, now) -> None:
        st = ui.calibration
        p.fillRect(w.rect(), qc("#05080C", 0.45))
        if w.monitor.primary:
            p.setFont(self.f_caption)
            p.setPen(qc(self.pal.text, 0.95))
            p.drawText(QRectF(0, w.height() * 0.40, w.width(), 40), Qt.AlignCenter, st.message)
            p.setFont(self.f_small)
            p.setPen(qc(self.pal.muted, 1))
            sub = {"hand": "CALIBRATION · STEP 1 OF 2", "targets":
                   f"CALIBRATION · STEP 2 OF 2 · TARGET {st.target + 1} OF 4"}.get(st.step.value, "")
            p.drawText(QRectF(0, w.height() * 0.40 + 46, w.width(), 20), Qt.AlignCenter, sub)
            if st.step.value == "hand":
                self._ring(p, QPointF(w.width() / 2, w.height() * 0.58), 36, st.progress,
                           self.pal.accent, 3)
        if ui.calib_target is not None and w.on_me(*ui.calib_target):
            c = w.L(*ui.calib_target)
            pulse = 1 + 0.08 * math.sin(now * 5)
            p.setPen(QPen(qc(self.pal.accent, 0.5), 1.2))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(c, 30 * pulse, 30 * pulse)
            p.setPen(Qt.NoPen)
            p.setBrush(qc(self.pal.accent, 1))
            p.drawEllipse(c, 6, 6)
            self._ring(p, c, 42, st.progress, self.pal.accent, 3)

    # --------------------------------------------------------------- demo
    def _demo_caption(self, p, w, demo, now) -> None:
        if demo.done:
            text, sub = "That's Vector.", ""
        else:
            text, sub = demo.caption, f"{demo.index + 1} / {demo.total}"
        p.setFont(self.f_caption)
        fm = QFontMetricsF(self.f_caption)
        tw = fm.horizontalAdvance(text) + 64
        box = QRectF((w.width() - tw) / 2, w.height() - 118, tw, 56)
        p.setPen(QPen(qc("#FFFFFF", 0.08), 1))
        p.setBrush(qc(self.pal.panel, 0.6))
        p.drawRoundedRect(box, 28, 28)
        p.setPen(qc(self.pal.text, 0.96))
        p.drawText(box, Qt.AlignCenter, text)
        if sub:
            p.setFont(self.f_small)
            p.setPen(qc(self.pal.accent, 0.9))
            p.drawText(QRectF(box.left(), box.bottom() + 6, box.width(), 16), Qt.AlignCenter, sub)

    def _recording(self, p, w, rec) -> None:
        state = rec["state"]
        text = {"countdown": f"GET READY · {rec['label'].upper()} · {rec['remaining']:.0f}",
                "recording": f"RECORDING · {rec['label'].upper()} · {max(0, rec['remaining']):.1f}s",
                "saved": f"SAVED · {rec['label'].upper()}"}[state]
        color = {"countdown": self.pal.warn, "recording": self.pal.danger,
                 "saved": self.pal.accent}[state]
        p.setFont(self.f_small)
        fm = QFontMetricsF(self.f_small)
        tw = fm.horizontalAdvance(text) + 44
        box = QRectF(18, 18, tw, 30)
        p.setPen(Qt.NoPen)
        p.setBrush(qc(self.pal.panel, 0.7))
        p.drawRoundedRect(box, 15, 15)
        p.setBrush(qc(color, 1))
        p.drawEllipse(QPointF(box.left() + 17, box.center().y()), 4.5, 4.5)
        p.setPen(qc(self.pal.text, 1))
        p.drawText(box.adjusted(30, 0, 0, 0), Qt.AlignVCenter | Qt.AlignLeft, text)


class OverlayController:
    """Owns the per-monitor windows and a render timer."""

    def __init__(self, cfg: Config, desktop, shared, canvas: Canvas, canvas_lock, actuator, metrics):
        self.cfg, self.desktop, self.shared = cfg, desktop, shared
        self.canvas, self.canvas_lock = canvas, canvas_lock
        self.actuator, self.metrics = actuator, metrics
        self.own_hwnds: set[int] = set()
        self.ui = None
        self.painter = HudPainter(self)
        self.windows: list[OverlayWindow] = []
        by_name = {m.name: m for m in desktop.monitors}
        for scr in QGuiApplication.screens():
            mon = by_name.get(scr.name())
            if mon is None:     # fall back: match by logical geometry
                g = scr.geometry()
                mon = min(desktop.monitors, key=lambda m: abs(m.rect.x / scr.devicePixelRatio() - g.x())
                          + abs(m.rect.y / scr.devicePixelRatio() - g.y()))
            win = OverlayWindow(self, scr, mon)
            self.windows.append(win)
        self._timer = QTimer()
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self._tick)
        self._topmost = QTimer()
        self._topmost.timeout.connect(lambda: [w.reassert_topmost() for w in self.windows])
        self._idle_frames = 0
        self._last_idx = -1

    def show(self) -> None:
        for w in self.windows:
            w.show()
        self._timer.start(int(1000 / max(10, self.cfg.overlay.fps)))
        self._topmost.start(2000)

    def _tick(self) -> None:
        self.ui = self.shared.read()
        snap = self.ui.snapshot if self.ui else None
        busy = (snap is not None and (snap.hands or snap.scroll_velocity)) or \
            (self.ui is not None and (self.ui.calibration is not None or self.ui.recording))
        fb = self.ui.feedback if self.ui else None
        animating = fb is not None and ((fb.toast and time.perf_counter() - fb.toast[0] < 1.5)
                                        or (fb.throw_trail and time.perf_counter() - fb.throw_trail[0] < 1.0))
        if busy or animating or self.canvas.active is not None:
            self._idle_frames = 0
        else:
            self._idle_frames += 1
        # Idle: repaint at ~4 Hz instead of 60 to leave the GPU/CPU alone.
        if self._idle_frames > 2 and self._idle_frames % 15:
            return
        for w in self.windows:
            w.update()

    @staticmethod
    def clock() -> float:
        """Same clock as camera frame timestamps (perf_counter), so animations
        keep fading even if vision stalls."""
        return time.perf_counter()

    def cursor_px(self, snap):
        """Prefer the actuator's 120 Hz spring position (what the OS cursor
        shows) so the HUD cursor and the real cursor never disagree."""
        if self.actuator is not None and self.actuator.armed:
            pos = self.actuator.cursor_position
            if pos is not None and snap.confidence.get("drives_cursor"):
                return pos
        return snap.cursor

    def live_frame(self, fb) -> Rect | None:
        """The locked window's *current* frame (it moves while dragged)."""
        d = self.actuator.drag if self.actuator is not None else None
        hwnd = d.hwnd if d is not None else getattr(self.actuator, "resize_hwnd", None)
        if not hwnd or fb.lock_state not in ("drag", "grab"):
            return None
        from vector.desktop.windows import frame_rect, is_window
        return frame_rect(hwnd) if is_window(hwnd) else None

    def debug_line(self) -> str:
        s = self.metrics.summary()

        def ms(k):
            return f"{s[k][0]:.1f}" if k in s else "–"
        return (f"{self.metrics.rate('vision'):.0f} FPS · inf {ms('inference')} · gest {ms('gesture')}"
                f" · e2e {ms('end_to_end_vision')} · paint {ms('overlay_paint')} ms")
