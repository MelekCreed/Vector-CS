"""Render the real HUD painter offscreen from synthetic hand input.

Drives the actual gesture engine + intent layer with the synthetic hand,
then paints HudPainter onto an image over a mock desktop. Used to verify
overlay visuals without screenshots, and to produce README illustrations
(clearly labelled as renders). Outputs docs/images/hud-*.png and an
animated docs/images/hud-demo.gif.

    python scripts/render_hud_preview.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPen  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from vector.config import Config  # noqa: E402
from vector.core.geometry import Rect  # noqa: E402
from vector.core.metrics import Metrics  # noqa: E402
from vector.desktop.backend import FakeBackend, FakeWindow  # noqa: E402
from vector.desktop.monitors import Monitor, VirtualDesktop  # noqa: E402
from vector.gestures.events import SystemState  # noqa: E402
from vector.intent.intent import IntentEngine  # noqa: E402
from vector.overlay.canvas import Canvas  # noqa: E402
from vector.overlay.hud import HudPainter  # noqa: E402
from vector.pipeline import GesturePipeline  # noqa: E402
from vector.runtime import UIState, VisionWorker  # noqa: E402
from vector.sim.scenario import Scenario  # noqa: E402
from vector.sim.synthetic_hand import HandPose  # noqa: E402

W_, H_ = 1600, 900
MON = Monitor("M", Rect(0, 0, W_, H_), Rect(0, 0, W_, H_ - 40), primary=True)
DESK = VirtualDesktop([MON])
OUT = Path(__file__).resolve().parent.parent / "docs" / "images"


class FakeWin:
    """What HudPainter needs from an OverlayWindow."""
    monitor = MON
    dpr = 1.0

    def L(self, x, y):
        return QPointF(x, y)

    def LR(self, r):
        return QRectF(r.x, r.y, r.w, r.h)

    def on_me(self, x, y, pad=0):
        return -pad <= x < W_ + pad and -pad <= y < H_ + pad

    def width(self):
        return W_

    def height(self):
        return H_

    def rect(self):
        return QRectF(0, 0, W_, H_).toRect()


class Ctrl:
    def __init__(self, cfg, canvas, fake_windows):
        self.cfg, self.canvas, self.canvas_lock = cfg, canvas, canvas.lock
        self.metrics = Metrics()
        self.actuator = None
        self.ui = None
        self.fake = fake_windows
        self.painter = HudPainter(self)

    def cursor_px(self, snap):
        return snap.cursor

    def live_frame(self, fb):
        return None

    def clock(self):
        return self.ui.snapshot.t if self.ui and self.ui.snapshot else 0.0

    def debug_line(self):
        return "30 FPS · inf 17.2 · gest 0.8 · e2e 48.0 · paint 3.8 ms"


def paint_desktop(p: QPainter, windows):
    g = QLinearGradient(0, 0, W_, H_)
    g.setColorAt(0, QColor("#1b2735"))
    g.setColorAt(1, QColor("#090a0f"))
    p.fillRect(0, 0, W_, H_, g)
    p.fillRect(0, H_ - 40, W_, 40, QColor(20, 24, 30, 235))
    for w in reversed(windows):
        r = QRectF(w.frame.x, w.frame.y, w.frame.w, w.frame.h)
        p.setPen(QPen(QColor(255, 255, 255, 28), 1))
        p.setBrush(QColor("#f3f4f6" if "Browser" in w.title else "#1f2430"))
        p.drawRoundedRect(r, 8, 8)
        p.fillRect(QRectF(r.x(), r.y(), r.width(), 34), QColor("#e5e7eb" if "Browser" in w.title else "#2a303c"))
        p.setPen(QColor("#111" if "Browser" in w.title else "#ccc"))
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(QRectF(r.x() + 12, r.y(), r.width(), 34), Qt.AlignVCenter, w.title)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 18) if "Browser" in w.title else QColor(255, 255, 255, 12))
        for i in range(6):
            p.drawRoundedRect(QRectF(r.x() + 24, r.y() + 60 + i * 34, r.width() * (0.8 - 0.08 * (i % 3)), 14), 4, 4)


def grab_and_throw(sc):
    P = HandPose
    k = sc.key
    k("R", 0.0, P("point", x=0.36, y=0.80)).key("R", 0.8, P("point", x=0.55, y=0.62))
    k("R", 1.2, P("point", x=0.55, y=0.62)).key("R", 1.25, P("pinch", x=0.55, y=0.62))
    k("R", 1.6, P("pinch", x=0.55, y=0.62)).key("R", 2.4, P("pinch", x=0.47, y=0.66))
    k("R", 2.55, P("pinch", x=0.50, y=0.66)).key("R", 2.72, P("pinch", x=0.70, y=0.65))
    k("R", 2.74, P("point", x=0.71, y=0.65)).key("R", 3.6, P("point", x=0.62, y=0.64))
    return sc


def air_drawing(sc):
    import math as _m
    P = HandPose
    sc.key("R", 0.0, P("point", x=0.30, y=0.66)).key("R", 0.5, P("point", x=0.30, y=0.66))
    sc.key("R", 0.55, P("pinch", x=0.30, y=0.66))
    for i in range(1, 61):                       # a smooth wave, drawn in the air
        t = 0.55 + i * 0.045
        x = 0.30 + i * 0.0068
        y = 0.66 + 0.07 * _m.sin(i / 60 * 2 * _m.pi * 1.5)
        sc.key("R", t, P("pinch", x=x, y=y))
    sc.key("R", 3.3, P("point", x=0.71, y=0.66)).key("R", 3.8, P("point", x=0.66, y=0.60))
    return sc


def app_carousel(sc):
    P = HandPose
    sc.key("R", 0.0, P("point", x=0.50, y=0.66)).key("R", 0.4, P("open", x=0.50, y=0.64))
    sc.key("R", 1.6, P("open", x=0.50, y=0.64)).key("R", 2.3, P("open", x=0.58, y=0.64))
    sc.key("R", 3.0, P("open", x=0.58, y=0.64))
    return sc


def simulate(cfg, frames_wanted, scene=grab_and_throw, draw=False):
    backend = FakeBackend([
        FakeWindow(1, "Browser — vector.dev", Rect(420, 150, 820, 520), "chrome.exe"),
        FakeWindow(2, "Terminal", Rect(120, 420, 560, 330), "wt.exe"),
        FakeWindow(3, "Spotify — Midnight City", Rect(900, 480, 500, 300), "Spotify.exe"),
        FakeWindow(4, "Figma — Vector HUD", Rect(200, 90, 600, 380), "Figma.exe"),
        FakeWindow(5, "Notes", Rect(1100, 100, 380, 300), "Notepad.exe"),
    ])
    canvas = Canvas(cfg.draw.colors, 6.0)
    pipe = GesturePipeline(cfg, DESK)
    pipe.engine.system = SystemState.ACTIVE
    pipe.engine.neutral_required = False
    if draw:
        pipe.engine.draw_mode = True
    intent = IntentEngine(cfg, DESK, backend, lambda c: None, lambda: 0,
                          on_toggle_draw=lambda t: pipe.engine.toggle_draw(t))
    sc = scene(Scenario(noise=0.0008))
    shots = {}
    worker_apply = VisionWorker._apply_locked
    for t, dets in sc.frames():
        events, snap = pipe.process(dets, t, sc.aspect)
        fb = intent.handle(events, snap)
        for e in events:
            if e.kind == "throw":
                backend.windows[0].frame = Rect(MON.work.w / 2, 0, MON.work.w / 2, MON.work.h)
        with canvas.lock:
            worker_apply(canvas, cfg.draw, fb.draw_ops)
        hand_px = [(f.track_id, True, pipe.mapper.to_desktop(np.array([f.palm_center[0] / f.aspect,
                                                                        f.palm_center[1]])))
                   for f in pipe.features]
        ui = UIState(snapshot=snap, feedback=fb, hand_px=hand_px, features=list(pipe.features))
        for name, when in frames_wanted.items():
            if abs(t - when) < 1 / 60:
                shots[name] = (ui, [FakeWindow(w.hwnd, w.title, w.frame) for w in backend.windows],
                               canvas)
        yield t, ui, backend.windows, canvas
    return shots


def render(ctrl, ui, windows) -> QImage:
    img = QImage(W_, H_, QImage.Format_ARGB32_Premultiplied)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    paint_desktop(p, windows)
    ctrl.ui = ui
    ctrl.painter.paint(p, FakeWin())
    p.end()
    return img


def to_pil(q):
    from PIL import Image
    q = q.convertToFormat(QImage.Format_RGBA8888)
    arr = np.frombuffer(q.constBits(), np.uint8).reshape(q.height(), q.width(), 4)
    return Image.fromarray(arr[..., :3].copy())


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    cfg.debug = True
    runs = [("throw", grab_and_throw, False, {"hover": 1.1, "pinch": 1.45, "drag": 2.3, "throw": 3.0}),
            ("draw", air_drawing, True, {"draw": 3.25}),
            ("carousel", app_carousel, False, {"carousel": 2.8})]
    gif_frames = []
    for name, scene, draw, stills in runs:
        for t, ui, windows, canvas in simulate(cfg, {}, scene, draw):
            ctrl = Ctrl(cfg, canvas, windows)
            img = render(ctrl, ui, windows)
            for still, when in stills.items():
                if abs(t - when) < 1 / 60:
                    img.save(str(OUT / f"hud-{still}.png"))
                    print("wrote", OUT / f"hud-{still}.png")
            if int(round(t * 30)) % 2 == 0:
                gif_frames.append(img.scaled(W_ // 2, H_ // 2, Qt.KeepAspectRatio,
                                             Qt.SmoothTransformation))
    try:
        pil = [to_pil(q) for q in gif_frames]
        gif = OUT / "hud-demo.gif"
        pil[0].save(gif, save_all=True, append_images=pil[1:], duration=66, loop=0, optimize=True)
        print("wrote", gif, f"({gif.stat().st_size / 1e6:.1f} MB)")
    except ImportError:
        print("Pillow not installed; skipped GIF")
    return 0


if __name__ == "__main__":
    sys.exit(main())
