"""Application wiring: builds every component and runs the Qt event loop."""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import time
from pathlib import Path

from vector import config as config_mod
from vector.config import Config

log = logging.getLogger("vector")
ROOT = Path(__file__).resolve().parent.parent

HOTKEYS_HELP = """\
  Esc Esc         emergency stop (disables all gesture OS actions)
  Ctrl+Alt+V      re-arm after emergency stop
  Ctrl+Alt+D      toggle debug window
  Ctrl+Alt+G      toggle draw mode
  Ctrl+Alt+Z / C  undo / clear drawing      Ctrl+Alt+K  next color
  Ctrl+Alt+[ / ]  thinner / thicker pen     Ctrl+Alt+S  export drawing
  Ctrl+Alt+R      record gesture clip (label from debug window)
  Ctrl+Alt+Q      quit"""


def run(cfg: Config, config_path: Path | None, calibrate: bool = False, demo: bool = False,
        debug_window: bool = False, quit_after: float = 0.0) -> int:
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QApplication

    from vector.core.metrics import Metrics
    from vector.desktop.actuator import Actuator, Win32Ops
    from vector.desktop.backend import Win32Backend
    from vector.desktop.dpi import current_awareness
    from vector.desktop.monitors import VirtualDesktop
    from vector.desktop.windows import WindowPolicy
    from vector.intent.intent import IntentEngine
    from vector.overlay.canvas import Canvas
    from vector.overlay.debug import DebugWindow
    from vector.overlay.hud import OverlayController
    from vector.pipeline import GesturePipeline
    from vector.runtime import SharedState, VisionWorker
    from vector.safety import KeyboardGuard
    from vector.vision.camera import Camera
    from vector.vision.tracker import MediaPipeTracker

    app = QApplication.instance() or QApplication(sys.argv)   # Qt sets per-monitor-v2 DPI awareness
    app.setQuitOnLastWindowClosed(False)
    if current_awareness() != 2:
        from vector.desktop.dpi import make_process_dpi_aware
        make_process_dpi_aware()
    desktop = VirtualDesktop.from_system()
    for m in desktop.monitors:
        log.info("monitor %s %s work=%s dpi=%d%s", m.name, m.rect, m.work, m.dpi,
                 " primary" if m.primary else "")

    metrics = Metrics()
    shared = SharedState()
    canvas = Canvas(cfg.draw.colors, cfg.draw.thickness, cfg.draw.min_point_spacing_px)
    policy = WindowPolicy(cfg.safety.blocked_classes, cfg.safety.blocked_processes)
    backend = Win32Backend(policy)
    actuator = Actuator(cfg, Win32Ops()).start()
    pipeline = GesturePipeline(cfg, desktop)

    camera = Camera(cfg.camera)
    try:
        camera.start()
    except RuntimeError as exc:
        log.error("%s", exc)
        return 2
    tracker = MediaPipeTracker(cfg.tracker, root=ROOT)

    worker: VisionWorker | None = None

    def toggle_draw(_t=None):
        worker.toggle_draw()

    intent = IntentEngine(cfg, desktop, backend, actuator.submit, lambda: actuator.generation,
                          on_toggle_draw=lambda t: pipeline.engine.toggle_draw(t))
    worker = VisionWorker(cfg, camera, tracker, pipeline, intent, actuator, shared, canvas, metrics,
                          calibrate=calibrate or not cfg.calibrated, demo=demo,
                          config_path=config_path)

    overlay = None
    if cfg.overlay.enabled:
        overlay = OverlayController(cfg, desktop, shared, canvas, canvas.lock, actuator, metrics)
        overlay.show()
        policy.own_hwnds |= overlay.own_hwnds   # never target our own HUD

    debug = DebugWindow(shared, metrics, actuator, worker, cfg)
    if debug_window or cfg.debug:
        debug.show()

    # --- hotkeys: the hook thread hands callbacks to the Qt thread via a signal
    class Bridge(QObject):
        call = Signal(object)

    bridge = Bridge()
    bridge.call.connect(lambda fn: fn())
    guard = KeyboardGuard(worker.emergency_stop, cfg.safety.failsafe_window_s,
                          cfg.safety.failsafe_double_tap_key)
    guard.dispatch = bridge.call.emit

    def with_canvas(fn):
        def _f():
            with canvas.lock:
                fn()
        return _f

    guard.add_hotkey(cfg.safety.rearm_hotkey, worker.rearm)
    guard.add_hotkey("ctrl+alt+d", lambda: debug.hide() if debug.isVisible() else debug.show())
    guard.add_hotkey("ctrl+alt+g", toggle_draw)
    guard.add_hotkey("ctrl+alt+z", with_canvas(canvas.undo))
    guard.add_hotkey("ctrl+alt+c", with_canvas(canvas.clear))
    guard.add_hotkey("ctrl+alt+k", with_canvas(canvas.next_color))
    guard.add_hotkey("ctrl+alt+[", with_canvas(lambda: canvas.set_width(canvas.width - 1.5)))
    guard.add_hotkey("ctrl+alt+]", with_canvas(lambda: canvas.set_width(canvas.width + 1.5)))
    guard.add_hotkey("ctrl+alt+s", lambda: export_drawing(cfg, canvas, overlay, desktop))
    guard.add_hotkey("ctrl+alt+r", lambda: worker.record(debug.label_edit.text().strip() or "unlabeled"))
    guard.add_hotkey("ctrl+alt+q", app.quit)
    guard.start()
    if not guard.active:
        log.error("keyboard failsafe could not be installed — refusing to run without it")
        return 3

    worker.start()
    log.info("Vector running. %s", "Calibration first." if worker.calibrator else
             "Raise an open palm and hold still to activate.")
    log.info("hotkeys:\n%s", HOTKEYS_HELP)
    if quit_after > 0:
        from PySide6.QtCore import QTimer
        QTimer.singleShot(int(quit_after * 1000), app.quit)
    try:
        code = app.exec()
    finally:
        summary = metrics.summary()
        log.info("metrics (mean/p95 ms): %s", ", ".join(
            f"{k}={v[0]:.1f}/{v[1]:.1f}" for k, v in sorted(summary.items())))
        log.info("rates: vision %.1f fps, camera %.1f fps, actuator %.0f Hz",
                 metrics.rate("vision"), camera.fps, actuator.stats.tick_hz)
        worker.stop()
        worker.join(timeout=1.0)
        actuator.disarm("shutdown")
        actuator.stop()
        guard.stop()
        camera.stop()
        tracker.close()
    return code


def export_drawing(cfg: Config, canvas, overlay, desktop) -> Path | None:
    """Save the drawing twice: a transparent PNG of just the ink, and a
    composite over a clean screenshot (HUD excluded from the capture)."""
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPainterPath, QPen
    from PySide6.QtCore import Qt

    with canvas.lock:
        strokes = [(s.color, s.width, canvas.smoothed(s)) for s in canvas.strokes if len(s.points) >= 2]
    if not strokes:
        log.info("nothing to export")
        return None
    out_dir = Path(os.path.expanduser(cfg.draw.export_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    b = desktop.bounds

    def ink(painter: QPainter, ox: float, oy: float) -> None:
        painter.setRenderHint(QPainter.Antialiasing)
        for color, width, pts in strokes:
            path = QPainterPath(QPointF(pts[0][0] - ox, pts[0][1] - oy))
            for x, y in pts[1:]:
                path.lineTo(QPointF(x - ox, y - oy))
            painter.setPen(QPen(QColor(color), width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPath(path)

    img = QImage(int(b.w), int(b.h), QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    ink(p, b.x, b.y)
    p.end()
    ink_path = out_dir / f"vector-drawing-{stamp}.png"
    img.save(str(ink_path))

    # Screenshot of the primary monitor without our overlay, then composite.
    WDA_NONE, WDA_EXCLUDEFROMCAPTURE = 0, 0x11
    hwnds = [int(w.winId()) for w in overlay.windows] if overlay else []
    for h in hwnds:
        ctypes.windll.user32.SetWindowDisplayAffinity(h, WDA_EXCLUDEFROMCAPTURE)
    try:
        scr = QGuiApplication.primaryScreen()
        shot = scr.grabWindow(0).toImage().convertToFormat(QImage.Format_ARGB32_Premultiplied)
    finally:
        for h in hwnds:
            ctypes.windll.user32.SetWindowDisplayAffinity(h, WDA_NONE)
    prim = desktop.primary
    shot = shot.scaled(int(prim.rect.w), int(prim.rect.h))
    p = QPainter(shot)
    ink(p, prim.rect.x, prim.rect.y)
    p.end()
    comp_path = out_dir / f"vector-screen-{stamp}.png"
    shot.save(str(comp_path))
    log.info("exported %s and %s", ink_path, comp_path)
    return comp_path


def load_config(path: str | None) -> tuple[Config, Path]:
    p = Path(path) if path else config_mod.default_user_path()
    return config_mod.load(p), p
