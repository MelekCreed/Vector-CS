"""The vision worker thread and the shared state the UI reads.

    Camera thread ──latest frame──▶ VisionWorker (this file)
        tracker → identity → features → temporal engine → intent
            ├─ discrete commands ─▶ Actuator queue   (generation-stamped)
            ├─ continuous motion ─▶ Actuator mailbox (latest value)
            └─ UI state ──────────▶ SharedState      (read by the overlay at 60 Hz)
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from vector import config as config_mod
from vector.calibration import Calibrator, Step
from vector.config import Config
from vector.core.metrics import Metrics, Timer
from vector.demo import DemoScript
from vector.desktop.actuator import Actuator
from vector.gestures.events import Snapshot, SystemState
from vector.intent.intent import Feedback, IntentEngine
from vector.overlay.canvas import Canvas
from vector.pipeline import GesturePipeline
from vector.recording import Recorder

log = logging.getLogger(__name__)


@dataclass
class UIState:
    snapshot: Snapshot | None = None
    feedback: Feedback | None = None
    hand_px: list = field(default_factory=list)     # (track_id, primary, palm desktop px)
    frame: np.ndarray | None = None                 # BGR, only while the debug view wants it
    landmarks: list = field(default_factory=list)   # [(label, primary, (21,3) image lms)]
    features: list = field(default_factory=list)
    calibration: object | None = None
    calib_target: tuple | None = None
    demo: object | None = None
    recording: dict | None = None                   # {"label", "state", "remaining"}
    camera_error: str | None = None
    camera_fps: float = 0.0
    frame_index: int = 0


class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._ui = UIState()
        self.want_frame = False

    def publish(self, ui: UIState) -> None:
        with self._lock:
            self._ui = ui

    def read(self) -> UIState:
        with self._lock:
            return self._ui


class VisionWorker(threading.Thread):
    def __init__(self, cfg: Config, camera, tracker, pipeline: GesturePipeline,
                 intent: IntentEngine, actuator: Actuator, shared: SharedState,
                 canvas: Canvas, metrics: Metrics, calibrate: bool = False,
                 demo: bool = False, config_path=None):
        super().__init__(name="vision", daemon=True)
        self.cfg, self.camera, self.tracker = cfg, camera, tracker
        self.pipeline, self.intent, self.actuator = pipeline, intent, actuator
        self.shared, self.canvas, self.metrics = shared, canvas, metrics
        self.config_path = config_path
        self.calibrator = Calibrator(cfg) if calibrate else None
        self.demo = DemoScript() if demo else None
        self.recorder = Recorder()
        self._rec_plan: dict | None = None
        self._requests: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()
        self._running = True
        self.disabled = False           # set by the failsafe hook thread; sticky until re-arm

    # ---------------------------------------------------- cross-thread API
    def request(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` on the vision thread before the next frame."""
        self._requests.put(fn)

    def emergency_stop(self) -> None:
        """Called from the keyboard-hook thread. Flag first, then disarm, so the
        vision thread can never re-arm the actuator in between."""
        self.disabled = True
        self.actuator.disarm("failsafe", latch=True)

    def rearm(self) -> None:
        def _do():
            self.disabled = False
            self.actuator.unlatch()
            nxt = SystemState.SLEEPING if self.cfg.activation.require_activation else SystemState.ACTIVE
            self.pipeline.engine.set_system(nxt, time.perf_counter())
        self.request(_do)

    def start_calibration(self) -> None:
        def _do():
            self.calibrator = Calibrator(self.cfg)
        self.request(_do)

    def toggle_draw(self) -> None:
        self.request(lambda: self.pipeline.engine.toggle_draw(time.perf_counter()))

    def record(self, label: str, countdown_s: float = 3.0, duration_s: float = 2.0) -> None:
        def _do():
            now = time.perf_counter()
            self._rec_plan = {"label": label, "start": now + countdown_s,
                              "end": now + countdown_s + duration_s}
        self.request(_do)

    def stop(self) -> None:
        self._running = False

    # --------------------------------------------------------------- loop
    def run(self) -> None:
        last = 0
        m = self.metrics
        while self._running:
            frame = self.camera.latest(last, timeout=0.5)
            while not self._requests.empty():
                try:
                    self._requests.get_nowait()()
                except Exception:
                    log.exception("request failed")
            if frame is None:
                ui = self.shared.read()
                ui.camera_error = self.camera.error or "waiting for camera"
                self.shared.publish(ui)
                continue
            last = frame.index
            t = frame.t
            m.add("capture_age", (time.perf_counter() - t) * 1000)
            m.add("capture_read", frame.read_ms)
            m.tick("vision")
            try:
                self._process(frame)
            except Exception:
                log.exception("vision frame failed")
            m.add("end_to_end_vision", (time.perf_counter() - t) * 1000)

    def _process(self, frame) -> None:
        m = self.metrics
        t = frame.t
        h, w = frame.image.shape[:2]
        aspect = w / h
        with Timer(m, "inference"):
            dets = self.tracker.detect(frame.image, t)
        with Timer(m, "gesture"):
            events, snap = self.pipeline.process(dets, t, aspect)
        engine = self.pipeline.engine

        self._recording_tick(t, dets, aspect)

        calib_state = None
        if self.calibrator is not None:
            if engine.system == SystemState.ACTIVE:
                engine.set_system(SystemState.SLEEPING, t)
            calib_state = self.calibrator.update(self.pipeline.features, t)
            if calib_state.step == Step.DONE:
                path = config_mod.save(self.cfg, self.config_path)
                log.info("calibration saved to %s: region=%s dominant=%s", path,
                         calib_state.result, self.cfg.dominant_hand)
                self.pipeline.mapper.reset()
                self.calibrator = None
                engine.set_system(SystemState.ACTIVE, t)
            events = []                       # no OS actions while calibrating

        # --- safety sync: the actuator is armed only while ACTIVE and not disabled
        if self.disabled:
            if engine.system != SystemState.DISABLED:
                engine.set_system(SystemState.DISABLED, t)
            if self.actuator.armed:
                self.actuator.disarm()
        elif engine.system == SystemState.ACTIVE and not self.actuator.armed and self.calibrator is None:
            self.actuator.arm()
        elif engine.system != SystemState.ACTIVE and self.actuator.armed:
            self.actuator.disarm("system " + engine.system.value.lower())

        with Timer(m, "intent"):
            fb = self.intent.handle(events, snap)
        if self.actuator.armed and self.intent.last_motion is not None:
            self.actuator.mailbox.put(self.intent.last_motion)
        self._apply_draw_ops(fb.draw_ops)
        if self.demo is not None:
            self.demo.update(events, fb, snap)

        ui = UIState(snapshot=snap, feedback=fb, calibration=calib_state,
                     calib_target=self.calibrator.target_px(self.pipeline.mapper.desktop)
                     if self.calibrator else None,
                     demo=self.demo.view() if self.demo else None,
                     recording=self._rec_view(t), camera_fps=self.camera.fps,
                     frame_index=frame.index)
        mapper = self.pipeline.mapper
        for f in self.pipeline.features:
            palm_norm = np.array([f.palm_center[0] / f.aspect, f.palm_center[1]])
            ui.hand_px.append((f.track_id, f.track_id == engine.primary_id, mapper.to_desktop(palm_norm)))
        ui.features = list(self.pipeline.features)
        if self.shared.want_frame:
            ui.frame = frame.image
            ui.landmarks = [(d.label, d.image) for d in dets]
        self.shared.publish(ui)

    def _apply_draw_ops(self, ops) -> None:
        if not ops:
            return
        cv, dcfg = self.canvas, self.cfg.draw
        with cv.lock:
            self._apply_locked(cv, dcfg, ops)

    @staticmethod
    def _apply_locked(cv, dcfg, ops) -> None:
        for op, p in ops:
            if op == "stroke_begin":
                cv.begin(p)
            elif op == "stroke_point":
                cv.add(p)
            elif op == "stroke_end":
                cv.end()
            elif op == "erase_point":
                cv.erase(p, dcfg.eraser_radius)
            elif op == "draw_undo":
                cv.undo()
            elif op == "draw_clear":
                cv.clear()
            elif op == "draw_color":
                cv.next_color()

    # ---------------------------------------------------------- recording
    def _recording_tick(self, t: float, dets, aspect: float) -> None:
        plan = self._rec_plan
        if plan is None:
            return
        now = time.perf_counter()
        if now >= plan["start"] and not self.recorder.recording:
            self.recorder.start(plan["label"], aspect, {"camera": self.cfg.camera.index})
        if self.recorder.recording:
            self.recorder.add(t, dets, self.pipeline.features)
        if now >= plan["end"]:
            path = self.recorder.stop()
            log.info("recorded clip %s", path)
            plan["saved"] = str(path) if path else None
            plan["done_at"] = now
            self._rec_plan = None
            self._last_saved = (plan["label"], now)

    def _rec_view(self, t: float) -> dict | None:
        plan = self._rec_plan
        now = time.perf_counter()
        if plan is None:
            last = getattr(self, "_last_saved", None)
            if last and now - last[1] < 1.5:
                return {"label": last[0], "state": "saved", "remaining": 0.0}
            return None
        if now < plan["start"]:
            return {"label": plan["label"], "state": "countdown", "remaining": plan["start"] - now}
        return {"label": plan["label"], "state": "recording", "remaining": plan["end"] - now}
