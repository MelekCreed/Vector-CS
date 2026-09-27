"""Threaded webcam capture that always hands out the *newest* frame.

OpenCV's internal buffer otherwise queues frames, and a consumer that is even
slightly slower than the camera ends up processing stale images — the single
biggest source of 'laggy webcam demo' feel.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from vector.config import CameraConfig

log = logging.getLogger(__name__)

_BACKENDS = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW, "any": cv2.CAP_ANY}


@dataclass
class Frame:
    image: np.ndarray        # BGR, already mirrored if configured
    t: float                 # perf_counter() when the frame was received
    index: int
    read_ms: float           # time spent inside cap.read()


class Camera:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Condition()
        self._frame: Frame | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self.fps = 0.0
        self.error: str | None = None

    def open(self) -> None:
        backend = _BACKENDS.get(self.cfg.backend, cv2.CAP_ANY)
        cap = cv2.VideoCapture(self.cfg.index, backend)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera {self.cfg.index} ({self.cfg.backend})")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        log.info("camera %d opened %dx%d @%.0f", self.cfg.index,
                 cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                 cap.get(cv2.CAP_PROP_FPS))

    def start(self) -> "Camera":
        if self._cap is None:
            self.open()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        idx, count, t_fps = 0, 0, time.perf_counter()
        failures = 0
        while self._running:
            t0 = time.perf_counter()
            ok, img = self._cap.read()
            t1 = time.perf_counter()
            if not ok or img is None:
                failures += 1
                if failures > 30:
                    self.error = "camera stopped delivering frames"
                    log.error(self.error)
                    failures = 0
                time.sleep(0.01)
                continue
            failures = 0
            if self.cfg.mirror:
                img = cv2.flip(img, 1)
            idx += 1
            with self._lock:
                self._frame = Frame(img, t1, idx, (t1 - t0) * 1000)
                self._lock.notify_all()
            count += 1
            if t1 - t_fps >= 1.0:
                self.fps = count / (t1 - t_fps)
                count, t_fps = 0, t1

    def latest(self, after_index: int = 0, timeout: float = 0.5) -> Frame | None:
        """Block until a frame newer than ``after_index`` exists."""
        with self._lock:
            self._lock.wait_for(lambda: self._frame is not None and self._frame.index > after_index
                                or not self._running, timeout=timeout)
            f = self._frame
        return f if f is not None and f.index > after_index else None

    def stop(self) -> None:
        self._running = False
        with self._lock:
            self._lock.notify_all()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
