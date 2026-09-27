"""Detections -> identities -> features -> temporal gesture engine.

Pure and synchronous: the live app calls it from the vision thread, tests and
replays call it directly with synthetic or recorded detections.
"""

from __future__ import annotations

from vector.config import Config
from vector.cursor.mapper import CursorMapper
from vector.desktop.monitors import VirtualDesktop
from vector.gestures.engine import GestureEngine
from vector.gestures.events import GestureEvent, Snapshot
from vector.vision.features import HandFeatures, compute
from vector.vision.tracker import Detection, HandIdentity


class GesturePipeline:
    def __init__(self, cfg: Config, desktop: VirtualDesktop):
        self.cfg = cfg
        self.identity = HandIdentity(grace_s=cfg.tracker.lost_grace_s)
        self.mapper = CursorMapper(cfg.cursor, desktop)
        self.engine = GestureEngine(cfg, self.mapper)
        self.features: list[HandFeatures] = []

    def process(self, dets: list[Detection], t: float, aspect: float
                ) -> tuple[list[GestureEvent], Snapshot]:
        obs = self.identity.update(dets, t, aspect)
        self.features = [compute(o) for o in obs]
        return self.engine.update(self.features, t)
