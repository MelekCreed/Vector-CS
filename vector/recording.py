"""Gesture recording (dataset mode) and replay.

A clip stores, per frame: timestamp, the raw detections for every hand
(image + world landmarks, handedness), and derived features for convenience
(velocity, palm normal, pinch ratio, finger extension). Raw detections make a
clip *replayable* through the full deterministic pipeline, which is how the
learned model and the heuristic engine get benchmarked on identical data.

Only landmarks are stored — never camera images. Files live under
``data/gestures/<label>/`` which is git-ignored.
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vector.vision.features import HandFeatures
from vector.vision.tracker import Detection

DATA_DIR = Path("data") / "gestures"


@dataclass
class Clip:
    label: str
    aspect: float
    frames: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def detections(self):
        """Yield (t, [Detection]) for replay."""
        for fr in self.frames:
            dets = [Detection(np.array(h["image"]), np.array(h["world"]), h["label"], h["score"])
                    for h in fr["hands"]]
            yield fr["t"], dets


class Recorder:
    def __init__(self, root: Path | None = None):
        self.root = root or DATA_DIR
        self.clip: Clip | None = None
        self.t0 = 0.0

    @property
    def recording(self) -> bool:
        return self.clip is not None

    def start(self, label: str, aspect: float, meta: dict | None = None) -> None:
        self.clip = Clip(label, aspect, meta=dict(meta or {}, started=time.time()))
        self.t0 = -1.0

    def add(self, t: float, dets: list[Detection], feats: list[HandFeatures]) -> None:
        if self.clip is None:
            return
        if self.t0 < 0:
            self.t0 = t
        hands = [{"image": np.round(d.image, 5).tolist(), "world": np.round(d.world, 5).tolist(),
                  "label": d.label, "score": round(float(d.score), 4)} for d in dets]
        derived = [{"track": f.track_id, "label": f.label,
                    "velocity": np.round(f.velocity, 4).tolist(),
                    "palm_normal": np.round(f.palm_normal, 4).tolist(),
                    "pinch_ratio": round(f.pinch_ratio, 4),
                    "extension": {k: round(v, 3) for k, v in f.extension.items()}}
                   for f in feats]
        self.clip.frames.append({"t": round(t - self.t0, 4), "hands": hands, "features": derived})

    def stop(self, save: bool = True) -> Path | None:
        clip, self.clip = self.clip, None
        if not save or clip is None or not clip.frames:
            return None
        return save_clip(clip, self.root)


def save_clip(clip: Clip, root: Path = DATA_DIR) -> Path:
    d = root / _safe(clip.label)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"label": clip.label, "aspect": clip.aspect, "meta": clip.meta}) + "\n")
        for fr in clip.frames:
            fh.write(json.dumps(fr) + "\n")
    return path


def load_clip(path: Path) -> Clip:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        head = json.loads(fh.readline())
        frames = [json.loads(line) for line in fh if line.strip()]
    return Clip(head["label"], head["aspect"], frames, head.get("meta", {}))


def list_clips(root: Path = DATA_DIR) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    if root.exists():
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            out[d.name] = sorted(d.glob("*.jsonl.gz"))
    return out


def _safe(label: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in label.strip().lower()) or "unlabeled"


def replay(clip: Clip, cfg, desktop, active: bool = True):
    """Run a clip through a fresh deterministic pipeline; returns events."""
    from vector.gestures.events import SystemState
    from vector.pipeline import GesturePipeline
    pipe = GesturePipeline(cfg, desktop)
    if active:
        pipe.engine.system = SystemState.ACTIVE
        pipe.engine.neutral_required = False
    events = []
    for t, dets in clip.detections():
        ev, _ = pipe.process(dets, t, clip.aspect)
        events.extend(ev)
    return events
