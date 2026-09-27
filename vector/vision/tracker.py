"""MediaPipe HandLandmarker wrapper + persistent hand identities.

MediaPipe returns hands in arbitrary order and its handedness label flickers
on ambiguous poses. Downstream state machines need stable identities, so we
associate detections to tracks by proximity and treat handedness as a vote
accumulated over time rather than a per-frame fact.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vector.config import TrackerConfig
from vector.vision import landmarks as L
from vector.vision.features import HandObservation

log = logging.getLogger(__name__)


@dataclass
class Detection:
    """Raw per-frame detection (tracker-agnostic, also produced by tests/replay)."""
    image: np.ndarray       # (21,3)
    world: np.ndarray       # (21,3)
    label: str              # MediaPipe's label for this frame
    score: float


@dataclass
class _Track:
    id: int
    palm: np.ndarray        # (2,) normalised image coords
    p_right: float          # accumulated belief the hand is the user's right hand
    last_seen: float
    last: Detection
    t_first: float

    @property
    def label(self) -> str:
        return "Right" if self.p_right >= 0.5 else "Left"


class HandIdentity:
    """Greedy nearest-neighbour association with sticky handedness votes."""

    def __init__(self, max_jump: float = 0.25, vote_rate: float = 0.15, grace_s: float = 0.25):
        self.max_jump = max_jump
        self.vote_rate = vote_rate
        self.grace_s = grace_s
        self._tracks: list[_Track] = []
        self._ids = itertools.count(1)

    def update(self, dets: list[Detection], t: float, aspect: float) -> list[HandObservation]:
        palms = [d.image[list(L.PALM), :2].mean(axis=0) for d in dets]
        pairs = sorted(
            ((float(np.linalg.norm(p - tr.palm)), i, j)
             for i, p in enumerate(palms) for j, tr in enumerate(self._tracks)),
            key=lambda x: x[0])
        used_d, used_t, matched = set(), set(), {}
        for dist, i, j in pairs:
            if dist > self.max_jump or i in used_d or j in used_t:
                continue
            matched[i] = j
            used_d.add(i)
            used_t.add(j)

        for i, d in enumerate(dets):
            vote = d.score if d.label == "Right" else 1.0 - d.score
            if i in matched:
                tr = self._tracks[matched[i]]
                tr.palm, tr.last, tr.last_seen = palms[i], d, t
                tr.p_right += self.vote_rate * (vote - tr.p_right)
            else:
                self._tracks.append(_Track(next(self._ids), palms[i], vote, t, d, t))

        # Drop tracks past their grace period.
        self._tracks = [tr for tr in self._tracks if t - tr.last_seen <= self.grace_s]
        self._resolve_duplicate_labels()

        out = []
        for tr in self._tracks:
            stale = tr.last_seen < t
            out.append(HandObservation(
                track_id=tr.id, label=tr.label, label_score=0.0 if stale else tr.last.score,
                image=tr.last.image, world=tr.last.world, t=t, aspect=aspect, stale=stale,
                label_confidence=abs(tr.p_right - 0.5) * 2, age_s=t - tr.t_first))
        return out

    def _resolve_duplicate_labels(self) -> None:
        """Two simultaneous hands can't both be 'Right'. Keep the more certain
        one and flip the other; ties are broken by image position (the user's
        right hand appears on the right of the mirrored image)."""
        if len(self._tracks) != 2:
            return
        a, b = self._tracks
        if a.label != b.label:
            return
        ca, cb = abs(a.p_right - 0.5), abs(b.p_right - 0.5)
        if abs(ca - cb) < 0.05:
            right, left = (a, b) if a.palm[0] > b.palm[0] else (b, a)
        else:
            sure, unsure = (a, b) if ca > cb else (b, a)
            if sure.label == "Right":
                right, left = sure, unsure
            else:
                left, right = sure, unsure
        right.p_right = max(right.p_right, 0.55)
        left.p_right = min(left.p_right, 0.45)

    def reset(self) -> None:
        self._tracks.clear()


class MediaPipeTracker:
    def __init__(self, cfg: TrackerConfig, root: Path | None = None):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        path = Path(cfg.model_path)
        if not path.is_absolute():
            path = (root or Path.cwd()) / path
        if not path.exists():
            raise FileNotFoundError(
                f"hand model not found at {path}. Run: python scripts/fetch_model.py")
        self._mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg.max_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence)
        self._landmarker = vision.HandLandmarker.create_from_options(opts)
        self._last_ts = -1

    def detect(self, bgr: np.ndarray, t: float) -> list[Detection]:
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(t * 1000), self._last_ts + 1)    # VIDEO mode needs strictly increasing ms
        self._last_ts = ts
        res = self._landmarker.detect_for_video(img, ts)
        dets = []
        for lms, wlms, hd in zip(res.hand_landmarks, res.hand_world_landmarks, res.handedness):
            dets.append(Detection(
                image=np.array([[p.x, p.y, p.z] for p in lms], dtype=float),
                world=np.array([[p.x, p.y, p.z] for p in wlms], dtype=float),
                label=hd[0].category_name, score=float(hd[0].score)))
        return dets

    def close(self) -> None:
        self._landmarker.close()
