"""Turn recorded clips into fixed-length feature sequences for a temporal model.

Per frame (primary hand = the one with the most frames in the clip):
  * 21 world landmarks, wrist-centred, scaled by palm length and rotated so
    the wrist->middle-knuckle axis points up  -> 63 values (pose shape,
    invariant to distance and in-plane rotation)
  * palm velocity in hand-lengths/s (2), palm normal (3), pinch ratio (1),
    finger extension (5)                        -> 11 values (dynamics)
Sequences are resampled on *time* (not frame index) to T steps so clips
recorded at different frame rates line up.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vector.config import Config
from vector.recording import Clip, list_clips, load_clip
from vector.vision import landmarks as L
from vector.vision.features import compute
from vector.vision.tracker import HandIdentity

FEATURES = 63 + 11
DEFAULT_T = 32


def canonical_world(world: np.ndarray) -> np.ndarray:
    w = world - world[L.WRIST]
    palm = np.linalg.norm(w[L.MIDDLE_MCP]) or 1.0
    w = w / palm
    up = w[L.MIDDLE_MCP, :2]
    ang = np.arctan2(up[0], -up[1])            # current roll of the palm axis from -y
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])   # rotate by -ang to undo it
    return w @ rot.T


def clip_sequence(clip: Clip, T: int = DEFAULT_T) -> np.ndarray | None:
    """(T, FEATURES) for the clip's primary hand, or None if no hand."""
    ident = HandIdentity()
    per_track: dict[int, list[tuple[float, np.ndarray]]] = {}
    prev: dict[int, tuple[float, np.ndarray, float]] = {}
    for t, dets in clip.detections():
        for obs in ident.update(dets, t, clip.aspect):
            if obs.stale:
                continue
            f = compute(obs)
            p = prev.get(obs.track_id)
            vel = np.zeros(2)
            if p is not None and t > p[0]:
                vel = (f.palm_center - p[1]) / (t - p[0]) / max(1e-3, p[2])
            prev[obs.track_id] = (t, f.palm_center, f.hand_scale)
            vec = np.concatenate([
                canonical_world(obs.world).ravel(), vel, f.palm_normal, [f.pinch_ratio],
                [f.extension[k] for k in L.FINGER_NAMES]])
            per_track.setdefault(obs.track_id, []).append((t, vec))
    if not per_track:
        return None
    seq = max(per_track.values(), key=len)
    if len(seq) < 3:
        return None
    ts = np.array([s[0] for s in seq])
    xs = np.stack([s[1] for s in seq])
    grid = np.linspace(ts[0], ts[-1], T)
    out = np.empty((T, xs.shape[1]))
    for j in range(xs.shape[1]):
        out[:, j] = np.interp(grid, ts, xs[:, j])
    return out.astype(np.float32)


@dataclass
class Dataset:
    X: np.ndarray             # (N, T, F)
    y: np.ndarray             # (N,)
    labels: list[str]
    sessions: np.ndarray      # (N,) group id for leave-session-out splits
    paths: list[Path]


def load_dataset(root: Path | None = None, T: int = DEFAULT_T, clips: dict | None = None) -> Dataset:
    clips = clips if clips is not None else list_clips(root) if root else list_clips()
    labels = sorted(clips)
    X, y, sess, paths = [], [], [], []
    for li, label in enumerate(labels):
        for path in clips[label]:
            clip = path if isinstance(path, Clip) else load_clip(path)
            seq = clip_sequence(clip, T)
            if seq is None:
                continue
            X.append(seq)
            y.append(li)
            # Session = recording day+hour: a proxy for "same sitting / lighting".
            sess.append(str(clip.meta.get("session") or getattr(path, "name", "x")[:11]))
            paths.append(path if isinstance(path, Path) else Path(label))
    if not X:
        return Dataset(np.zeros((0, T, FEATURES), np.float32), np.zeros(0, int), labels,
                       np.zeros(0), [])
    _, sess_ids = np.unique(np.array(sess), return_inverse=True)
    return Dataset(np.stack(X), np.array(y), labels, sess_ids, paths)


def expected_engine_event(label: str) -> tuple[str, dict] | None:
    """Map a clip label onto the deterministic engine's event, when one exists,
    so both approaches can be scored on the same clips."""
    parts = label.split("_")
    if "swipe" in parts and parts[-1] in ("left", "right"):
        fam = "_".join(parts[: parts.index("swipe")])
        return "swipe", {"family": fam, "direction": parts[-1]}
    if label in ("fist_hold", "three_finger_hold"):
        return "hold", {"pose": label[:-5]}
    if label in ("click", "pinch_click"):
        return "click", {}
    if label.startswith("throw_"):
        return "throw", {}
    if label in ("none", "idle", "background"):
        return None
    return None


def config_for_benchmark() -> Config:
    return Config()
