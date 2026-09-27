"""Pose layer: continuous per-frame pose scores + temporal pose tracking.

Per-frame scores are *evidence*, not decisions. The PoseTracker smooths them
and only switches pose after the candidate has dominated for a minimum
*duration* (time-based, so it behaves the same at 20 or 60 fps), with
separate enter/exit thresholds so poses don't flicker at the boundary.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np

from vector.vision.features import HandFeatures


class Pose(str, enum.Enum):
    NONE = "none"
    POINT = "point"
    PINCH = "pinch"
    MIDDLE_PINCH = "middle_pinch"
    OPEN_PALM = "open_palm"
    FIST = "fist"
    TWO = "two"
    THREE = "three"
    SHAKA = "shaka"


def _gm(*xs: float) -> float:
    """Geometric mean: one weak factor drags the score down (AND-like), but
    unlike min() it still rewards the other factors being strong."""
    acc = 0.0
    for x in xs:
        acc += math.log(min(1.0, max(1e-4, x)))
    return math.exp(acc / len(xs))


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 at lo, 1 at hi (works for lo > hi as a descending ramp)."""
    v = (x - lo) / (hi - lo)
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def pinch_closeness(ratio: float) -> float:
    return _ramp(ratio, 0.48, 0.12)


def pose_scores(f: HandFeatures) -> dict[Pose, float]:
    e = f.extension
    ext = {k: float(v) for k, v in e.items()}
    cur = {k: 1.0 - v for k, v in ext.items()}
    pinch = pinch_closeness(f.pinch_ratio)
    mpinch = pinch_closeness(f.middle_pinch_ratio)
    not_pinching = 1.0 - max(pinch, mpinch)
    thumb_in = 1.0 - 0.7 * ext["thumb"]

    s = {
        Pose.POINT: _gm(ext["index"], cur["middle"], cur["ring"], cur["pinky"], not_pinching),
        # Index and middle pinches are mutually exclusive: whichever finger is
        # closer to the thumb wins, and the other's score is suppressed.
        Pose.PINCH: pinch * (1.0 if f.pinch_ratio <= f.middle_pinch_ratio else 0.2),
        Pose.MIDDLE_PINCH: mpinch * _ramp(ext["index"], 0.4, 0.75)
        * (1.0 if f.middle_pinch_ratio < f.pinch_ratio else 0.1),
        Pose.OPEN_PALM: _gm(ext["thumb"] * 0.5 + 0.5, ext["index"], ext["middle"], ext["ring"],
                            ext["pinky"], not_pinching),
        Pose.FIST: _gm(cur["index"], cur["middle"], cur["ring"], cur["pinky"], thumb_in),
        Pose.TWO: _gm(ext["index"], ext["middle"], cur["ring"], cur["pinky"], not_pinching),
        Pose.THREE: _gm(ext["index"], ext["middle"], ext["ring"], cur["pinky"], not_pinching),
        Pose.SHAKA: _gm(ext["thumb"], cur["index"], cur["middle"], cur["ring"], ext["pinky"]),
    }
    return s


@dataclass
class PoseState:
    pose: Pose
    score: float          # smoothed score of the current pose
    since: float          # when the current pose was confirmed
    scores: dict[Pose, float]

    def held(self, t: float) -> float:
        return t - self.since


class PoseTracker:
    """Temporal pose decision with EMA smoothing, hysteresis and time-based
    confirmation. PINCH is excluded here — it has its own low-latency latch."""

    TRACKED = (Pose.POINT, Pose.OPEN_PALM, Pose.FIST, Pose.TWO, Pose.THREE, Pose.SHAKA,
               Pose.MIDDLE_PINCH)

    def __init__(self, alpha: float = 0.45, enter: float = 0.62, exit: float = 0.42,
                 confirm_s: float = 0.09, margin: float = 0.08):
        self.alpha, self.enter, self.exit = alpha, enter, exit
        self.confirm_s, self.margin = confirm_s, margin
        self.smoothed: dict[Pose, float] = {p: 0.0 for p in self.TRACKED}
        self.pose = Pose.NONE
        self.since = 0.0
        self._cand: Pose | None = None
        self._cand_since = 0.0

    def update(self, raw: dict[Pose, float], t: float) -> PoseState:
        for p in self.TRACKED:
            self.smoothed[p] += self.alpha * (raw.get(p, 0.0) - self.smoothed[p])
        best = max(self.TRACKED, key=lambda p: self.smoothed[p])
        best_s = self.smoothed[best]
        cur_s = self.smoothed.get(self.pose, 0.0) if self.pose != Pose.NONE else 0.0

        # Exit the current pose only once it falls below the exit threshold.
        if self.pose != Pose.NONE and cur_s < self.exit:
            self.pose, self.since = Pose.NONE, t

        challenger = (best != self.pose and best_s >= self.enter
                      and best_s >= cur_s + self.margin)
        if challenger:
            if self._cand != best:
                self._cand, self._cand_since = best, t
            elif t - self._cand_since >= self.confirm_s:
                self.pose, self.since = best, t
                self._cand = None
        else:
            self._cand = None
        return PoseState(self.pose, self.smoothed.get(self.pose, 0.0) if self.pose != Pose.NONE
                         else 0.0, self.since, dict(self.smoothed))

    def reset(self, t: float = 0.0) -> None:
        for p in self.smoothed:
            self.smoothed[p] = 0.0
        self.pose, self.since, self._cand = Pose.NONE, t, None


class PinchLatch:
    """Pinch needs lower latency than other poses (it's a click), so it uses a
    direct hysteresis latch on the scale-invariant pinch ratio plus a very
    short time confirmation to reject single-frame landmark glitches."""

    def __init__(self, enter: float = 0.28, exit: float = 0.42, confirm_s: float = 0.03,
                 release_confirm_s: float = 0.03):
        self.enter, self.exit = enter, exit
        self.confirm_s, self.release_confirm_s = confirm_s, release_confirm_s
        self.on = False
        self.since = 0.0
        self._pending_since: float | None = None

    def update(self, ratio: float, t: float, allowed: bool = True) -> bool:
        want = (ratio < self.enter) if not self.on else (ratio < self.exit)
        if not allowed and not self.on:
            want = False
        if want != self.on:
            if self._pending_since is None:
                self._pending_since = t
            need = self.confirm_s if want else self.release_confirm_s
            if t - self._pending_since >= need:
                self.on, self.since, self._pending_since = want, t, None
        else:
            self._pending_since = None
        return self.on

    def force_off(self, t: float) -> None:
        self.on, self.since, self._pending_since = False, t, None
