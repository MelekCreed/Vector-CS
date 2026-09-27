"""First-launch calibration.

Steps (driven from the vision thread, rendered by the overlay):
  1. HAND     — raise the hand you want to point with (sets dominant hand)
  2. TARGETS  — point at four on-screen targets and hold still (dwell)
  3. DONE     — derive the comfortable camera region + typical hand size

The four targets sit at 20% / 80% of the desktop. Where your fingertip was
when pointing at them tells us where 20% and 80% live in *camera* space,
which we extrapolate to the full 0..100% region. This captures camera
position/tilt, reach and preferred movement size in one short ritual.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

from vector.config import Config

TARGET_FRACS = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
DWELL_S = 0.9
STILL_HL_S = 0.35          # fingertip speed (hand-lengths/s) that counts as still
HAND_S = 1.2


class Step(str, enum.Enum):
    HAND = "hand"
    TARGETS = "targets"
    DONE = "done"


def region_from_samples(samples: list[tuple[float, float]], margin: float = 0.0
                        ) -> tuple[float, float, float, float]:
    """Samples: normalised fingertip positions for TARGET_FRACS (in order).
    Returns (x0, y0, x1, y1) such that the samples map back onto the targets."""
    s = np.asarray(samples, dtype=float)
    f = np.asarray(TARGET_FRACS, dtype=float)
    out = []
    for axis in (0, 1):
        # Least-squares line: camera = a + b * desktop_frac (robust to a sloppy corner).
        A = np.column_stack([np.ones(len(f)), f[:, axis]])
        (a, b), *_ = np.linalg.lstsq(A, s[:, axis], rcond=None)
        if abs(b) < 0.05:
            raise ValueError("targets too close together in camera space; move more")
        lo, hi = a, a + b
        out.append((min(lo, hi) - margin, max(lo, hi) + margin))
    (x0, x1), (y0, y1) = out
    return (float(np.clip(x0, 0, 0.95)), float(np.clip(y0, 0, 0.95)),
            float(np.clip(x1, x0 + 0.05, 1)), float(np.clip(y1, y0 + 0.05, 1)))


@dataclass
class CalibrationState:
    step: Step = Step.HAND
    target: int = 0
    progress: float = 0.0
    message: str = "Raise the hand you want to point with"
    samples: list = field(default_factory=list)
    dominant: str | None = None
    hand_scales: list = field(default_factory=list)
    result: tuple | None = None


class Calibrator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = CalibrationState()
        self._since: float | None = None
        self._buf: list[np.ndarray] = []

    def target_px(self, desktop) -> tuple[float, float] | None:
        if self.state.step != Step.TARGETS:
            return None
        u, v = TARGET_FRACS[self.state.target]
        return desktop.from_unit(u, v)

    def update(self, feats, t: float) -> CalibrationState:
        st = self.state
        if st.step == Step.DONE:
            return st
        if st.step == Step.HAND:
            if len(feats) == 1 and not feats[0].stale:
                self._since = self._since if self._since is not None else t
                st.progress = min(1.0, (t - self._since) / HAND_S)
                if st.progress >= 1.0:
                    st.dominant = feats[0].label
                    st.step, st.message = Step.TARGETS, "Point at the target and hold still"
                    self._since, st.progress = None, 0.0
            else:
                self._since, st.progress = None, 0.0
                if len(feats) > 1:
                    st.message = "Just one hand, please"
            return st
        # TARGETS
        hand = next((f for f in feats if f.label == st.dominant and not f.stale), None)
        if hand is None and len(feats) == 1:
            hand = feats[0]
        if hand is None:
            self._since, st.progress, self._buf = None, 0.0, []
            return st
        tip = np.array([hand.index_tip[0] / hand.aspect, hand.index_tip[1]])
        still = hand.speed < STILL_HL_S if hand.speed else True
        if still:
            if self._since is None:
                self._since, self._buf = t, []
            self._buf.append(tip)
            st.progress = min(1.0, (t - self._since) / DWELL_S)
            if st.progress >= 1.0:
                st.samples.append(tuple(np.median(np.array(self._buf), axis=0)))
                st.hand_scales.append(hand.hand_scale)
                st.target += 1
                self._since, st.progress, self._buf = None, 0.0, []
                if st.target >= len(TARGET_FRACS):
                    self._finish()
        else:
            self._since, st.progress, self._buf = None, 0.0, []
        return st

    def _finish(self) -> None:
        st = self.state
        try:
            st.result = region_from_samples(st.samples)
        except ValueError as exc:
            st.samples, st.target, st.message = [], 0, f"{exc}. Let's try again."
            return
        c = self.cfg.cursor
        c.region_x0, c.region_y0, c.region_x1, c.region_y1 = st.result
        self.cfg.dominant_hand = st.dominant or self.cfg.dominant_hand
        self.cfg.calibrated = True
        st.step, st.message = Step.DONE, "Calibrated"
