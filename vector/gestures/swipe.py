"""Horizontal swipe detection with anti-double-fire logic.

A swipe is a *stroke*: it starts when horizontal speed rises, and is accepted
only if, within ``max_duration``, it covers ``min_distance`` hand-lengths,
peaks above ``min_speed``, and stays horizontally dominant. After a swipe:
* a cooldown blocks any new swipe, and
* the *opposite* direction stays locked out longer, because bringing your
  hand back to the start position is itself a fast horizontal movement.
The detector must also see the hand slow down (re-arm) before firing again.

Intent structure: a deliberate swipe is *pause -> flick*. Strokes may only
begin within ``still_window`` seconds of the hand being nearly still, which
rejects the continuous wandering of a hand that is just moving around.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vector.gestures.confidence import Confidence, combine, velocity_margin


@dataclass
class SwipeResult:
    direction: str            # "left" | "right"
    confidence: Confidence
    distance: float
    peak_speed: float


class SwipeDetector:
    def __init__(self, min_speed=2.2, min_distance=0.9, max_duration=0.6, direction_ratio=1.8,
                 cooldown=0.7, return_lockout=1.1, rearm_speed=0.8, still_speed=1.0,
                 still_window=0.35, still_min_s=0.1):
        self.min_speed, self.min_distance = min_speed, min_distance
        self.max_duration, self.direction_ratio = max_duration, direction_ratio
        self.cooldown, self.return_lockout, self.rearm_speed = cooldown, return_lockout, rearm_speed
        self.still_speed, self.still_window, self.still_min_s = still_speed, still_window, still_min_s
        self.reset()

    def reset(self) -> None:
        self._start_t: float | None = None
        self._start_p: np.ndarray | None = None
        self._peak = 0.0
        self._last_dir: str | None = None
        self._last_t = -1e9
        self._armed = True
        self._samples: list[tuple[float, np.ndarray]] = []
        self._min_geometry = 1.0
        self._last_still = -1e9         # last time the hand had been still for >= still_min_s
        self._still_since: float | None = None

    def cancel_stroke(self) -> None:
        self._start_t = None
        self._samples = []

    def update(self, pos: np.ndarray, vel: np.ndarray, t: float,
               geometry: float = 1.0, tracking: float = 1.0,
               can_start: bool = True) -> SwipeResult | None:
        """``pos`` in hand-lengths (any origin), ``vel`` in hand-lengths/s."""
        speed_x = abs(float(vel[0]))
        # A *sustained* pause counts; the instantaneous zero-velocity of a
        # direction reversal (waving, wandering) does not.
        if float(np.linalg.norm(vel)) < self.still_speed:
            if self._still_since is None:
                self._still_since = t
            if t - self._still_since >= self.still_min_s:
                self._last_still = t
        else:
            self._still_since = None
        if not self._armed:
            if float(np.linalg.norm(vel)) < self.rearm_speed:
                self._armed = True
            return None
        if self._start_t is None:
            if (can_start and speed_x > self.min_speed * 0.45
                    and t - self._last_still <= self.still_window):
                self._start_t, self._start_p, self._peak = t, pos.copy(), speed_x
                self._samples = [(t, pos.copy())]
                self._min_geometry = geometry
            return None

        self._samples.append((t, pos.copy()))
        self._peak = max(self._peak, speed_x)
        self._min_geometry = min(self._min_geometry, geometry)
        dur = t - self._start_t
        d = pos - self._start_p
        dx, dy = float(d[0]), float(d[1])

        if dur > self.max_duration:
            self.cancel_stroke()           # too slow: that's just moving the hand
            return None
        if speed_x < self.min_speed * 0.25 and abs(dx) < self.min_distance:
            self.cancel_stroke()           # stroke fizzled out
            return None
        if abs(dx) < self.min_distance:
            return None
        if abs(dx) < self.direction_ratio * abs(dy):
            self.cancel_stroke()           # diagonal / vertical: not a swipe
            return None
        if self._peak < self.min_speed:
            return None

        direction = "right" if dx > 0 else "left"
        since = t - self._last_t
        if since < self.cooldown:
            self.cancel_stroke()
            return None
        if self._last_dir and direction != self._last_dir and since < self.return_lockout:
            self.cancel_stroke()           # the return stroke of the previous swipe
            self._armed = False
            return None

        # Direction consistency: fraction of steps that moved the right way.
        steps = np.diff(np.array([p[0] for _, p in self._samples]))
        consistent = float((np.sign(steps) == np.sign(dx)).mean()) if len(steps) else 1.0
        # The pose must stay crisp for the *whole* stroke: a pose dissolving
        # mid-motion is the signature of a hand moving between poses.
        geom = float(np.clip((self._min_geometry - 0.55) / 0.35, 0.0, 1.0))
        conf = combine({
            "geometry": geom,
            "velocity": velocity_margin(self._peak, self.min_speed),
            "direction": min(1.0, consistent * min(1.0, abs(dx) / (self.direction_ratio * abs(dy) + 1e-6))),
            "duration": 1.0 if dur > 0.05 else 0.6,
            "tracking": tracking,
        })
        res = SwipeResult(direction, conf, abs(dx), self._peak)
        self._last_dir, self._last_t = direction, t
        self._armed = False
        self.cancel_stroke()
        return res
