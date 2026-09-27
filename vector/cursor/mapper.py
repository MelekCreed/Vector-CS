"""Spatial cursor: hand position -> desktop pixels that feel like a trackpad.

Pipeline (per camera frame):
  1. One-Euro filter on the hand point (adaptive jitter suppression)
  2. calibrated region -> unit square -> virtual desktop (multi-monitor aware)
  3. hybrid mode: speed-dependent gain on deltas (precise when slow, fast when
     flicking) + drift correction toward the absolute mapping *only while
     moving and not engaged*, so the cursor never creeps on its own
  4. rope dead-zone: output trails the target by at most N px, killing
     sub-pixel shimmer without the 'sticky then jump' feel of a hard dead zone
  5. source-switch blending: changing the tracked point (index tip -> palm
     while pinching) never makes the cursor jump
The actuator then adds a 120 Hz critically damped spring between frames.
"""

from __future__ import annotations

import math

import numpy as np

from vector.config import CursorConfig
from vector.core.filters import History, OneEuroFilter
from vector.desktop.monitors import VirtualDesktop


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


class CursorMapper:
    def __init__(self, cfg: CursorConfig, desktop: VirtualDesktop):
        self.cfg = cfg
        self.desktop = desktop
        self.filter = OneEuroFilter(cfg.min_cutoff, cfg.beta, cfg.d_cutoff)
        self.history: History[np.ndarray] = History(span_s=1.0)
        self._out: np.ndarray | None = None       # pre-dead-zone position
        self._emit: np.ndarray | None = None      # emitted position
        self._prev_abs: np.ndarray | None = None
        self._prev_t: float | None = None
        self._source: str | None = None
        self._offset = np.zeros(2)                # source-switch blend offset (iso units)
        self._last_raw: np.ndarray | None = None
        self.speed_px_s = 0.0

    # -- mapping -----------------------------------------------------------
    def region(self) -> tuple[float, float, float, float]:
        """Calibrated camera region, shrunk around its centre by sensitivity.
        Region is in *normalised image* coords (0..1) for both axes."""
        c = self.cfg
        cx, cy = (c.region_x0 + c.region_x1) / 2, (c.region_y0 + c.region_y1) / 2
        hw = (c.region_x1 - c.region_x0) / 2 / max(0.2, c.sensitivity)
        hh = (c.region_y1 - c.region_y0) / 2 / max(0.2, c.sensitivity)
        return cx - hw, cy - hh, cx + hw, cy + hh

    def to_desktop(self, p_norm: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.region()
        u = (p_norm[0] - x0) / (x1 - x0)
        v = (p_norm[1] - y0) / (y1 - y0)
        return np.array(self.desktop.from_unit(min(1, max(0, u)), min(1, max(0, v))))

    # -- update ------------------------------------------------------------
    def update(self, p_norm, t: float, source: str = "index", engaged: bool = False) -> np.ndarray:
        """``p_norm``: hand point in normalised image coords (x,y in 0..1)."""
        raw = np.asarray(p_norm, dtype=float)[:2]
        if self._source is not None and source != self._source and self._last_raw is not None:
            # Keep the *effective* input continuous across the switch, then let
            # the difference decay away.
            self._offset = self._offset + (self._last_raw - raw)
        self._source = source
        self._last_raw = raw.copy()
        dt = 1 / 30 if self._prev_t is None else max(1e-3, t - self._prev_t)
        self._offset *= math.exp(-dt / 0.12)

        p = self.filter(raw + self._offset, t)
        abs_px = self.to_desktop(p)
        c = self.cfg
        if self._out is None or c.mode == "absolute":
            out = abs_px
        else:
            delta = abs_px - self._prev_abs
            speed = float(np.linalg.norm(delta)) / dt / self.desktop.primary.rect.h
            g = c.gain_slow + (c.gain_fast - c.gain_slow) * _smoothstep(
                (speed - c.speed_slow) / (c.speed_fast - c.speed_slow))
            out = self._out + delta * g
            if not engaged:
                k = min(1.0, c.drift_correction * dt * min(1.0, speed / c.speed_slow))
                out = out + (abs_px - out) * k
            out = np.array(self.desktop.clamp(*out))
        self._prev_abs = abs_px
        self._out = out

        if self._emit is None:
            self._emit = out.copy()
        else:
            d = out - self._emit
            n = float(np.linalg.norm(d))
            if n > c.dead_zone_px:
                self._emit = out - d / n * c.dead_zone_px
        prev_t = self._prev_t
        self._prev_t = t
        emitted = self._emit.copy()
        if self.history.latest is not None and prev_t is not None:
            self.speed_px_s = float(np.linalg.norm(emitted - self.history.latest.value)) / dt
        self.history.push(t, emitted)
        return emitted

    def position_at(self, t: float) -> np.ndarray | None:
        v = self.history.at(t)
        return None if v is None else np.asarray(v).copy()

    @property
    def position(self) -> np.ndarray | None:
        return None if self._emit is None else self._emit.copy()

    def reset(self) -> None:
        self.filter.reset()
        self._out = self._emit = self._prev_abs = self._prev_t = None
        self._source, self._last_raw = None, None
        self._offset = np.zeros(2)
        self.history.clear()
