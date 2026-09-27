"""Signal-processing primitives. All take explicit timestamps (seconds) so they
are deterministic under test and robust to irregular camera frame timing."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Generic, Iterator, TypeVar

import numpy as np

T = TypeVar("T")


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Casiez et al. 2012. Adaptive low-pass: heavy smoothing when the signal is
    slow (kills jitter at rest), light smoothing when fast (keeps latency low).
    Works on scalars or numpy vectors."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._x = self._dx = self._t = None

    @property
    def value(self) -> np.ndarray | None:
        return self._x

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._x is None or self._t is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return self._x.copy()
        dt = t - self._t
        if dt <= 1e-6:                      # duplicate timestamp: nothing new to learn
            return self._x.copy()
        self._t = t
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        speed = float(np.linalg.norm(self._dx))
        cutoff = self.min_cutoff + self.beta * speed
        a = _alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        return self._x.copy()


@dataclass
class Sample(Generic[T]):
    t: float
    value: T


class History(Generic[T]):
    """Time-bounded ring buffer of timestamped samples."""

    def __init__(self, span_s: float = 1.0, maxlen: int = 512):
        self.span_s = span_s
        self._buf: deque[Sample[T]] = deque(maxlen=maxlen)

    def push(self, t: float, value: T) -> None:
        self._buf.append(Sample(t, value))
        while self._buf and t - self._buf[0].t > self.span_s:
            self._buf.popleft()

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    def __iter__(self) -> Iterator[Sample[T]]:
        return iter(self._buf)

    @property
    def latest(self) -> Sample[T] | None:
        return self._buf[-1] if self._buf else None

    def since(self, t0: float) -> list[Sample[T]]:
        return [s for s in self._buf if s.t >= t0]

    def at(self, t: float) -> T | None:
        """Most recent value at or before ``t`` (falls back to the oldest)."""
        best = None
        for s in self._buf:
            if s.t <= t:
                best = s
            else:
                break
        if best is None:
            return self._buf[0].value if self._buf else None
        return best.value


def regression_velocity(samples: list[Sample[np.ndarray]]) -> np.ndarray:
    """Least-squares slope of position vs time. Far less noisy than a two-point
    finite difference, and unbiased for constant velocity."""
    if len(samples) < 2:
        dim = len(samples[0].value) if samples else 2
        return np.zeros(dim)
    t = np.array([s.t for s in samples])
    x = np.stack([np.asarray(s.value, dtype=float) for s in samples])
    t = t - t.mean()
    denom = float((t * t).sum())
    if denom < 1e-9:
        return np.zeros(x.shape[1])
    return (t[:, None] * (x - x.mean(axis=0))).sum(axis=0) / denom


class VelocityEstimator:
    """Windowed regression velocity over the last ``window_s`` seconds."""

    def __init__(self, window_s: float = 0.1):
        self.window_s = window_s
        self.history: History[np.ndarray] = History(span_s=max(1.0, window_s * 4))

    def reset(self) -> None:
        self.history.clear()

    def update(self, pos, t: float) -> np.ndarray:
        self.history.push(t, np.asarray(pos, dtype=float))
        return self.velocity(t)

    def velocity(self, t: float, window_s: float | None = None) -> np.ndarray:
        w = self.window_s if window_s is None else window_s
        return regression_velocity(self.history.since(t - w))


class CriticalSpring:
    """Critically damped spring toward a moving target. Used by the actuator to
    turn 30 Hz camera samples into smooth 120 Hz motion without overshoot."""

    def __init__(self, frequency_hz: float = 12.0):
        self.omega = 2.0 * math.pi * frequency_hz
        self.pos: np.ndarray | None = None
        self.vel: np.ndarray | None = None

    def reset(self, pos=None) -> None:
        self.pos = None if pos is None else np.asarray(pos, dtype=float).copy()
        self.vel = None if pos is None else np.zeros_like(self.pos)

    def step(self, target, dt: float) -> np.ndarray:
        target = np.asarray(target, dtype=float)
        if self.pos is None:
            self.reset(target)
            return self.pos.copy()
        # Exact integration of x'' = -w^2 (x - target) - 2w x' over dt.
        w = self.omega
        x0 = self.pos - target
        v0 = self.vel
        e = math.exp(-w * dt)
        self.pos = target + (x0 + (v0 + w * x0) * dt) * e
        self.vel = (v0 - w * (v0 + w * x0) * dt) * e
        return self.pos.copy()


class EMA:
    def __init__(self, alpha: float, initial: float | None = None):
        self.alpha = alpha
        self.value = initial

    def __call__(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class Hysteresis:
    """Boolean latch: turns on below/above ``enter`` and off only past ``exit``."""

    def __init__(self, enter: float, exit: float, low_is_on: bool = False):
        self.enter, self.exit, self.low_is_on = enter, exit, low_is_on
        self.on = False

    def __call__(self, x: float) -> bool:
        if self.low_is_on:
            self.on = x < self.enter if not self.on else x < self.exit
        else:
            self.on = x > self.enter if not self.on else x > self.exit
        return self.on
