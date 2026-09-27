"""Lightweight latency profiler for the debug HUD."""

from __future__ import annotations

import threading
import time
from collections import deque


class Rolling:
    def __init__(self, n: int = 90):
        self.values: deque[float] = deque(maxlen=n)

    def add(self, v: float) -> None:
        self.values.append(v)

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    def pct(self, q: float) -> float:
        if not self.values:
            return 0.0
        s = sorted(self.values)
        return s[min(len(s) - 1, int(q * len(s)))]


class Metrics:
    """Named rolling timings (ms) + rate counters, safe to read from any thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stats: dict[str, Rolling] = {}
        self._rates: dict[str, deque[float]] = {}

    def add(self, name: str, ms: float) -> None:
        with self._lock:
            self._stats.setdefault(name, Rolling()).add(ms)

    def tick(self, name: str) -> None:
        now = time.perf_counter()
        with self._lock:
            d = self._rates.setdefault(name, deque(maxlen=120))
            d.append(now)

    def rate(self, name: str) -> float:
        with self._lock:
            d = self._rates.get(name)
            if not d or len(d) < 2:
                return 0.0
            span = d[-1] - d[0]
            return (len(d) - 1) / span if span > 0 else 0.0

    def summary(self) -> dict[str, tuple[float, float]]:
        """name -> (mean ms, p95 ms)"""
        with self._lock:
            return {k: (v.mean, v.pct(0.95)) for k, v in self._stats.items()}


class Timer:
    def __init__(self, metrics: Metrics, name: str):
        self.m, self.name = metrics, name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self.t0) * 1000
        self.m.add(self.name, self.ms)
