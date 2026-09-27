"""Air-drawing model (pure logic; the overlay renders it).

Raw fingertip samples arrive at camera rate and are already One-Euro
filtered by the cursor mapper. Strokes are stored as those control points and
rendered through a centripetal Catmull-Rom spline, so a 30 Hz sampled
gesture draws as a smooth curve instead of a polyline with corners.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Stroke:
    color: str
    width: float
    points: list[tuple[float, float]] = field(default_factory=list)

    def bounds(self) -> tuple[float, float, float, float]:
        a = np.asarray(self.points)
        return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())


def catmull_rom(points: list[tuple[float, float]], samples_per_seg: int = 8,
                alpha: float = 0.5) -> np.ndarray:
    """Centripetal Catmull-Rom (alpha=0.5): no cusps or self-intersections on
    uneven spacing — exactly what jittery camera samples produce."""
    p = np.asarray(points, dtype=float)
    if len(p) < 3:
        return p
    p = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = [p[1]]
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]

        def tj(ti, a, b):
            return ti + max(np.linalg.norm(b - a), 1e-6) ** alpha

        t0 = 0.0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)
        for t in np.linspace(t1, t2, samples_per_seg + 1)[1:]:
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.asarray(out)


class Canvas:
    def __init__(self, colors: list[str], width: float = 5.0, min_spacing: float = 2.5,
                 max_undo: int = 50):
        self.colors = colors
        self.color_idx = 0
        self.width = width
        self.min_spacing = min_spacing
        self.strokes: list[Stroke] = []
        self.active: Stroke | None = None
        self._undo: list[list[Stroke]] = []
        self.max_undo = max_undo
        self.version = 0            # bumps on every change (render cache key)
        self.lock = threading.RLock()   # vision thread writes, UI thread reads

    @property
    def color(self) -> str:
        return self.colors[self.color_idx % len(self.colors)]

    def _snapshot(self) -> None:
        self._undo.append([Stroke(s.color, s.width, list(s.points)) for s in self.strokes])
        del self._undo[:-self.max_undo]

    def _touch(self) -> None:
        self.version += 1

    def begin(self, p) -> None:
        self._snapshot()
        self.active = Stroke(self.color, self.width, [tuple(map(float, p))])
        self.strokes.append(self.active)
        self._touch()

    def add(self, p) -> None:
        if self.active is None:
            self.begin(p)
            return
        last = self.active.points[-1]
        if np.hypot(p[0] - last[0], p[1] - last[1]) >= self.min_spacing:
            self.active.points.append((float(p[0]), float(p[1])))
            self._touch()

    def end(self) -> None:
        if self.active is not None and len(self.active.points) < 2:
            self.strokes.remove(self.active)       # a tap, not a stroke
            self._undo.pop()
        self.active = None
        self._touch()

    def erase(self, p, radius: float) -> bool:
        """Remove points within ``radius`` and split strokes at the gap."""
        px, py = float(p[0]), float(p[1])
        hit = any(np.hypot(x - px, y - py) <= radius for s in self.strokes for x, y in s.points)
        if not hit:
            return False
        self._snapshot()
        out = []
        for s in self.strokes:
            run: list[tuple[float, float]] = []
            for x, y in s.points:
                if np.hypot(x - px, y - py) <= radius:
                    if len(run) >= 2:
                        out.append(Stroke(s.color, s.width, run))
                    run = []
                else:
                    run.append((x, y))
            if len(run) >= 2:
                out.append(Stroke(s.color, s.width, run))
        self.strokes = out
        self.active = None
        self._touch()
        return True

    def clear(self) -> None:
        if self.strokes:
            self._snapshot()
            self.strokes = []
        self.active = None
        self._touch()

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.strokes = self._undo.pop()
        self.active = None
        self._touch()
        return True

    def next_color(self) -> str:
        self.color_idx = (self.color_idx + 1) % len(self.colors)
        self._touch()
        return self.color

    def set_width(self, w: float) -> None:
        self.width = float(min(40.0, max(1.0, w)))
        self._touch()

    def smoothed(self, s: Stroke) -> np.ndarray:
        return catmull_rom(s.points)
