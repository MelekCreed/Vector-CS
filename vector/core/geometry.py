from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @classmethod
    def from_ltrb(cls, l: float, t: float, r: float, b: float) -> "Rect":
        return cls(l, t, r - l, b - t)

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    def ltrb(self) -> tuple[int, int, int, int]:
        return round(self.x), round(self.y), round(self.right), round(self.bottom)

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x < self.right and self.y <= y < self.bottom

    def clamp_point(self, x: float, y: float) -> tuple[float, float]:
        return (min(max(x, self.x), self.right - 1), min(max(y, self.y), self.bottom - 1))

    def distance_to(self, x: float, y: float) -> float:
        dx = max(self.x - x, 0.0, x - self.right)
        dy = max(self.y - y, 0.0, y - self.bottom)
        return math.hypot(dx, dy)

    def intersection_area(self, o: "Rect") -> float:
        w = min(self.right, o.right) - max(self.x, o.x)
        h = min(self.bottom, o.bottom) - max(self.y, o.y)
        return max(0.0, w) * max(0.0, h)

    def translated(self, dx: float, dy: float) -> "Rect":
        return Rect(self.x + dx, self.y + dy, self.w, self.h)

    def with_center(self, cx: float, cy: float) -> "Rect":
        return Rect(cx - self.w / 2, cy - self.h / 2, self.w, self.h)

    def fit_inside(self, bounds: "Rect") -> "Rect":
        """Shrink to fit if needed, then shift fully inside ``bounds``."""
        w, h = min(self.w, bounds.w), min(self.h, bounds.h)
        x = min(max(self.x, bounds.x), bounds.right - w)
        y = min(max(self.y, bounds.y), bounds.bottom - h)
        return Rect(x, y, w, h)

    def inset(self, left: float, top: float, right: float, bottom: float) -> "Rect":
        return Rect(self.x + left, self.y + top, self.w - left - right, self.h - top - bottom)

    def rounded(self) -> "Rect":
        return Rect(round(self.x), round(self.y), round(self.w), round(self.h))
