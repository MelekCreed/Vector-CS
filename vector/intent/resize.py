"""Two-hand resize geometry.

Width follows the change in *horizontal* hand separation and height the
change in *vertical* separation, blended with the uniform (diagonal) scale so
that hands held side-by-side (tiny vertical separation) still resize
height naturally instead of dividing by ~0. The window centre follows the
midpoint between the hands, so you can stretch and reposition in one motion.
"""

from __future__ import annotations

import numpy as np

from vector.core.geometry import Rect

SOFT_PX = 120.0      # separation softening: avoids exploding ratios near zero
AXIS_BLEND = 0.65    # 0 = uniform scaling only, 1 = independent axes only


def resize_rect(start: Rect, p0: tuple[np.ndarray, np.ndarray], p: tuple[np.ndarray, np.ndarray],
                work: Rect, min_w: float = 360, min_h: float = 240) -> Rect:
    a0, b0 = p0
    a, b = p
    d0, d = np.abs(a0 - b0), np.abs(a - b)
    uniform = (np.linalg.norm(a - b) + SOFT_PX) / (np.linalg.norm(a0 - b0) + SOFT_PX)
    sx = (d[0] + SOFT_PX) / (d0[0] + SOFT_PX)
    sy = (d[1] + SOFT_PX) / (d0[1] + SOFT_PX)
    sx = uniform ** (1 - AXIS_BLEND) * sx ** AXIS_BLEND
    sy = uniform ** (1 - AXIS_BLEND) * sy ** AXIS_BLEND

    w = min(max(start.w * sx, min_w), work.w)
    h = min(max(start.h * sy, min_h), work.h)
    mid0 = (a0 + b0) / 2
    mid = (a + b) / 2
    cx, cy = start.center
    cx, cy = cx + (mid[0] - mid0[0]), cy + (mid[1] - mid0[1])
    return Rect(cx - w / 2, cy - h / 2, w, h).fit_inside(work)
