"""Interpret a physical throw as an intentional desktop command.

We never animate a window flying across the OS. The release velocity is
projected forward to find where the user was *aiming*, then mapped to one
discrete, predictable outcome:
  * projection lands on another monitor       -> move there (same relative geometry)
  * horizontal                                -> snap to that half of the monitor
      (already snapped that side + a neighbour monitor exists -> hop to it,
       like pressing Win+Arrow twice)
  * up                                        -> maximize
  * down                                      -> minimize (restore if maximized)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from vector.core.geometry import Rect
from vector.desktop.monitors import Monitor, VirtualDesktop


@dataclass
class ThrowDecision:
    action: str                 # monitor | snap_left | snap_right | maximize | minimize | restore | none
    rect: Rect | None = None    # target frame for monitor/snap moves
    monitor: Monitor | None = None
    maximize_after: bool = False
    projected: tuple[float, float] | None = None


def direction_of(angle: float) -> str:
    """Screen coords (y down). 45° sectors."""
    deg = math.degrees(angle)
    if -45 <= deg <= 45:
        return "right"
    if 45 < deg < 135:
        return "down"
    if -135 < deg < -45:
        return "up"
    return "left"


def half(work: Rect, side: str) -> Rect:
    w = work.w / 2
    return Rect(work.x if side == "left" else work.x + work.w - w, work.y, w, work.h)


def _is_snapped(frame: Rect, work: Rect, side: str, tol: float = 12) -> bool:
    h = half(work, side)
    return all(abs(a - b) <= tol for a, b in ((frame.x, h.x), (frame.y, h.y), (frame.w, h.w), (frame.h, h.h)))


def decide(frame: Rect, maximized: bool, release: np.ndarray, velocity_px: np.ndarray,
           desktop: VirtualDesktop, project_s: float = 0.28, to_monitors: bool = True,
           snap: bool = True) -> ThrowDecision:
    src = desktop.monitor_for_rect(frame)
    angle = math.atan2(float(velocity_px[1]), float(velocity_px[0]))
    d = direction_of(angle)
    proj = (float(release[0] + velocity_px[0] * project_s),
            float(release[1] + velocity_px[1] * project_s))

    if to_monitors and len(desktop.monitors) > 1:
        dst = desktop.monitor_at(*proj)
        if dst is not None and dst != src:
            return _to_monitor(frame, maximized, src, dst, proj)

    if d in ("left", "right"):
        if to_monitors and _is_snapped(frame, src.work, d):
            neighbour = desktop.monitor_in_direction(src, 1 if d == "right" else -1, 0)
            if neighbour is not None:
                opposite = "left" if d == "right" else "right"
                return ThrowDecision("monitor", half(neighbour.work, opposite), neighbour,
                                     projected=proj)
        if snap:
            return ThrowDecision(f"snap_{d}", half(src.work, d), src, projected=proj)
        return ThrowDecision("none", projected=proj)
    if d == "up":
        return ThrowDecision("maximize", monitor=src, projected=proj)
    return ThrowDecision("restore" if maximized else "minimize", monitor=src, projected=proj)


def _to_monitor(frame: Rect, maximized: bool, src: Monitor, dst: Monitor, proj) -> ThrowDecision:
    rect = VirtualDesktop.transfer_rect(frame, src, dst)
    return ThrowDecision("monitor", rect, dst, maximize_after=maximized, projected=proj)
