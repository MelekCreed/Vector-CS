"""Unified virtual-desktop model.

All coordinates are *physical* pixels in Windows' virtual-screen space (the
primary monitor's top-left is (0, 0); other monitors may be negative). The
process is per-monitor DPI aware, so this matches every Win32 API we call.
Layout logic is pure so it can be tested with synthetic mixed-DPI layouts.
"""

from __future__ import annotations

import ctypes
import math
from ctypes import wintypes
from dataclasses import dataclass

from vector.core.geometry import Rect


@dataclass(frozen=True)
class Monitor:
    name: str
    rect: Rect          # full monitor bounds
    work: Rect          # excludes taskbar
    dpi: int = 96
    primary: bool = False

    @property
    def scale(self) -> float:
        return self.dpi / 96.0


class VirtualDesktop:
    def __init__(self, monitors: list[Monitor]):
        if not monitors:
            raise ValueError("no monitors")
        self.monitors = sorted(monitors, key=lambda m: (m.rect.x, m.rect.y))
        xs0 = min(m.rect.x for m in monitors)
        ys0 = min(m.rect.y for m in monitors)
        xs1 = max(m.rect.right for m in monitors)
        ys1 = max(m.rect.bottom for m in monitors)
        self.bounds = Rect(xs0, ys0, xs1 - xs0, ys1 - ys0)

    @property
    def primary(self) -> Monitor:
        return next((m for m in self.monitors if m.primary), self.monitors[0])

    def monitor_at(self, x: float, y: float) -> Monitor | None:
        for m in self.monitors:
            if m.rect.contains(x, y):
                return m
        return None

    def nearest_monitor(self, x: float, y: float) -> Monitor:
        return self.monitor_at(x, y) or min(self.monitors, key=lambda m: m.rect.distance_to(x, y))

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        """Clamp a point onto the nearest real monitor. Needed because the
        bounding box of an L-shaped or offset layout contains dead zones."""
        if self.monitor_at(x, y):
            return x, y
        m = self.nearest_monitor(x, y)
        return m.rect.clamp_point(x, y)

    def from_unit(self, u: float, v: float) -> tuple[float, float]:
        """Map [0,1]^2 onto the virtual desktop bounding box, then onto a monitor."""
        b = self.bounds
        return self.clamp(b.x + u * (b.w - 1), b.y + v * (b.h - 1))

    def monitor_for_rect(self, r: Rect) -> Monitor:
        """Monitor with the largest overlap (what Windows uses for 'owning' monitor)."""
        best, area = None, -1.0
        for m in self.monitors:
            a = m.rect.intersection_area(r)
            if a > area:
                best, area = m, a
        if area <= 0:
            cx, cy = r.center
            return self.nearest_monitor(cx, cy)
        return best

    def monitor_in_direction(self, origin: Monitor, dx: float, dy: float,
                             max_angle_deg: float = 50.0) -> Monitor | None:
        """Closest other monitor whose centre lies within a cone around (dx, dy)."""
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            return None
        ux, uy = dx / norm, dy / norm
        ox, oy = origin.rect.center
        best, best_dist = None, math.inf
        for m in self.monitors:
            if m is origin or m == origin:
                continue
            mx, my = m.rect.center
            vx, vy = mx - ox, my - oy
            d = math.hypot(vx, vy)
            if d < 1e-9:
                continue
            cos = (vx * ux + vy * uy) / d
            if cos >= math.cos(math.radians(max_angle_deg)) and d < best_dist:
                best, best_dist = m, d
        return best

    @staticmethod
    def transfer_rect(r: Rect, src: Monitor, dst: Monitor) -> Rect:
        """Place ``r`` on ``dst`` at the same relative position and relative size
        (so a window taking half of a 4K screen takes half of a 1080p screen),
        clamped into the destination work area."""
        sw, dw = src.work, dst.work
        fx = (r.x - sw.x) / sw.w
        fy = (r.y - sw.y) / sw.h
        fw, fh = r.w / sw.w, r.h / sw.h
        out = Rect(dw.x + fx * dw.w, dw.y + fy * dw.h, fw * dw.w, fh * dw.h)
        return out.fit_inside(dw)

    @classmethod
    def from_system(cls) -> "VirtualDesktop":
        return cls(enumerate_monitors())


class _MONITORINFOEXW(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD),
                ("szDevice", wintypes.WCHAR * 32)]


def enumerate_monitors() -> list[Monitor]:
    user32 = ctypes.windll.user32
    shcore = ctypes.windll.shcore
    out: list[Monitor] = []
    proc_t = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC,
                                ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)

    def cb(hmon, _hdc, _rect, _lp):
        info = _MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(info)
        user32.GetMonitorInfoW(hmon, ctypes.byref(info))
        dpx, dpy = wintypes.UINT(), wintypes.UINT()
        try:
            shcore.GetDpiForMonitor(hmon, 0, ctypes.byref(dpx), ctypes.byref(dpy))
            dpi = int(dpx.value) or 96
        except OSError:
            dpi = 96
        out.append(Monitor(
            name=info.szDevice,
            rect=Rect.from_ltrb(info.rcMonitor.left, info.rcMonitor.top,
                                info.rcMonitor.right, info.rcMonitor.bottom),
            work=Rect.from_ltrb(info.rcWork.left, info.rcWork.top,
                                info.rcWork.right, info.rcWork.bottom),
            dpi=dpi, primary=bool(info.dwFlags & 1)))
        return True

    user32.EnumDisplayMonitors(None, None, proc_t(cb), 0)
    return out
