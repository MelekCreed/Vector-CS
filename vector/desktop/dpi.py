"""DPI awareness must be set before any window (including Qt's) is created,
otherwise Windows virtualises coordinates and every Win32 rectangle lies by
the scaling factor (e.g. 1536x864 instead of 1920x1080 at 125%)."""

from __future__ import annotations

import ctypes
import logging

log = logging.getLogger(__name__)

DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)
_done = False


def make_process_dpi_aware() -> str:
    global _done
    if _done:
        return "already"
    _done = True
    user32 = ctypes.windll.user32
    try:
        if user32.SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2):
            return "per-monitor-v2"
        # ERROR_ACCESS_DENIED means it was already set (e.g. by Qt) — fine if it's PMv2.
    except AttributeError:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        return "per-monitor"
    except (AttributeError, OSError):
        pass
    user32.SetProcessDPIAware()
    return "system"


def current_awareness() -> int:
    """Returns the awareness of the calling thread: 0 unaware, 1 system, 2 per-monitor."""
    user32 = ctypes.windll.user32
    user32.GetThreadDpiAwarenessContext.restype = ctypes.c_void_p
    user32.GetAwarenessFromDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    ctx = user32.GetThreadDpiAwarenessContext()
    return int(user32.GetAwarenessFromDpiAwarenessContext(ctx))
