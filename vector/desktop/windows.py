"""Real Win32 window control via ctypes.

Two rectangles matter for every top-level window on Windows 10/11:
* the *window rect* (GetWindowRect / SetWindowPos) which includes invisible
  resize borders (~7px at 100% scale), and
* the *frame* (DWMWA_EXTENDED_FRAME_BOUNDS) which is what the user sees.
All public functions here speak in *frame* coordinates so that a window
snapped to the left half is flush with the screen edge, not 7px short.
"""

from __future__ import annotations

import ctypes
import logging
import os
import time
from ctypes import wintypes
from dataclasses import dataclass
from functools import lru_cache

from vector.core.geometry import Rect

log = logging.getLogger(__name__)

user32 = ctypes.WinDLL("user32", use_last_error=True)
dwmapi = ctypes.WinDLL("dwmapi")
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

GWL_STYLE, GWL_EXSTYLE = -16, -20
WS_CAPTION = 0x00C00000
WS_THICKFRAME = 0x00040000
WS_MAXIMIZEBOX = 0x00010000
WS_MINIMIZEBOX = 0x00020000
WS_POPUP = 0x80000000
WS_CHILD = 0x40000000
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_APPWINDOW = 0x00040000
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOPMOST = 0x00000008
DWMWA_EXTENDED_FRAME_BOUNDS = 9
DWMWA_CLOAKED = 14
GA_ROOT, GA_ROOTOWNER = 2, 3
GW_OWNER = 4
SW_MAXIMIZE, SW_MINIMIZE, SW_RESTORE, SW_SHOWMINNOACTIVE = 3, 6, 9, 7
SWP_NOSIZE, SWP_NOMOVE, SWP_NOZORDER, SWP_NOACTIVATE = 0x1, 0x2, 0x4, 0x10
SWP_ASYNCWINDOWPOS, SWP_NOOWNERZORDER = 0x4000, 0x200
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

WS_EX_TRANSPARENT, WS_EX_LAYERED = 0x00000020, 0x00080000
_EnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def _proto(dll, name, restype, *argtypes):
    fn = getattr(dll, name)
    fn.restype, fn.argtypes = restype, list(argtypes)


# Full prototypes: without them ctypes truncates handles to 32-bit ints on x64.
H, B, U, I = wintypes.HWND, wintypes.BOOL, wintypes.UINT, ctypes.c_int
_proto(user32, "GetWindowLongW", ctypes.c_long, H, I)
_proto(user32, "GetAncestor", H, H, U)
_proto(user32, "GetWindow", H, H, U)
_proto(user32, "GetForegroundWindow", H)
_proto(user32, "SetForegroundWindow", B, H)
_proto(user32, "BringWindowToTop", B, H)
_proto(user32, "SetWindowPos", B, H, H, I, I, I, I, U)
_proto(user32, "ShowWindowAsync", B, H, I)
_proto(user32, "IsWindow", B, H)
_proto(user32, "IsWindowVisible", B, H)
_proto(user32, "IsIconic", B, H)
_proto(user32, "IsZoomed", B, H)
_proto(user32, "GetWindowRect", B, H, ctypes.POINTER(wintypes.RECT))
_proto(user32, "GetClassNameW", I, H, wintypes.LPWSTR, I)
_proto(user32, "GetWindowTextLengthW", I, H)
_proto(user32, "GetWindowTextW", I, H, wintypes.LPWSTR, I)
_proto(user32, "GetWindowThreadProcessId", wintypes.DWORD, H, ctypes.POINTER(wintypes.DWORD))
_proto(user32, "EnumWindows", B, _EnumProc, wintypes.LPARAM)
_proto(user32, "FindWindowW", H, wintypes.LPCWSTR, wintypes.LPCWSTR)
_proto(dwmapi, "DwmGetWindowAttribute", ctypes.c_long, H, wintypes.DWORD, ctypes.c_void_p,
       wintypes.DWORD)
_proto(kernel32, "OpenProcess", wintypes.HANDLE, wintypes.DWORD, B, wintypes.DWORD)
_proto(kernel32, "CloseHandle", B, wintypes.HANDLE)
_proto(kernel32, "QueryFullProcessImageNameW", B, wintypes.HANDLE, wintypes.DWORD,
       wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD))


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    class_name: str
    pid: int
    process: str
    frame: Rect           # visible bounds (DWM)
    rect: Rect            # GetWindowRect bounds (incl. invisible borders)
    maximized: bool
    minimized: bool
    resizable: bool
    topmost: bool

    @property
    def insets(self) -> tuple[float, float, float, float]:
        """Invisible border sizes (left, top, right, bottom)."""
        return (self.frame.x - self.rect.x, self.frame.y - self.rect.y,
                self.rect.right - self.frame.right, self.rect.bottom - self.frame.bottom)


def _rect(hwnd: int) -> Rect:
    r = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return Rect.from_ltrb(r.left, r.top, r.right, r.bottom)


def frame_rect(hwnd: int) -> Rect:
    r = wintypes.RECT()
    if dwmapi.DwmGetWindowAttribute(wintypes.HWND(hwnd), DWMWA_EXTENDED_FRAME_BOUNDS,
                                    ctypes.byref(r), ctypes.sizeof(r)) == 0:
        return Rect.from_ltrb(r.left, r.top, r.right, r.bottom)
    return _rect(hwnd)


def is_cloaked(hwnd: int) -> bool:
    """UWP apps on other virtual desktops / suspended are 'cloaked': visible
    according to IsWindowVisible but not actually on screen."""
    v = wintypes.DWORD()
    dwmapi.DwmGetWindowAttribute(wintypes.HWND(hwnd), DWMWA_CLOAKED, ctypes.byref(v),
                                 ctypes.sizeof(v))
    return v.value != 0


def class_name(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def title(hwnd: int) -> str:
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def pid_of(hwnd: int) -> int:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


@lru_cache(maxsize=512)
def _process_name(pid: int) -> str:
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return os.path.basename(buf.value)
        return ""
    finally:
        kernel32.CloseHandle(h)


def process_name(pid: int) -> str:
    return _process_name(pid)


def style(hwnd: int) -> int:
    return user32.GetWindowLongW(hwnd, GWL_STYLE) & 0xFFFFFFFF


def exstyle(hwnd: int) -> int:
    return user32.GetWindowLongW(hwnd, GWL_EXSTYLE) & 0xFFFFFFFF


def info(hwnd: int) -> WindowInfo:
    s, pid = style(hwnd), pid_of(hwnd)
    return WindowInfo(
        hwnd=hwnd, title=title(hwnd), class_name=class_name(hwnd), pid=pid,
        process=process_name(pid), frame=frame_rect(hwnd), rect=_rect(hwnd),
        maximized=bool(user32.IsZoomed(hwnd)), minimized=bool(user32.IsIconic(hwnd)),
        resizable=bool(s & WS_THICKFRAME), topmost=bool(exstyle(hwnd) & WS_EX_TOPMOST))


class WindowPolicy:
    """Which windows gestures may touch. Protects the shell, lock screen,
    our own overlays and anything the user blocks in config."""

    def __init__(self, blocked_classes=(), blocked_processes=(), own_hwnds=()):
        self.blocked_classes = set(blocked_classes)
        self.blocked_processes = {p.lower() for p in blocked_processes}
        self.own_hwnds: set[int] = set(own_hwnds)
        self.own_pid = os.getpid()

    def is_candidate(self, hwnd: int) -> bool:
        """Cheap checks for hit-testing: a real, visible, top-level app window."""
        if hwnd in self.own_hwnds or not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return False
        if style(hwnd) & WS_CHILD:
            return False
        if exstyle(hwnd) & WS_EX_TOOLWINDOW and not exstyle(hwnd) & WS_EX_APPWINDOW:
            return False
        if is_cloaked(hwnd):
            return False
        if class_name(hwnd) in self.blocked_classes:
            return False
        pid = pid_of(hwnd)
        if pid == self.own_pid:
            return False
        if process_name(pid).lower() in self.blocked_processes:
            return False
        fr = frame_rect(hwnd)
        return fr.w > 40 and fr.h > 30

    def is_switchable(self, hwnd: int) -> bool:
        """Alt-Tab style eligibility: unowned (or app-window) with a title."""
        if not self.is_candidate(hwnd) or not title(hwnd):
            return False
        owner = user32.GetWindow(hwnd, GW_OWNER)
        return not owner or bool(exstyle(hwnd) & WS_EX_APPWINDOW)


def z_order() -> list[int]:
    """Top-level windows, topmost first."""
    out: list[int] = []

    def cb(hwnd, _):
        out.append(hwnd)
        return True

    user32.EnumWindows(_EnumProc(cb), 0)
    return out


def _is_see_through(hwnd: int, policy: WindowPolicy) -> bool:
    """Windows that don't occlude the pointer: ours, hidden, cloaked, minimised,
    or click-through overlays (layered + transparent, e.g. other HUDs)."""
    if hwnd in policy.own_hwnds or not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
        return True
    ex = exstyle(hwnd)
    if ex & WS_EX_TRANSPARENT and ex & WS_EX_LAYERED:
        return True
    return is_cloaked(hwnd) or pid_of(hwnd) == policy.own_pid


def window_at(x: float, y: float, policy: WindowPolicy) -> int | None:
    """Topmost window under the point, or None if that window is protected.
    We walk the z-order ourselves (instead of WindowFromPoint) to see through
    our own overlays — but we *stop* at a protected occluder such as the
    taskbar, rather than reaching through it to the app underneath."""
    for hwnd in z_order():
        if _is_see_through(hwnd, policy):
            continue
        if not frame_rect(hwnd).contains(x, y):
            continue
        return hwnd if policy.is_candidate(hwnd) else None
    return None


def switchable_windows(policy: WindowPolicy) -> list[int]:
    return [h for h in z_order() if policy.is_switchable(h)]


_proto(user32, "SendMessageTimeoutW", ctypes.c_ssize_t, H, U, wintypes.WPARAM, wintypes.LPARAM,
       U, U, ctypes.POINTER(ctypes.c_size_t))
_proto(user32, "GetClassLongPtrW", ctypes.c_size_t, H, I)


def icon_handle(hwnd: int) -> int:
    """The window's HICON (0 if none). Uses SendMessageTimeout with
    SMTO_ABORTIFHUNG so a frozen app can never block the caller."""
    WM_GETICON, ICON_BIG, ICON_SMALL2, SMTO_ABORTIFHUNG = 0x7F, 1, 2, 0x2
    for which in (ICON_BIG, ICON_SMALL2):
        res = ctypes.c_size_t(0)
        if user32.SendMessageTimeoutW(hwnd, WM_GETICON, which, 0, SMTO_ABORTIFHUNG, 60,
                                      ctypes.byref(res)) and res.value:
            return int(res.value)
    for idx in (-14, -34):                          # GCLP_HICON, GCLP_HICONSM
        h = user32.GetClassLongPtrW(hwnd, idx)
        if h:
            return int(h)
    return 0


def foreground() -> int | None:
    return user32.GetForegroundWindow() or None


def is_window(hwnd: int) -> bool:
    return bool(user32.IsWindow(hwnd))


def set_frame(hwnd: int, frame: Rect, insets: tuple[float, float, float, float] | None = None,
              move_only: bool = False) -> bool:
    """Position a window so its *visible* frame is ``frame``. Async so a hung
    target app can never stall the actuator thread."""
    if insets is None:
        insets = info(hwnd).insets
    l, t, r, b = insets
    x, y = round(frame.x - l), round(frame.y - t)
    w, h = round(frame.w + l + r), round(frame.h + t + b)
    flags = SWP_NOZORDER | SWP_NOACTIVATE | SWP_ASYNCWINDOWPOS | SWP_NOOWNERZORDER
    if move_only:
        flags |= SWP_NOSIZE
    ok = bool(user32.SetWindowPos(hwnd, None, x, y, w, h, flags))
    if not ok:
        log.debug("SetWindowPos failed for %s (err %s)", hwnd, ctypes.get_last_error())
    return ok


def show(hwnd: int, cmd: int) -> None:
    user32.ShowWindowAsync(hwnd, cmd)


def maximize(hwnd: int) -> None:
    show(hwnd, SW_MAXIMIZE)


def minimize(hwnd: int) -> None:
    show(hwnd, SW_MINIMIZE)


def restore(hwnd: int) -> None:
    show(hwnd, SW_RESTORE)


def restore_sync(hwnd: int, timeout_s: float = 0.25) -> None:
    """Restore and wait (briefly) until the window reports it is no longer
    maximized, so a following SetWindowPos isn't overridden."""
    user32.ShowWindowAsync(hwnd, SW_RESTORE)
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline and (user32.IsZoomed(hwnd) or user32.IsIconic(hwnd)):
        time.sleep(0.005)


def focus(hwnd: int) -> bool:
    """Bring a window to the foreground despite the foreground-lock rules.
    Same technique as PowerToys: an *empty* synthetic mouse input (no move,
    no buttons) satisfies the 'last input event' rule without side effects —
    unlike the classic Alt-key trick, which can open an app's menu bar."""
    from vector.desktop.input import null_mouse_input
    if user32.IsIconic(hwnd):
        user32.ShowWindowAsync(hwnd, SW_RESTORE)
    if user32.SetForegroundWindow(hwnd):
        return True
    null_mouse_input()
    ok = bool(user32.SetForegroundWindow(hwnd))
    user32.BringWindowToTop(hwnd)
    return ok
