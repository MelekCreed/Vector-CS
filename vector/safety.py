"""Emergency failsafe + global hotkeys via a low-level keyboard hook.

ESC pressed twice within ``failsafe_window_s`` disarms the actuator *from the
hook thread itself* — it doesn't wait for the vision or UI threads, so it
works even if those are stuck. Injected keystrokes (including our own) are
ignored, so gestures can never trigger or suppress the failsafe.
"""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from ctypes import wintypes
from typing import Callable

log = logging.getLogger(__name__)

WH_KEYBOARD_LL = 13
WM_KEYDOWN, WM_SYSKEYDOWN = 0x0100, 0x0104
WM_QUIT = 0x0012
LLKHF_INJECTED = 0x10
VK_ESCAPE, VK_CONTROL, VK_MENU, VK_SHIFT = 0x1B, 0x11, 0x12, 0x10

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD),
                ("flags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


LRESULT = ctypes.c_ssize_t
HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD]
user32.SetWindowsHookExW.restype = wintypes.HHOOK
user32.CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
user32.CallNextHookEx.restype = LRESULT
user32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
user32.GetAsyncKeyState.restype = ctypes.c_short
user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]


def parse_hotkey(spec: str) -> tuple[frozenset[str], int]:
    """'ctrl+alt+v' -> ({'ctrl','alt'}, VK)."""
    from vector.desktop.input import VK
    parts = [p.strip().lower() for p in spec.split("+")]
    mods = frozenset(p for p in parts if p in ("ctrl", "alt", "shift"))
    keys = [p for p in parts if p not in mods]
    if len(keys) != 1:
        raise ValueError(f"hotkey needs exactly one non-modifier key: {spec!r}")
    return mods, VK[keys[0]]


class DoubleTap:
    """Pure double-tap detector (unit-tested without a hook)."""

    def __init__(self, window_s: float):
        self.window_s = window_s
        self._last = -1e9

    def press(self, t: float) -> bool:
        hit = t - self._last <= self.window_s
        self._last = -1e9 if hit else t
        return hit


class KeyboardGuard:
    def __init__(self, on_failsafe: Callable[[], None], window_s: float = 0.45,
                 failsafe_key: str = "esc"):
        from vector.desktop.input import VK
        self.on_failsafe = on_failsafe
        self._tap = DoubleTap(window_s)
        self._failsafe_vk = VK[failsafe_key]
        self._hotkeys: dict[tuple[frozenset[str], int], Callable[[], None]] = {}
        self._thread: threading.Thread | None = None
        self._tid = 0
        self._hook = None
        self._proc = HOOKPROC(self._callback)     # keep a reference: GC would crash the hook
        self.dispatch: Callable[[Callable[[], None]], None] = lambda fn: fn()

    def add_hotkey(self, spec: str, fn: Callable[[], None]) -> None:
        self._hotkeys[parse_hotkey(spec)] = fn

    @staticmethod
    def _mods() -> frozenset[str]:
        down = lambda vk: bool(user32.GetAsyncKeyState(vk) & 0x8000)  # noqa: E731
        return frozenset(n for n, vk in (("ctrl", VK_CONTROL), ("alt", VK_MENU), ("shift", VK_SHIFT))
                         if down(vk))

    def _callback(self, code, wparam, lparam):
        try:
            if code >= 0 and wparam in (WM_KEYDOWN, WM_SYSKEYDOWN):
                kb = ctypes.cast(lparam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                if not kb.flags & LLKHF_INJECTED:
                    if kb.vkCode == self._failsafe_vk and self._tap.press(time.perf_counter()):
                        self.on_failsafe()               # fast, thread-safe, synchronous
                    else:
                        fn = self._hotkeys.get((self._mods(), kb.vkCode))
                        if fn is not None:
                            self.dispatch(fn)
                            return 1                     # swallow our own hotkeys
        except Exception:  # a hook must never raise
            log.exception("keyboard hook error")
        return user32.CallNextHookEx(self._hook, code, wparam, lparam)

    def start(self) -> "KeyboardGuard":
        ready = threading.Event()

        def run():
            self._tid = kernel32.GetCurrentThreadId()
            self._hook = user32.SetWindowsHookExW(WH_KEYBOARD_LL, self._proc, None, 0)
            ready.set()
            if not self._hook:
                log.error("SetWindowsHookEx failed: %s", ctypes.get_last_error())
                return
            msg = wintypes.MSG()
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
            user32.UnhookWindowsHookEx(self._hook)

        self._thread = threading.Thread(target=run, name="keyboard-guard", daemon=True)
        self._thread.start()
        ready.wait(2.0)
        return self

    @property
    def active(self) -> bool:
        return bool(self._hook)

    def stop(self) -> None:
        if self._tid:
            user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
        if self._thread:
            self._thread.join(timeout=1.0)
