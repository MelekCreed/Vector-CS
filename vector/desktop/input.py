"""Synthetic mouse / keyboard input through SendInput.

Everything here is a *non-destructive* input primitive. Arbitrary key chords
come only from the user's own config bindings.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)

INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x0002, 0x0004
MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP = 0x0008, 0x0010
MOUSEEVENTF_WHEEL, MOUSEEVENTF_HWHEEL = 0x0800, 0x1000
KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP = 0x0001, 0x0002
WHEEL_DELTA = 120

# Tag injected events so our own keyboard hook can recognise them.
VECTOR_EXTRA_INFO = 0x5645_4354  # "VECT"


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class _U(ctypes.Union):
    # HARDWAREINPUT is the smallest member, so MOUSEINPUT defines the size.
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _U)]


user32.SendInput.restype = wintypes.UINT
user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
assert ctypes.sizeof(INPUT) == (40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)


def _send(*inputs: INPUT) -> int:
    arr = (INPUT * len(inputs))(*inputs)
    return user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))


def _mouse(flags: int, data: int = 0) -> INPUT:
    i = INPUT(type=INPUT_MOUSE)
    i.mi = MOUSEINPUT(0, 0, ctypes.c_uint32(data).value, flags, 0, VECTOR_EXTRA_INFO)
    return i


def _key(vk: int, up: bool = False) -> INPUT:
    flags = KEYEVENTF_KEYUP if up else 0
    if vk in _EXTENDED:
        flags |= KEYEVENTF_EXTENDEDKEY
    i = INPUT(type=INPUT_KEYBOARD)
    i.ki = KEYBDINPUT(vk, 0, flags, 0, VECTOR_EXTRA_INFO)
    return i


def null_mouse_input() -> None:
    """An input event with no effect; used to satisfy foreground-lock rules."""
    _send(_mouse(0))


def cursor_pos() -> tuple[int, int]:
    p = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(p))
    return p.x, p.y


def move_cursor(x: float, y: float) -> None:
    # SetCursorPos takes physical coords in a per-monitor-v2 aware process.
    user32.SetCursorPos(int(round(x)), int(round(y)))


def left_down() -> None:
    _send(_mouse(MOUSEEVENTF_LEFTDOWN))


def left_up() -> None:
    _send(_mouse(MOUSEEVENTF_LEFTUP))


def click(button: str = "left") -> None:
    if button == "right":
        _send(_mouse(MOUSEEVENTF_RIGHTDOWN), _mouse(MOUSEEVENTF_RIGHTUP))
    else:
        _send(_mouse(MOUSEEVENTF_LEFTDOWN), _mouse(MOUSEEVENTF_LEFTUP))


def release_all_buttons() -> None:
    """Failsafe: never leave a synthetic button stuck down."""
    _send(_mouse(MOUSEEVENTF_LEFTUP), _mouse(MOUSEEVENTF_RIGHTUP))


def wheel(delta: int, horizontal: bool = False) -> None:
    """Positive = up/right. Sub-120 deltas give smooth scrolling in modern apps."""
    if delta:
        _send(_mouse(MOUSEEVENTF_HWHEEL if horizontal else MOUSEEVENTF_WHEEL, delta))


VK = {
    "ctrl": 0x11, "control": 0x11, "shift": 0x10, "alt": 0x12, "win": 0x5B, "tab": 0x09,
    "esc": 0x1B, "escape": 0x1B, "enter": 0x0D, "space": 0x20, "backspace": 0x08,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28, "home": 0x24, "end": 0x23,
    "pageup": 0x21, "pagedown": 0x22, "delete": 0x2E,
    "browser_back": 0xA6, "browser_forward": 0xA7,
    "volume_mute": 0xAD, "volume_down": 0xAE, "volume_up": 0xAF,
    "media_next": 0xB0, "media_previous": 0xB1, "media_stop": 0xB2, "media_play_pause": 0xB3,
    **{f"f{i}": 0x6F + i for i in range(1, 25)},
    **{chr(c): c for c in range(ord("0"), ord("9") + 1)},
    **{chr(c).lower(): c for c in range(ord("A"), ord("Z") + 1)},
    "[": 0xDB, "]": 0xDD, "-": 0xBD, "=": 0xBB,
}
_EXTENDED = {0x25, 0x26, 0x27, 0x28, 0x24, 0x23, 0x21, 0x22, 0x2E, 0x5B,
             0xA6, 0xA7, 0xAD, 0xAE, 0xAF, 0xB0, 0xB1, 0xB2, 0xB3}


def parse_chord(chord: str) -> list[int]:
    keys = [k.strip().lower() for k in chord.split("+") if k.strip()]
    try:
        return [VK[k] for k in keys]
    except KeyError as exc:
        raise ValueError(f"unknown key {exc.args[0]!r} in chord {chord!r}") from None


def press_chord(chord: str) -> None:
    vks = parse_chord(chord)
    _send(*[_key(v) for v in vks], *[_key(v, up=True) for v in reversed(vks)])


def tap(vk_name: str) -> None:
    press_chord(vk_name)
