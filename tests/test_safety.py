import ctypes
import sys

import pytest

from vector.safety import (KBDLLHOOKSTRUCT, LLKHF_INJECTED, WM_KEYDOWN, DoubleTap, KeyboardGuard,
                           parse_hotkey)

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Win32 only")


def test_double_tap_window():
    d = DoubleTap(0.45)
    assert not d.press(0.0)
    assert d.press(0.3)
    assert not d.press(0.35)            # consumed: a third press starts over
    assert not d.press(1.0)
    assert not d.press(1.6)             # too slow


def test_parse_hotkey():
    mods, vk = parse_hotkey("Ctrl+Alt+V")
    assert mods == {"ctrl", "alt"} and vk == ord("V")
    with pytest.raises(ValueError):
        parse_hotkey("ctrl+alt")


def _event(vk, injected=False):
    kb = KBDLLHOOKSTRUCT(vk, 0, LLKHF_INJECTED if injected else 0, 0, 0)
    return kb, ctypes.addressof(kb)


def test_callback_physical_double_esc_fires_injected_never_does():
    """Drives the hook callback directly with crafted events — no keystrokes
    are sent to the real desktop."""
    fired = []
    g = KeyboardGuard(lambda: fired.append(1))
    for _ in range(2):
        kb, p = _event(0x1B, injected=True)
        g._callback(0, WM_KEYDOWN, p)
    assert fired == []                  # synthetic input can't trip the failsafe
    for _ in range(2):
        kb, p = _event(0x1B)
        g._callback(0, WM_KEYDOWN, p)
    assert fired == [1]


def test_real_hook_installs_and_uninstalls():
    g = KeyboardGuard(lambda: None).start()
    try:
        assert g.active
    finally:
        g.stop()
