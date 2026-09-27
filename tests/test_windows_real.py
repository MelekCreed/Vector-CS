"""Integration tests against a REAL top-level window owned by another process."""

import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Win32 only")

from vector.core.geometry import Rect  # noqa: E402
from vector.desktop import windows as W  # noqa: E402
from vector.desktop.dpi import make_process_dpi_aware  # noqa: E402

TITLE = "VectorTestTarget-7f3a"
SCRIPT = f"""
import tkinter as tk
r = tk.Tk(); r.title({TITLE!r}); r.geometry('500x350+200+150'); r.mainloop()
"""


def _wait(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.03)
    return pred()


@pytest.fixture(scope="module")
def target():
    make_process_dpi_aware()
    proc = subprocess.Popen([sys.executable, "-c", SCRIPT])
    policy = W.WindowPolicy()
    hwnd = _wait(lambda: next((h for h in W.z_order() if W.title(h) == TITLE), None))
    assert hwnd, "test window did not appear"
    _wait(lambda: W.user32.IsWindowVisible(hwnd))
    yield hwnd, policy
    proc.kill()


def test_info_and_policy(target):
    hwnd, policy = target
    i = W.info(hwnd)
    assert i.title == TITLE and i.process.lower().startswith("python")
    assert i.resizable and not i.maximized
    assert policy.is_candidate(hwnd) and policy.is_switchable(hwnd)
    l, t, r, b = i.insets
    assert all(v >= 0 for v in (l, t, r, b))


def test_set_frame_is_exact_on_visible_bounds(target):
    hwnd, _ = target
    want = Rect(300, 200, 640, 420)
    assert W.set_frame(hwnd, want)
    got = _wait(lambda: (W.frame_rect(hwnd) == want) and W.frame_rect(hwnd))
    assert W.frame_rect(hwnd) == want, got


def test_move_only_keeps_size(target):
    hwnd, _ = target
    before = W.frame_rect(hwnd)
    W.set_frame(hwnd, before.translated(37, 21), move_only=True)
    _wait(lambda: W.frame_rect(hwnd).x == before.x + 37)
    after = W.frame_rect(hwnd)
    assert (after.x, after.y, after.w, after.h) == (before.x + 37, before.y + 21, before.w, before.h)


def test_window_at_hits_target_and_respects_own_exclusion(target):
    hwnd, policy = target
    W.focus(hwnd)
    _wait(lambda: W.foreground() == hwnd)
    cx, cy = W.frame_rect(hwnd).center
    assert W.window_at(cx, cy, policy) == hwnd
    excluded = W.WindowPolicy(own_hwnds=[hwnd])
    assert W.window_at(cx, cy, excluded) != hwnd


def test_maximize_minimize_restore(target):
    hwnd, _ = target
    W.maximize(hwnd)
    assert _wait(lambda: W.info(hwnd).maximized)
    W.restore_sync(hwnd)
    assert not W.info(hwnd).maximized
    W.minimize(hwnd)
    assert _wait(lambda: W.info(hwnd).minimized)
    W.focus(hwnd)
    assert _wait(lambda: not W.info(hwnd).minimized)


def test_shell_windows_are_protected():
    policy = W.WindowPolicy(blocked_classes=["Shell_TrayWnd", "Progman"])
    tray = W.user32.FindWindowW("Shell_TrayWnd", None)
    if tray:
        assert not policy.is_candidate(tray)


def test_protected_occluder_blocks_hit_test():
    """Pointing at the taskbar must yield None, not the app window beneath it."""
    make_process_dpi_aware()
    tray = W.user32.FindWindowW("Shell_TrayWnd", None)
    if not tray or not W.user32.IsWindowVisible(tray):
        pytest.skip("no visible taskbar")
    fr = W.frame_rect(tray)
    policy = W.WindowPolicy(blocked_classes=["Shell_TrayWnd"])
    assert W.window_at(fr.x + fr.w * 0.5, fr.y + fr.h * 0.5, policy) is None


def test_actuator_drags_and_snaps_a_real_window(target):
    """Real Win32Ops: motion through the actuator mailbox moves the spawned
    test window by exactly the hand delta, and a snap lands flush on the
    visible frame. The cursor is never driven."""
    import numpy as np
    from vector.config import Config
    from vector.desktop.actuator import Actuator, Win32Ops
    from vector.desktop.commands import Command, Motion
    hwnd, _ = target
    W.set_frame(hwnd, Rect(300, 200, 640, 420))
    _wait(lambda: W.frame_rect(hwnd) == Rect(300, 200, 640, 420))
    act = Actuator(Config(), Win32Ops())
    act.arm()
    t = time.perf_counter()
    act.submit(Command("drag_begin", act.generation, t, 1.0, 1,
                       {"hwnd": hwnd, "anchor": np.array([400.0, 300]), "point": np.array([400.0, 300])}))
    for i in range(90):
        now = time.perf_counter()
        act.mailbox.put(Motion(now, drag_point=np.array([520.0, 360])))
        act.tick(now)
        time.sleep(1 / 120)
    assert _wait(lambda: W.frame_rect(hwnd).x == 420 and W.frame_rect(hwnd).y == 260), W.frame_rect(hwnd)
    act.submit(Command("drag_end", act.generation, time.perf_counter(), 1.0, 1, {}))
    snap_rect = Rect(0, 0, 700, 500)
    act.submit(Command("set_frame", act.generation, time.perf_counter(), 1.0, 1,
                       {"hwnd": hwnd, "rect": snap_rect}))
    act.tick(time.perf_counter())
    assert _wait(lambda: W.frame_rect(hwnd) == snap_rect), W.frame_rect(hwnd)
