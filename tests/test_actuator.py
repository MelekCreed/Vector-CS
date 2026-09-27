import numpy as np
import pytest

from vector.config import Config
from vector.core.geometry import Rect
from vector.desktop.actuator import Actuator
from vector.desktop.commands import Command, Motion
from vector.desktop.windows import WindowInfo


class FakeOps:
    def __init__(self):
        self.calls = []
        self.windows = {}
        self.cursor = (0, 0)

    def add(self, hwnd, frame, maximized=False):
        self.windows[hwnd] = dict(frame=frame, maximized=maximized, minimized=False)

    def info(self, hwnd):
        w = self.windows.get(hwnd)
        if w is None:
            return None
        return WindowInfo(hwnd, "w", "C", 1, "p.exe", w["frame"], w["frame"], w["maximized"],
                          w["minimized"], True, False)

    def set_frame(self, hwnd, rect, insets=None, move_only=False):
        self.calls.append(("set_frame", hwnd, rect))
        w = self.windows[hwnd]
        w["frame"] = Rect(rect.x, rect.y, w["frame"].w, w["frame"].h) if move_only else rect

    def move_cursor(self, x, y):
        self.cursor = (x, y)
        self.calls.append(("move_cursor", round(x), round(y)))

    def click(self, button="left"):
        self.calls.append(("click", button))

    def wheel(self, d):
        self.calls.append(("wheel", d))

    def press_chord(self, c):
        self.calls.append(("keys", c))

    def focus(self, hwnd):
        self.calls.append(("focus", hwnd))

    def maximize(self, hwnd):
        self.windows[hwnd]["maximized"] = True
        self.calls.append(("maximize", hwnd))

    def minimize(self, hwnd):
        self.calls.append(("minimize", hwnd))

    def restore(self, hwnd):
        w = self.windows[hwnd]
        w["maximized"] = False
        w["frame"] = Rect(0, 0, 800, 600)
        self.calls.append(("restore", hwnd))

    def release_all_buttons(self):
        self.calls.append(("release_all",))

    def kinds(self):
        return [c[0] for c in self.calls]


@pytest.fixture
def act():
    ops = FakeOps()
    a = Actuator(Config(), ops, rate_hz=120)
    a.arm()
    return a, ops


def cmd(a, kind, t=0.0, **args):
    a.submit(Command(kind, a.generation, t, 0.35, 1, args))


def test_click_moves_then_clicks(act):
    a, ops = act
    cmd(a, "click", point=np.array([300.0, 200]), button="left")
    a.tick(0.01)
    assert ops.calls[:2] == [("move_cursor", 300, 200), ("click", "left")]
    assert a.stats.last_cmd_latency_ms >= 0


def test_stale_generation_commands_are_dropped(act):
    a, ops = act
    cmd(a, "click", point=np.array([1.0, 1]))
    a.disarm("test")                      # e.g. ESC ESC between intent and actuator
    a.arm()
    a.tick(0.01)
    assert "click" not in ops.kinds()
    assert a.stats.dropped_generation >= 0     # queue was cleared by disarm


def test_old_generation_submitted_after_disarm_is_dropped(act):
    a, ops = act
    gen = a.generation
    a.disarm()
    a.arm()
    a.submit(Command("keys", gen, 0.0, 1.0, 0, {"chord": "media_play_pause"}))
    a.tick(0.01)
    assert "keys" not in ops.kinds() and a.stats.dropped_generation == 1


def test_expired_commands_are_dropped(act):
    a, ops = act
    cmd(a, "keys", t=0.0, chord="media_next")
    a.tick(1.0)
    assert "keys" not in ops.kinds() and a.stats.dropped_expired == 1


def test_disarm_releases_buttons_and_stops_drag(act):
    a, ops = act
    ops.add(5, Rect(100, 100, 400, 300))
    cmd(a, "drag_begin", hwnd=5, anchor=np.array([200.0, 150]), point=np.array([200.0, 150]))
    a.tick(0.01)
    a.disarm("esc esc")
    assert a.drag is None and ("release_all",) in ops.calls
    assert not a.armed


def test_drag_moves_window_by_hand_delta_smoothly(act):
    a, ops = act
    ops.add(5, Rect(100, 100, 400, 300))
    cmd(a, "drag_begin", hwnd=5, anchor=np.array([200.0, 150]), point=np.array([210.0, 150]))
    t = 0.0
    for i in range(120):                                    # 1s at 120 Hz
        t = i / 120
        a.mailbox.put(Motion(t, drag_point=np.array([410.0, 250])))
        a.tick(t)
    fr = ops.windows[5]["frame"]
    assert (fr.x, fr.y) == (300, 200)                       # moved by (200, 100)
    xs = [c[2].x for c in ops.calls if c[0] == "set_frame"]
    assert xs == sorted(xs) and len(xs) > 10                # spring: monotonic, many small steps
    assert ("focus", 5) in ops.calls


def test_dragging_maximized_window_restores_under_hand(act):
    a, ops = act
    ops.add(5, Rect(0, 0, 1920, 1040), maximized=True)
    cmd(a, "drag_begin", hwnd=5, anchor=np.array([960.0, 10]), point=np.array([960.0, 10]))
    for i in range(30):
        a.mailbox.put(Motion(i / 120, drag_point=np.array([960.0, 300])))
        a.tick(i / 120)
    fr = ops.windows[5]["frame"]
    assert "restore" in ops.kinds()
    assert fr.w == 800 and abs((fr.x + fr.w / 2) - 960) < 5  # grab point stays mid-window


def test_scroll_accumulates_sub_notch_deltas(act):
    a, ops = act
    for i in range(120):
        a.mailbox.put(Motion(i / 120, scroll_velocity=600.0))
        a.tick(i / 120)
    total = sum(c[1] for c in ops.calls if c[0] == "wheel")
    assert 560 <= total <= 600
    assert all(abs(c[1]) < 120 for c in ops.calls if c[0] == "wheel")


def test_watchdog_stops_motion_when_vision_stalls(act):
    a, ops = act
    ops.add(5, Rect(100, 100, 400, 300))
    cmd(a, "drag_begin", hwnd=5, anchor=np.array([0.0, 0]), point=np.array([0.0, 0]))
    a.mailbox.put(Motion(0.0, drag_point=np.array([50.0, 0]), scroll_velocity=900))
    a.tick(0.01)
    n = len(ops.calls)
    a.tick(0.6)                                              # vision silent 0.6 s
    assert len(ops.calls) == n                               # nothing moves, no wheel
    a.tick(2.0)
    assert a.drag is None


def test_cursor_spring_reaches_target(act):
    a, ops = act
    for i in range(60):
        a.mailbox.put(Motion(i / 120, cursor=np.array([800.0, 600]), drive_cursor=True))
        a.tick(i / 120)
    assert ops.cursor == pytest.approx((800, 600), abs=1)


def test_set_frame_on_maximized_restores_first_then_maximizes_on_new_monitor(act):
    a, ops = act
    ops.add(5, Rect(0, 0, 1920, 1040), maximized=True)
    cmd(a, "set_frame", hwnd=5, rect=Rect(2000, 0, 1200, 700), maximize_after=True)
    for i in range(5):
        a.tick(0.01 + i / 120)
    k = ops.kinds()
    assert k.index("restore") < k.index("set_frame") < k.index("maximize")
