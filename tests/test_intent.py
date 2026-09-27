import math

import numpy as np
import pytest

from vector.config import Config
from vector.core.geometry import Rect
from vector.desktop.backend import FakeBackend, FakeWindow
from vector.desktop.monitors import Monitor, VirtualDesktop
from vector.gestures.confidence import Confidence
from vector.gestures.events import GestureEvent, HandMode, Snapshot, SystemState
from vector.intent import throw as TH
from vector.intent.actions import threshold_class_for
from vector.intent.intent import IntentEngine
from vector.intent.resize import resize_rect

MAIN = Monitor("M", Rect(0, 0, 1920, 1080), Rect(0, 0, 1920, 1040), primary=True)
RIGHT = Monitor("R", Rect(1920, 0, 2560, 1440), Rect(1920, 0, 2560, 1400), dpi=144)
ONE = VirtualDesktop([MAIN])
TWO = VirtualDesktop([MAIN, RIGHT])


# ------------------------------------------------------------------ throw
@pytest.mark.parametrize("vel,expect", [
    ((3000, 200), "snap_right"), ((-3000, -100), "snap_left"),
    ((100, -3000), "maximize"), ((0, 3000), "minimize"),
])
def test_throw_directions_single_monitor(vel, expect):
    d = TH.decide(Rect(600, 300, 700, 500), False, np.array([900, 500]), np.array(vel), ONE)
    assert d.action == expect
    if expect.startswith("snap"):
        side = expect.split("_")[1]
        assert d.rect == TH.half(MAIN.work, side)


def test_throw_down_on_maximized_restores():
    d = TH.decide(MAIN.work, True, np.array([900, 500]), np.array([0, 3000]), ONE)
    assert d.action == "restore"


def test_throw_toward_second_monitor_transfers_with_relative_geometry():
    frame = Rect(960, 0, 960, 1040)                   # right half of MAIN
    d = TH.decide(frame, False, np.array([1700, 500]), np.array([4000, 0]), TWO, project_s=0.28)
    assert d.action == "monitor" and d.monitor.name == "R"
    assert d.rect == Rect(1920 + 1280, 0, 1280, 1400)  # right half of R's work area


def test_short_throw_right_snaps_instead_of_transferring():
    d = TH.decide(Rect(200, 200, 600, 400), False, np.array([500, 400]), np.array([2500, 0]), TWO)
    assert d.action == "snap_right"


def test_throw_already_snapped_hops_monitor_like_win_arrow():
    d = TH.decide(TH.half(MAIN.work, "right"), False, np.array([1000, 500]),
                  np.array([2500, 0]), TWO)
    assert d.action == "monitor" and d.rect == TH.half(RIGHT.work, "left")


def test_throw_maximized_window_to_other_monitor_stays_maximized():
    d = TH.decide(MAIN.work, True, np.array([1800, 500]), np.array([5000, 0]), TWO)
    assert d.action == "monitor" and d.maximize_after


def test_direction_sectors():
    assert TH.direction_of(0) == "right" and TH.direction_of(math.pi) == "left"
    assert TH.direction_of(-math.pi / 2) == "up" and TH.direction_of(math.pi / 2) == "down"


# ------------------------------------------------------------------ resize
def test_resize_hands_apart_grows_both_axes_about_midpoint():
    start = Rect(600, 300, 600, 400)
    p0 = (np.array([700, 500.0]), np.array([1100, 500.0]))
    p = (np.array([500, 500.0]), np.array([1300, 500.0]))           # 2x horizontal separation
    r = resize_rect(start, p0, p, MAIN.work)
    assert r.w > start.w * 1.5
    assert r.h > start.h * 1.1                                      # uniform part grows height
    assert r.center == pytest.approx(start.center)


def test_resize_diagonal_separation_scales_axes_independently():
    start = Rect(600, 300, 600, 400)
    p0 = (np.array([700, 400.0]), np.array([1100, 700.0]))
    p = (np.array([700, 250.0]), np.array([1100, 850.0]))           # only vertical grows
    r = resize_rect(start, p0, p, MAIN.work)
    assert r.h / start.h > r.w / start.w


def test_resize_respects_min_size_and_screen():
    start = Rect(600, 300, 600, 400)
    p0 = (np.array([700, 500.0]), np.array([1100, 500.0]))
    tiny = resize_rect(start, p0, (np.array([895, 500.0]), np.array([905, 500.0])), MAIN.work)
    assert tiny.w >= 360 and tiny.h >= 240
    huge = resize_rect(start, p0, (np.array([-3000, -2000.0]), np.array([5000, 3000.0])), MAIN.work)
    assert huge.w <= MAIN.work.w and huge.h <= MAIN.work.h
    assert MAIN.work.intersection_area(huge) == pytest.approx(huge.w * huge.h)


# ------------------------------------------------------------------ intent engine
def ev(kind, t=1.0, conf=0.95, interaction=1, **data):
    return GestureEvent(kind, t, Confidence(conf, {"geometry": conf}), 1, interaction, data)


def snap(t=1.0, **kw):
    base = dict(t=t, system=SystemState.ACTIVE, draw_mode=False, hands=[], cursor=np.array([500.0, 400]),
                cursor_visible=True, primary_mode=HandMode.POINTING, interaction=0, owner=None)
    base.update(kw)
    return Snapshot(**base)


@pytest.fixture
def rig():
    backend = FakeBackend([
        FakeWindow(10, "Editor", Rect(100, 100, 800, 600), "code.exe"),
        FakeWindow(20, "Browser", Rect(300, 200, 1200, 700), "chrome.exe"),
        FakeWindow(30, "Music", Rect(1000, 500, 600, 400), "spotify.exe"),
    ])
    cmds = []
    toggles = []
    eng = IntentEngine(Config(), ONE, backend, cmds.append, lambda: 7,
                       on_toggle_draw=toggles.append)
    return eng, backend, cmds, toggles


def test_pinch_locks_target_and_click_goes_through(rig):
    eng, _, cmds, _ = rig
    eng.handle([ev("pinch_start", point=np.array([400.0, 300]))], snap())
    assert eng.fb.lock_state == "pinch" and eng.fb.lock_title == "Editor"
    eng.handle([ev("click", point=np.array([400.0, 300]))], snap())
    assert [c.kind for c in cmds] == ["click"] and cmds[0].generation == 7


def test_low_confidence_click_is_rejected_and_logged(rig):
    eng, _, cmds, _ = rig
    eng.handle([ev("click", conf=0.4, point=np.array([1, 1.0]))], snap())
    assert cmds == []
    assert not eng.history[-1].accepted and "confidence" in eng.history[-1].reason


def test_drag_locks_window_even_when_pointer_leaves_it(rig):
    eng, _, cmds, _ = rig
    eng.handle([ev("pinch_start", point=np.array([150.0, 150]))], snap())
    eng.handle([ev("drag_start", point=np.array([160.0, 150]), anchor=np.array([150.0, 150]))], snap())
    assert cmds[-1].kind == "drag_begin" and cmds[-1].args["hwnd"] == 10
    s = snap(drag_point=np.array([1800.0, 900]), primary_mode=HandMode.DRAGGING)
    eng.handle([], s)
    assert np.allclose(eng.last_motion.drag_point, [1800, 900])   # still following, still hwnd 10
    assert eng.fb.lock_state == "drag"


def test_drag_on_empty_desktop_is_refused(rig):
    eng, _, cmds, _ = rig
    eng.handle([ev("pinch_start", point=np.array([1900.0, 50]))], snap())
    eng.handle([ev("drag_start", point=np.array([1900.0, 60]), anchor=np.array([1900.0, 50]))], snap())
    assert cmds == []


def test_throw_issues_snap(rig):
    eng, _, cmds, _ = rig
    eng.handle([ev("pinch_start", point=np.array([150.0, 150]))], snap())
    eng.handle([ev("drag_start", point=np.array([160.0, 150]), anchor=np.array([150.0, 150]))], snap())
    eng.handle([ev("throw", release_point=np.array([500.0, 300]), velocity_px=np.array([-3000.0, 0]),
                   angle=math.pi, speed=4.0)], snap())
    kinds = [c.kind for c in cmds]
    assert kinds == ["drag_begin", "drag_end", "set_frame"]
    assert cmds[-1].args["rect"] == TH.half(MAIN.work, "left")
    assert eng.fb.throw_trail is not None


def test_app_switch_cycles_frozen_zorder_then_resets(rig):
    eng, _, cmds, _ = rig
    for i, t in enumerate((1.0, 2.0, 3.0)):
        eng.handle([ev("swipe", t=t, family="palm", direction="right")], snap(t=t))
    targets = [c.args["hwnd"] for c in cmds if c.kind == "focus"]
    assert targets == [20, 30, 10]                                  # walks the frozen list
    eng.handle([ev("swipe", t=10.0, family="palm", direction="left")], snap(t=10.0))
    assert cmds[-1].args["hwnd"] == 30                              # new session: last in z-order


def test_swipe_cooldown(rig):
    eng, _, cmds, _ = rig
    eng.handle([ev("swipe", t=1.0, family="palm", direction="right")], snap())
    eng.handle([ev("swipe", t=1.2, family="palm", direction="right")], snap())
    assert sum(c.kind == "focus" for c in cmds) == 1
    assert "cooldown" in eng.history[-1].reason


def test_two_finger_swipe_is_browser_back_in_browser_else_media(rig):
    eng, backend, cmds, _ = rig
    backend.focus(20)                                               # chrome in front
    eng.handle([ev("swipe", t=1.0, family="two_finger", direction="left")], snap())
    assert cmds[-1].args["chord"] == "browser_back"
    backend.focus(30)                                               # spotify in front
    eng.handle([ev("swipe", t=5.0, family="two_finger", direction="right")], snap(t=5.0))
    assert cmds[-1].args["chord"] == "media_next"


def test_fist_hold_is_play_pause_but_clear_in_draw_mode(rig):
    eng, _, cmds, _ = rig
    eng.handle([ev("hold", t=1.0, pose="fist")], snap())
    assert cmds[-1].args["chord"] == "media_play_pause"
    fb = eng.handle([ev("hold", t=5.0, pose="fist")], snap(t=5.0, draw_mode=True))
    assert ("draw_clear", None) in fb.draw_ops


def test_three_finger_hold_toggles_draw(rig):
    eng, _, _, toggles = rig
    eng.handle([ev("hold", t=1.0, pose="three_finger")], snap())
    assert toggles == [1.0]


def test_custom_key_binding(rig):
    eng, _, cmds, _ = rig
    eng.cfg.bindings["palm_swipe_right"] = "keys:ctrl+shift+t"
    eng.handle([ev("swipe", t=1.0, family="palm", direction="right")], snap())
    assert cmds[-1].kind == "keys" and cmds[-1].args["chord"] == "ctrl+shift+t"


def test_unknown_binding_is_rejected(rig):
    eng, _, cmds, _ = rig
    eng.cfg.bindings["palm_swipe_right"] = "format_c_drive"
    eng.handle([ev("swipe", t=1.0, family="palm", direction="right")], snap())
    assert cmds == [] and "unknown action" in eng.history[-1].reason


def test_threshold_classes():
    cfg = Config()
    assert threshold_class_for(ev("swipe", family="palm", direction="left"), cfg) == "app_switch"
    assert threshold_class_for(ev("hold", pose="fist"), cfg) == "media"
    assert threshold_class_for(ev("throw"), cfg) == "throw"
    assert threshold_class_for(ev("click"), cfg) == "click"


def test_resize_flow_produces_bounded_rect(rig):
    eng, _, cmds, _ = rig
    eng.handle([ev("pinch_start", point=np.array([300.0, 300]))], snap())
    eng.handle([ev("drag_start", point=np.array([310.0, 300]), anchor=np.array([300.0, 300]))], snap())
    p0 = (np.array([300.0, 400]), np.array([700.0, 400]))
    eng.handle([ev("resize_start", points=p0)], snap())
    assert [c.kind for c in cmds][-2:] == ["drag_end", "resize_begin"]
    eng.handle([], snap(resize_points=(np.array([100.0, 400]), np.array([900.0, 400]))))
    r = eng.last_motion.resize_rect
    assert r is not None and r.w > 800 and MAIN.work.intersection_area(r) == pytest.approx(r.w * r.h)
