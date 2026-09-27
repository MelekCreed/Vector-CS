"""Temporal gesture engine: scripted scenarios through the full pipeline
(synthetic landmarks -> identity -> features -> poses -> state machine)."""

import math

import numpy as np
import pytest

from vector.config import Config
from vector.core.geometry import Rect
from vector.desktop.monitors import Monitor, VirtualDesktop
from vector.gestures.events import HandMode, SystemState
from vector.pipeline import GesturePipeline
from vector.sim.scenario import Scenario, run
from vector.sim.synthetic_hand import HandPose

DESK = VirtualDesktop([Monitor("M", Rect(0, 0, 1920, 1080), Rect(0, 0, 1920, 1040), primary=True)])
COMMANDS = {"click", "right_click", "throw", "swipe", "hold", "volume_step", "drag_start",
            "pinch_start", "resize_start", "wake", "sleep"}


def make(active=False, **cfg_over) -> GesturePipeline:
    cfg = Config()
    cfg.activation.start_active = active
    for k, v in cfg_over.items():
        section, key = k.split("__")
        setattr(getattr(cfg, section), key, v)
    return GesturePipeline(cfg, DESK)


def kinds(events):
    return [e.kind for e in events]


def P(pose="point", x=0.5, y=0.45, **kw):
    return HandPose(pose=pose, x=x, y=y, **kw)


def woke(sc: Scenario, t_end=1.0) -> float:
    """Palm held still, then relax to a point (neutral) — the wake ritual."""
    sc.key("R", 0.0, P("open")).key("R", t_end, P("open")).key("R", t_end + 0.05, P("point"))
    return t_end + 0.4


# --------------------------------------------------------------- activation
def test_sleeping_system_ignores_everything_but_wake():
    pipe = make(active=False)
    sc = Scenario()
    sc.key("R", 0, P("point")).key("R", 0.3, P("pinch")).key("R", 0.5, P("point"))
    sc.key("R", 0.6, P("fist")).key("R", 1.5, P("fist"))
    ev, snaps = run(pipe, sc)
    assert not (set(kinds(ev)) & COMMANDS)
    assert snaps[-1].system == SystemState.SLEEPING
    assert not snaps[-1].cursor_visible


def test_wake_requires_held_still_palm():
    pipe = make(active=False)
    sc = Scenario()
    woke(sc)
    ev, snaps = run(pipe, sc)
    assert kinds(ev).count("wake") == 1
    assert snaps[-1].system == SystemState.ACTIVE


def test_moving_palm_does_not_wake():
    pipe = make(active=False)
    sc = Scenario()
    sc.key("R", 0, P("open", x=0.2)).key("R", 1.2, P("open", x=0.8)).key("R", 2.4, P("open", x=0.2))
    ev, _ = run(pipe, sc)
    assert "wake" not in kinds(ev)


def test_wake_palm_cannot_immediately_swipe():
    """Neutral reset: the palm that woke the system must relax before commands."""
    pipe = make(active=False)
    sc = Scenario()
    sc.key("R", 0, P("open", x=0.5)).key("R", 0.9, P("open", x=0.5))
    sc.key("R", 1.0, P("open", x=0.5)).key("R", 1.25, P("open", x=0.85))   # fast swipe right
    ev, snaps = run(pipe, sc)
    assert "wake" in kinds(ev)
    assert "swipe" not in kinds(ev)


def test_two_palms_put_system_to_sleep():
    pipe = make(active=True)
    sc = Scenario()
    sc.key("R", 0, P("point", x=0.65)).key("R", 0.5, P("open", x=0.65)).key("R", 1.6, P("open", x=0.65))
    sc.key("L", 0, P("open", x=0.3, label="Left")).key("L", 1.6, P("open", x=0.3, label="Left"))
    ev, snaps = run(pipe, sc)
    assert kinds(ev).count("sleep") == 1
    assert snaps[-1].system == SystemState.SLEEPING


# --------------------------------------------------------------- click / drag
def test_quick_pinch_is_a_click_at_pre_pinch_position():
    pipe = make(active=True)
    sc = Scenario(noise=0.0)
    sc.key("R", 0, P("point")).key("R", 0.8, P("point"))
    sc.key("R", 0.85, P("pinch")).key("R", 1.0, P("pinch")).key("R", 1.05, P("point"))
    sc.key("R", 1.5, P("point"))
    ev, _ = run(pipe, sc)
    k = kinds(ev)
    assert k.count("pinch_start") == 1 and k.count("click") == 1
    assert "drag_start" not in k and "throw" not in k
    click = next(e for e in ev if e.kind == "click")
    start = next(e for e in ev if e.kind == "pinch_start")
    assert np.allclose(click.data["point"], start.data["point"])
    assert click.confidence.value > 0.7


def test_pinch_hold_move_slow_release_is_drag_not_throw():
    pipe = make(active=True)
    sc = Scenario()
    sc.key("R", 0, P("point")).key("R", 0.5, P("point"))
    sc.key("R", 0.55, P("pinch", x=0.4)).key("R", 0.9, P("pinch", x=0.4))
    sc.key("R", 2.4, P("pinch", x=0.6))              # slow drag right
    sc.key("R", 2.8, P("pinch", x=0.6)).key("R", 2.85, P("point", x=0.6)).key("R", 3.2, P("point", x=0.6))
    ev, snaps = run(pipe, sc)
    k = kinds(ev)
    assert k.count("drag_start") == 1 and k.count("drag_end") == 1
    assert "throw" not in k and "click" not in k
    assert any(s.primary_mode == HandMode.DRAGGING for s in snaps)
    moving = [s.drag_point[0] for s in snaps if s.drag_point is not None]
    assert moving[-1] - moving[0] > 150                # the window followed the hand


def fast_throw(direction=(1, 0), speed_units=0.25, label="R"):
    sc = Scenario()
    dx, dy = direction
    sc.key(label, 0, P("point")).key(label, 0.5, P("point"))
    sc.key(label, 0.55, P("pinch")).key(label, 0.8, P("pinch"))
    sc.key(label, 1.0, P("pinch", x=0.5 + dx * 0.05, y=0.45 + dy * 0.05))
    sc.key(label, 1.25, P("pinch", x=0.5 + dx * (0.05 + speed_units), y=0.45 + dy * (0.05 + speed_units)))
    sc.key(label, 1.26, P("point", x=0.5 + dx * (0.05 + speed_units), y=0.45 + dy * (0.05 + speed_units)))
    sc.key(label, 1.6, P("point", x=0.5 + dx * (0.05 + speed_units), y=0.45 + dy * (0.05 + speed_units)))
    return sc


@pytest.mark.parametrize("direction,expect", [((1, 0), 0.0), ((-1, 0), math.pi), ((0, -1), -math.pi / 2)])
def test_fast_release_while_dragging_is_a_throw(direction, expect):
    pipe = make(active=True)
    ev, _ = run(pipe, fast_throw(direction))
    throws = [e for e in ev if e.kind == "throw"]
    assert len(throws) == 1, kinds(ev)
    th = throws[0]
    assert abs(math.remainder(th.data["angle"] - expect, 2 * math.pi)) < 0.35
    vx, vy = th.data["velocity_px"]
    assert np.sign(vx) == np.sign(direction[0]) or direction[0] == 0
    assert th.confidence.value > 0.6


def test_downward_throw_needs_more_speed():
    pipe = make(active=True)
    ev, _ = run(pipe, fast_throw((0, 1), speed_units=0.13))
    assert "throw" not in kinds(ev)


def test_hand_lost_mid_drag_cancels_without_click_or_throw():
    pipe = make(active=True)
    sc = Scenario()
    sc.key("R", 0, P("point")).key("R", 0.5, P("point"))
    sc.key("R", 0.55, P("pinch")).key("R", 0.9, P("pinch", x=0.55)).key("R", 1.1, P("pinch", x=0.75))
    sc.key("R", 1.12, None)                          # tracking lost while moving fast
    sc.key("L", 1.12, None).key("L", 2.0, None)
    ev, _ = run(pipe, sc)
    k = kinds(ev)
    assert "cancel" in k
    assert "throw" not in k and "click" not in k and "drag_end" not in k


def test_brief_dropout_does_not_break_drag():
    pipe = make(active=True)
    sc = Scenario(drop_rate=0.12, seed=3)
    sc.key("R", 0, P("point")).key("R", 0.5, P("point"))
    sc.key("R", 0.55, P("pinch", x=0.4)).key("R", 2.5, P("pinch", x=0.6))
    sc.key("R", 2.9, P("pinch", x=0.6)).key("R", 2.95, P("point", x=0.6)).key("R", 3.3, P("point", x=0.6))
    ev, _ = run(pipe, sc)
    k = kinds(ev)
    assert "cancel" not in k and k.count("drag_start") == 1 and k.count("drag_end") == 1


# --------------------------------------------------------------- resize
def test_two_hand_resize_lifecycle():
    pipe = make(active=True)
    sc = Scenario()
    sc.key("R", 0, P("point", x=0.6)).key("R", 0.5, P("point", x=0.6))
    sc.key("R", 0.55, P("pinch", x=0.6)).key("R", 3.0, P("pinch", x=0.6))
    sc.key("L", 0, P("point", x=0.35, label="Left")).key("L", 1.1, P("point", x=0.35, label="Left"))
    sc.key("L", 1.15, P("pinch", x=0.35, label="Left")).key("L", 1.4, P("pinch", x=0.35, label="Left"))
    sc.key("L", 2.4, P("pinch", x=0.15, label="Left"))                      # hands move apart
    sc.key("L", 2.45, P("point", x=0.15, label="Left")).key("L", 3.0, P("point", x=0.15, label="Left"))
    ev, snaps = run(pipe, sc)
    k = kinds(ev)
    assert k.count("resize_start") == 1 and k.count("resize_end") == 1
    pts = [s.resize_points for s in snaps if s.resize_points is not None]
    d0 = np.linalg.norm(pts[0][0] - pts[0][1])
    d1 = np.linalg.norm(pts[-1][0] - pts[-1][1])
    assert d1 > d0 * 1.4
    # Primary kept pinching -> drag resumes (rebased), no throw.
    assert k[k.index("resize_end") + 1] == "drag_start"
    assert "throw" not in k


# --------------------------------------------------------------- swipes
def palm_swipe(sc, t, x0, x1, dur=0.22, hand="R"):
    sc.key(hand, t, P("open", x=x0)).key(hand, t + dur, P("open", x=x1))


def test_palm_swipe_fires_once_and_return_stroke_is_ignored():
    pipe = make(active=True)
    sc = Scenario()
    sc.key("R", 0, P("point", x=0.3)).key("R", 0.5, P("point", x=0.3))
    sc.key("R", 0.6, P("open", x=0.3)).key("R", 1.0, P("open", x=0.3))
    palm_swipe(sc, 1.0, 0.3, 0.7)                     # swipe right
    sc.key("R", 1.5, P("open", x=0.7))
    palm_swipe(sc, 1.5, 0.7, 0.3, dur=0.3)           # bring the hand back
    sc.key("R", 2.3, P("open", x=0.3))
    ev, _ = run(pipe, sc)
    sw = [e for e in ev if e.kind == "swipe"]
    assert [(e.data["family"], e.data["direction"]) for e in sw] == [("palm", "right")]
    assert sw[0].confidence.value > 0.7


def test_slow_palm_movement_is_not_a_swipe():
    pipe = make(active=True)
    sc = Scenario()
    sc.key("R", 0, P("point", x=0.3)).key("R", 0.4, P("open", x=0.3)).key("R", 0.8, P("open", x=0.3))
    sc.key("R", 3.0, P("open", x=0.75))
    ev, _ = run(pipe, sc)
    assert "swipe" not in kinds(ev)


def test_two_finger_swipe_family():
    pipe = make(active=True)
    sc = Scenario()
    sc.key("R", 0, P("point", x=0.7)).key("R", 0.4, P("two", x=0.7)).key("R", 0.9, P("two", x=0.7))
    sc.key("R", 1.12, P("two", x=0.3)).key("R", 1.6, P("two", x=0.3))
    ev, _ = run(pipe, sc)
    sw = [e for e in ev if e.kind == "swipe"]
    assert [(e.data["family"], e.data["direction"]) for e in sw] == [("two_finger", "left")]


# --------------------------------------------------------------- scroll / holds / volume
def test_two_finger_vertical_scrolls_in_hand_direction_with_inertia():
    pipe = make(active=True)
    sc = Scenario()
    sc.key("R", 0, P("point")).key("R", 0.3, P("two", y=0.6)).key("R", 0.7, P("two", y=0.6))
    sc.key("R", 1.2, P("two", y=0.35))               # hand moves UP
    sc.key("R", 1.25, P("fist", y=0.35)).key("R", 1.6, P("fist", y=0.35))
    ev, snaps = run(pipe, sc)
    during = [s.scroll_velocity for s in snaps if 0.8 < s.t < 1.2]
    assert max(during) > 200                         # scrolls up (positive)
    after = [s.scroll_velocity for s in snaps if 1.3 < s.t < 1.4]
    assert any(v > 0 for v in after)                 # inertia coasts
    assert snaps[-1].scroll_velocity < max(during) * 0.5
    assert "swipe" not in kinds(ev)


def test_fist_hold_fires_once():
    pipe = make(active=True)
    sc = Scenario()
    sc.key("R", 0, P("point")).key("R", 0.3, P("fist")).key("R", 2.0, P("fist"))
    ev, _ = run(pipe, sc)
    holds = [e for e in ev if e.kind == "hold"]
    assert [e.data["pose"] for e in holds] == ["fist"]


def test_moving_fist_is_not_a_hold():
    pipe = make(active=True)
    sc = Scenario()
    sc.key("R", 0, P("fist", x=0.3)).key("R", 1.5, P("fist", x=0.8))
    ev, _ = run(pipe, sc)
    assert "hold" not in kinds(ev)


def test_shaka_vertical_volume_steps():
    pipe = make(active=True)
    sc = Scenario()
    sc.key("R", 0, P("point", y=0.6)).key("R", 0.3, P("shaka", y=0.6)).key("R", 0.8, P("shaka", y=0.6))
    sc.key("R", 1.8, P("shaka", y=0.35)).key("R", 2.0, P("shaka", y=0.35))
    ev, _ = run(pipe, sc)
    steps = [e.data["direction"] for e in ev if e.kind == "volume_step"]
    assert len(steps) >= 2 and all(s == 1 for s in steps)


# --------------------------------------------------------------- draw mode
def test_draw_mode_pinch_strokes_and_palm_erases():
    pipe = make(active=True)
    pipe.engine.toggle_draw(0.0)
    sc = Scenario()
    sc.key("R", 0, P("point", x=0.4)).key("R", 0.4, P("point", x=0.4))
    sc.key("R", 0.45, P("pinch", x=0.4)).key("R", 1.2, P("pinch", x=0.6))
    sc.key("R", 1.25, P("point", x=0.6)).key("R", 1.5, P("open", x=0.6)).key("R", 2.0, P("open", x=0.5))
    ev, _ = run(pipe, sc)
    k = kinds(ev)
    assert k.count("stroke_begin") == 1 and k.count("stroke_end") == 1
    assert k.count("stroke_point") > 10
    assert "erase_point" in k
    assert "drag_start" not in k and "click" not in k


# --------------------------------------------------------------- negative / robustness
def test_everyday_motion_produces_no_commands():
    """Waving, relaxed hands, scratching (hand near face region, erratic) —
    with realistic noise and dropouts, nothing may fire."""
    rng = np.random.default_rng(42)
    pipe = make(active=True)
    pipe.engine.neutral_required = False
    sc = Scenario(noise=0.003, drop_rate=0.05, seed=7)
    t = 0.0
    for _ in range(60):                              # ~24 s of aimless motion
        pose = rng.choice(["relaxed", "relaxed", "open", "relaxed"])
        x, y = rng.uniform(0.25, 0.75), rng.uniform(0.2, 0.8)
        sc.key("R", t, P(str(pose), x=x, y=y, roll=rng.uniform(-0.6, 0.6)))
        t += rng.uniform(0.3, 0.6)                   # moderate speeds
    ev, _ = run(pipe, sc)
    fired = [e for e in ev if e.kind in COMMANDS and e.confidence.value >= 0.7]
    assert fired == [], fired


def test_disabled_state_blocks_everything():
    pipe = make(active=True)
    pipe.engine.set_system(SystemState.DISABLED, 0.0)
    ev, snaps = run(pipe, fast_throw())
    assert not (set(kinds(ev)) & COMMANDS)
    assert snaps[-1].system == SystemState.DISABLED


def test_interaction_owner_is_exclusive_during_drag():
    """A palm swipe performed by the other hand mid-drag must not fire."""
    pipe = make(active=True)
    sc = Scenario()
    sc.key("R", 0, P("point", x=0.65)).key("R", 0.5, P("point", x=0.65))
    sc.key("R", 0.55, P("pinch", x=0.65)).key("R", 2.5, P("pinch", x=0.7))
    sc.key("L", 0, P("open", x=0.2, label="Left")).key("L", 1.0, P("open", x=0.2, label="Left"))
    sc.key("L", 1.25, P("open", x=0.45, label="Left")).key("L", 2.5, P("open", x=0.45, label="Left"))
    ev, _ = run(pipe, sc)
    assert "swipe" not in kinds(ev)


def test_pinch_onset_does_not_move_cursor_or_window():
    """Regression: switching the pointer source at pinch onset (fingertip ->
    palm) once made grabbed windows glide ~360 px. The anchored pointer must be
    continuous: holding a pinch still keeps the cursor still, and a purely
    horizontal drag moves the window horizontally only."""
    pipe = make(active=True)
    sc = Scenario(noise=0.0)
    sc.key("R", 0, P("point", x=0.5, y=0.62)).key("R", 0.8, P("point", x=0.5, y=0.62))
    sc.key("R", 0.85, P("pinch", x=0.5, y=0.62)).key("R", 1.6, P("pinch", x=0.5, y=0.62))
    sc.key("R", 2.4, P("pinch", x=0.62, y=0.62)).key("R", 2.8, P("pinch", x=0.62, y=0.62))
    ev, snaps = run(pipe, sc)
    before = next(s.cursor for s in reversed(snaps) if s.t < 0.8)
    held = [s.cursor for s in snaps if 1.0 < s.t < 1.6]
    assert max(np.linalg.norm(c - before) for c in held) < 12
    drag = [s.drag_point for s in snaps if s.drag_point is not None]
    grabbed_at = next(e.data["point"] for e in ev if e.kind == "pinch_start")
    assert np.linalg.norm(drag[0] - grabbed_at) < 40        # no jump when the drag begins
    dx = drag[-1][0] - drag[0][0]
    dy = drag[-1][1] - drag[0][1]
    assert dx > 150 and abs(dy) < 0.08 * dx
