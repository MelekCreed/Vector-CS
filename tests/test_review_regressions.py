"""Regression tests for the findings of the Codex final review (handoff 002).
Each test reproduces the reported failure scenario."""

import numpy as np

from vector.config import Config
from vector.core.metrics import Metrics
from vector.desktop.actuator import Actuator
from vector.gestures.confidence import Confidence
from vector.gestures.events import GestureEvent, SystemState
from vector.overlay.canvas import Canvas
from vector.pipeline import GesturePipeline
from vector.runtime import SharedState, VisionWorker
from vector.sim.scenario import Scenario, run
from tests.test_actuator import FakeOps
from tests.test_engine import DESK, P, kinds, make
from tests.test_intent import rig, snap  # noqa: F401  (fixture)


def _drain(worker):
    while not worker._requests.empty():
        worker._requests.get_nowait()()


# [P1] a re-arm queued before an emergency stop must not undo the newer stop
def test_stale_rearm_cannot_undo_newer_emergency_stop():
    cfg = Config()
    cfg.activation.require_activation = False
    pipe = GesturePipeline(cfg, DESK)
    act = Actuator(cfg, FakeOps())
    w = VisionWorker(cfg, None, None, pipe, None, act, SharedState(), Canvas(["#fff"]), Metrics())
    w.rearm()                 # queued first ...
    w.emergency_stop()        # ... then Esc Esc
    _drain(w)
    assert w.disabled and act.latched and act.arm() is False
    w.rearm()                 # a fresh re-arm after the stop works
    _drain(w)
    assert not w.disabled and act.arm() is True


# [P1] holding the sleep palms must not immediately wake the system again
def test_sleep_palms_held_do_not_rewake():
    pipe = make(active=True)
    sc = Scenario()
    sc.key("R", 0, P("point", x=0.65)).key("R", 0.5, P("open", x=0.65)).key("R", 4.0, P("open", x=0.65))
    sc.key("L", 0, P("open", x=0.3, label="Left")).key("L", 4.0, P("open", x=0.3, label="Left"))
    ev, snaps = run(pipe, sc)
    assert kinds(ev).count("sleep") == 1 and "wake" not in kinds(ev)
    assert snaps[-1].system == SystemState.SLEEPING
    # Dropping the palms, then raising one again, is a genuine new wake.
    sc2 = Scenario()
    sc2.key("R", 0, P("fist", x=0.65)).key("R", 0.6, P("fist", x=0.65))
    sc2.key("R", 0.7, P("open", x=0.65)).key("R", 1.8, P("open", x=0.65))
    ev2, _ = run(pipe, sc2, t0=5.0)
    assert "wake" in kinds(ev2)


# [P2] right click must fire once its evidence has matured
def test_right_click_passes_click_threshold():
    pipe = make(active=True)
    sc = Scenario()
    sc.key("R", 0, P("point")).key("R", 0.6, P("point"))
    sc.key("R", 0.65, P("middle_pinch")).key("R", 1.2, P("middle_pinch")).key("R", 1.25, P("point"))
    ev, _ = run(pipe, sc)
    rc = [e for e in ev if e.kind == "right_click"]
    assert len(rc) == 1 and rc[0].confidence.value >= Config().threshold("click")


# [P2] resize that ends without a drag must release the cursor lock and target
def test_resize_end_without_drag_releases_locks():
    pipe = make(active=True)
    sc = Scenario()
    sc.key("R", 0, P("point", x=0.6)).key("R", 0.5, P("point", x=0.6))
    sc.key("R", 0.55, P("pinch", x=0.6)).key("R", 1.5, P("pinch", x=0.6))      # stationary grab
    sc.key("L", 0, P("point", x=0.35, label="Left")).key("L", 1.0, P("point", x=0.35, label="Left"))
    sc.key("L", 1.05, P("pinch", x=0.35, label="Left")).key("L", 1.8, P("pinch", x=0.30, label="Left"))
    sc.key("R", 1.55, P("point", x=0.6)).key("R", 2.5, P("point", x=0.75))    # primary releases, moves
    ev, snaps = run(pipe, sc)
    k = kinds(ev)
    assert "resize_start" in k and "release" in k[k.index("resize_end"):]
    late = [s.cursor for s in snaps if s.t > 2.2]
    assert np.linalg.norm(late[-1] - late[0]) > 20        # cursor follows the hand again


# [P2] two raised palms must put draw mode to sleep instead of erasing forever
def test_sleep_reachable_in_draw_mode():
    pipe = make(active=True)
    pipe.engine.toggle_draw(0.0)
    sc = Scenario()
    sc.key("R", 0, P("point", x=0.65)).key("R", 0.4, P("open", x=0.65)).key("R", 2.0, P("open", x=0.65))
    sc.key("L", 0, P("open", x=0.3, label="Left")).key("L", 2.0, P("open", x=0.3, label="Left"))
    ev, snaps = run(pipe, sc)
    assert "sleep" in kinds(ev) and snaps[-1].system == SystemState.SLEEPING


# [P2] toggling draw mode mid-drag must deliver teardown to the intent layer
def test_toggle_draw_mid_drag_releases_window(rig):  # noqa: F811
    eng_intent, backend, cmds, _ = rig
    pipe = make(active=True)
    sc = Scenario()
    sc.key("R", 0, P("point", x=0.4)).key("R", 0.5, P("point", x=0.4))
    sc.key("R", 0.55, P("pinch", x=0.4)).key("R", 1.5, P("pinch", x=0.5))
    got = []
    for t, dets in sc.frames():
        ev, s = pipe.process(dets, t, sc.aspect)
        got += ev
        if t >= 1.2 and not pipe.engine.draw_mode:
            pipe.engine.toggle_draw(t)
    assert "drag_start" in kinds(got)
    cancel = [e for e in got if e.kind == "cancel"]
    assert cancel and cancel[0].data["what"] == "drag"
    assert pipe.engine.inter is None or pipe.engine.inter.kind != "drag"


# [P2] leaning toward the camera must not manufacture travel
def test_depth_change_does_not_create_drag_or_volume():
    pipe = make(active=True)
    sc = Scenario(noise=0.0)
    sc.key("R", 0, P("point", scale=2.6)).key("R", 0.5, P("point", scale=2.6))
    sc.key("R", 0.55, P("pinch", scale=2.6)).key("R", 0.6, P("pinch", scale=2.6))
    sc.key("R", 1.6, P("pinch", scale=3.4))                 # approach the camera, don't move
    sc.key("R", 1.65, P("point", scale=3.4)).key("R", 2.0, P("shaka", scale=3.4))
    sc.key("R", 2.4, P("shaka", scale=3.4)).key("R", 3.6, P("shaka", scale=2.5))
    ev, _ = run(pipe, sc)
    k = kinds(ev)
    assert "drag_start" not in k and "volume_step" not in k


# [P2] configured bindings for middle pinch and shaka are honoured
def _ev(kind, **data):
    return GestureEvent(kind, 1.0, Confidence(0.95, {}), 1, 0, data)


def test_middle_pinch_binding_none_suppresses_right_click(rig):  # noqa: F811
    eng, _, cmds, _ = rig
    eng.cfg.bindings["middle_pinch"] = "none"
    eng.handle([_ev("right_click", point=np.array([10.0, 10]))], snap())
    assert cmds == []


def test_shaka_binding_remap(rig):  # noqa: F811
    eng, _, cmds, _ = rig
    eng.cfg.bindings["shaka_vertical"] = "keys:ctrl+tab"
    eng.handle([_ev("volume_step", direction=1)], snap())
    assert [c.args.get("chord") for c in cmds] == ["ctrl+tab"]


# [P2] completed stroke geometry is cached, not recomputed every repaint
def test_canvas_render_list_caches_completed_strokes():
    cv = Canvas(["#fff"])
    cv.begin((0, 0))
    for x in range(5, 400, 5):
        cv.add((x, (x % 40) * 2.0))
    cv.end()
    a = cv.render_list()[0][2]
    b = cv.render_list()[0][2]
    assert a is b
