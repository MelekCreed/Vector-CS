"""Application carousel: hold palm -> browse by moving -> pinch to open."""

import sys

import numpy as np
import pytest

from vector.gestures.events import HandMode
from vector.sim.scenario import Scenario, run
from tests.test_engine import P, kinds, make
from tests.test_intent import rig, snap  # noqa: F401
from tests.test_review_regressions import _ev


def carousel_scenario(browse_dx=0.0, finish="pinch"):
    sc = Scenario()
    sc.key("R", 0, P("point", x=0.45)).key("R", 0.4, P("open", x=0.45)).key("R", 1.6, P("open", x=0.45))
    sc.key("R", 2.6, P("open", x=0.45 + browse_dx)).key("R", 2.9, P("open", x=0.45 + browse_dx))
    if finish == "pinch":
        sc.key("R", 2.95, P("pinch", x=0.45 + browse_dx)).key("R", 3.3, P("pinch", x=0.45 + browse_dx))
    else:
        sc.key("R", 2.95, P("fist", x=0.45 + browse_dx)).key("R", 3.6, P("fist", x=0.45 + browse_dx))
    return sc


def test_hold_palm_opens_browse_and_pinch_selects():
    pipe = make(active=True)
    ev, snaps = run(pipe, carousel_scenario(browse_dx=0.12))
    k = kinds(ev)
    assert k.count("carousel_open") == 1 and k.count("carousel_select") == 1
    offsets = [s.carousel_offset for s in snaps if s.carousel_offset is not None]
    assert offsets[0] == 0 and offsets[-1] >= 1                   # moving right browses right
    assert any(s.primary_mode == HandMode.CAROUSEL for s in snaps)
    sel = next(e for e in ev if e.kind == "carousel_select")
    assert sel.data["offset"] == offsets[-1]
    assert "swipe" not in k and "click" not in k and "pinch_start" not in k


def test_fist_dismisses_without_switching():
    pipe = make(active=True)
    ev, snaps = run(pipe, carousel_scenario(finish="fist"))
    k = kinds(ev)
    assert "carousel_open" in k and "carousel_close" in k and "carousel_select" not in k
    assert "hold" not in k                                        # dismiss fist isn't play/pause


def test_moving_palm_or_two_palms_do_not_open_carousel():
    pipe = make(active=True)
    sc = Scenario()
    sc.key("R", 0, P("open", x=0.3)).key("R", 2.5, P("open", x=0.7))
    ev, _ = run(pipe, sc)
    assert "carousel_open" not in kinds(ev)
    pipe = make(active=True)
    sc = Scenario()
    sc.key("R", 0, P("open", x=0.65)).key("R", 2.0, P("open", x=0.65))
    sc.key("L", 0, P("open", x=0.3, label="Left")).key("L", 2.0, P("open", x=0.3, label="Left"))
    ev, _ = run(pipe, sc)
    assert "carousel_open" not in kinds(ev) and "sleep" in kinds(ev)


def test_intent_carousel_starts_on_previous_app_and_focuses_selection(rig):  # noqa: F811
    eng, backend, cmds, _ = rig
    eng.handle([_ev("carousel_open")], snap(owner="carousel", carousel_offset=0))
    car = eng.fb.carousel
    assert [it[1] for it in car["items"]] == ["Editor", "Browser", "Music"] and car["index"] == 1
    eng.handle([], snap(owner="carousel", carousel_offset=1))
    assert eng.fb.carousel["index"] == 2
    eng.handle([_ev("carousel_select", offset=1)], snap())
    assert cmds[-1].kind == "focus" and cmds[-1].args["hwnd"] == 30
    assert eng.fb.carousel is None


def test_intent_carousel_closes_when_engine_leaves(rig):  # noqa: F811
    eng, _, cmds, _ = rig
    eng.handle([_ev("carousel_open")], snap(owner="carousel", carousel_offset=0))
    eng.handle([_ev("cancel", what="carousel", reason="hand lost")], snap(owner=None))
    assert eng.fb.carousel is None and not [c for c in cmds if c.kind == "focus"]


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 only")
def test_icon_handle_for_real_window():
    from tests.test_windows_real import target  # noqa: F401
    from vector.desktop import windows as W
    shell = W.user32.FindWindowW("Shell_TrayWnd", None)
    assert isinstance(W.icon_handle(shell or 0), int)            # never raises / blocks
