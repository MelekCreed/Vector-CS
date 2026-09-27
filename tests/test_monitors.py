"""Multi-monitor / mixed-DPI coordinate tests on synthetic layouts (the dev
machine has one monitor, so these are the primary safety net)."""

import pytest

from vector.core.geometry import Rect
from vector.desktop.monitors import Monitor, VirtualDesktop

# 4K @150% on the left at negative coords, 1080p @100% primary, portrait 1440p on the right.
LEFT = Monitor("L", Rect(-3840, -600, 3840, 2160), Rect(-3840, -600, 3840, 2100), dpi=144)
MAIN = Monitor("M", Rect(0, 0, 1920, 1080), Rect(0, 0, 1920, 1040), dpi=96, primary=True)
RIGHT = Monitor("R", Rect(1920, -400, 1440, 2560), Rect(1920, -400, 1440, 2560), dpi=120)
VD = VirtualDesktop([MAIN, LEFT, RIGHT])


def test_bounds_span_negative_coords():
    assert VD.bounds == Rect(-3840, -600, 3840 + 1920 + 1440, 2560 - 400 + 600)
    assert VD.primary is MAIN


def test_monitor_scale():
    assert LEFT.scale == 1.5 and RIGHT.scale == 1.25


@pytest.mark.parametrize("pt,name", [((-10, 0), "L"), ((0, 0), "M"), ((1919, 1079), "M"),
                                     ((1920, 0), "R"), ((2500, 2000), "R")])
def test_monitor_at(pt, name):
    assert VD.monitor_at(*pt).name == name


def test_dead_zone_points_clamp_onto_nearest_monitor():
    # Below the main monitor, which is shorter than its neighbours: a dead zone.
    assert VD.monitor_at(900, 1500) is None
    x, y = VD.clamp(900, 1500)
    assert VD.monitor_at(x, y).name == "M" and y == 1079


def test_from_unit_corners_land_on_real_pixels():
    for u, v in [(0, 0), (1, 1), (0, 1), (1, 0), (0.5, 0.5)]:
        assert VD.monitor_at(*VD.from_unit(u, v)) is not None


def test_monitor_in_direction():
    assert VD.monitor_in_direction(MAIN, 1, 0).name == "R"
    assert VD.monitor_in_direction(MAIN, -1, 0).name == "L"
    assert VD.monitor_in_direction(MAIN, 0, 1) is None        # nothing below
    assert VD.monitor_in_direction(MAIN, 0, 0) is None


def test_monitor_for_rect_uses_largest_overlap():
    straddling = Rect(1700, 100, 400, 300)                    # 220px on M, 180 on R
    assert VD.monitor_for_rect(straddling).name == "M"
    assert VD.monitor_for_rect(Rect(1800, 100, 400, 300)).name == "R"


def test_transfer_rect_preserves_relative_geometry_across_dpi():
    half_left = Rect(0, 0, 960, 1040)                         # left half of MAIN work area
    out = VirtualDesktop.transfer_rect(half_left, MAIN, LEFT)
    assert out == Rect(-3840, -600, 1920, 2100)               # left half of LEFT work area


def test_transfer_rect_clamps_into_destination():
    big = Rect(-3840, -600, 3840, 2100)
    out = VirtualDesktop.transfer_rect(big, LEFT, RIGHT)
    assert RIGHT.work.intersection_area(out) == pytest.approx(out.w * out.h)


def test_rect_fit_inside_shrinks_and_shifts():
    r = Rect(1800, -50, 400, 2000).fit_inside(MAIN.work)
    assert r == Rect(1520, 0, 400, 1040)


def test_single_monitor_system_enumeration_is_physical():
    """Real system check: the process must see physical pixels, not DPI-virtualised."""
    import sys
    if sys.platform != "win32":
        pytest.skip("windows only")
    from vector.desktop.dpi import make_process_dpi_aware
    make_process_dpi_aware()
    vd = VirtualDesktop.from_system()
    assert vd.monitors
    for m in vd.monitors:
        assert m.dpi >= 96
        assert m.work.w <= m.rect.w and m.work.h <= m.rect.h
