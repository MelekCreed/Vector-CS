import numpy as np
import pytest

from vector.calibration import TARGET_FRACS, Calibrator, Step, region_from_samples
from vector.config import Config
from vector.pipeline import GesturePipeline
from vector.sim.scenario import Scenario
from vector.sim.synthetic_hand import HandPose
from tests.test_engine import DESK


def test_region_extrapolates_from_20_80_targets():
    # Camera region truly spans x 0.3..0.7, y 0.2..0.6
    samples = [(0.3 + 0.4 * u, 0.2 + 0.4 * v) for u, v in TARGET_FRACS]
    r = region_from_samples(samples)
    assert r == pytest.approx((0.3, 0.2, 0.7, 0.6), abs=1e-6)


def test_region_is_robust_to_one_sloppy_corner():
    samples = [(0.3 + 0.4 * u, 0.2 + 0.4 * v) for u, v in TARGET_FRACS]
    samples[2] = (samples[2][0] + 0.03, samples[2][1] - 0.02)
    r = region_from_samples(samples)
    assert r == pytest.approx((0.3, 0.2, 0.7, 0.6), abs=0.04)


def test_degenerate_samples_rejected():
    with pytest.raises(ValueError):
        region_from_samples([(0.5, 0.5)] * 4)


def test_full_calibration_flow_with_synthetic_hand():
    cfg = Config()
    pipe = GesturePipeline(cfg, DESK)
    cal = Calibrator(cfg)
    sc = Scenario(noise=0.001)
    t = 0.0
    sc.key("L", t, HandPose("point", x=0.5, y=0.5, label="Left"))
    t += 1.6
    # Region we "want": camera x 0.35..0.65, y 0.25..0.65 for the palm-centred
    # synthetic hand; the fingertip offset is constant so the fit absorbs it.
    for u, v in TARGET_FRACS:
        pos = HandPose("point", x=0.35 + 0.3 * u, y=0.25 + 0.4 * v, label="Left")
        sc.key("L", t, pos).key("L", t + 0.3, pos).key("L", t + 1.5, pos)
        t += 1.6
    for tt, dets in sc.frames():
        pipe.process(dets, tt, sc.aspect)
        st = cal.update(pipe.features, tt)
    assert st.step == Step.DONE, st.message
    assert cfg.dominant_hand == "Left" and cfg.calibrated
    c = cfg.cursor
    assert (c.region_x1 - c.region_x0) == pytest.approx(0.3, abs=0.02)
    assert (c.region_y1 - c.region_y0) == pytest.approx(0.4, abs=0.02)


def _run_calibration(offsets, cfg=None):
    """Point at the four targets with the given hand positions (x, y)."""
    cfg = cfg or Config()
    pipe = GesturePipeline(cfg, DESK)
    cal = Calibrator(cfg)
    sc = Scenario(noise=0.001)
    t = 0.0
    sc.key("R", t, HandPose("point", x=0.5, y=0.5))
    t += 1.6
    for x, y in offsets:
        pos = HandPose("point", x=x, y=y)
        sc.key("R", t, pos).key("R", t + 0.3, pos).key("R", t + 1.5, pos)
        t += 1.6
    for tt, dets in sc.frames():
        pipe.process(dets, tt, sc.aspect)
        st = cal.update(pipe.features, tt)
    return cal, cfg


def test_tiny_hand_travel_is_rejected_then_expanded():
    """Regression: a real first launch accepted a 9%-wide region (barely moving
    the hand), which makes the cursor hypersensitive."""
    tiny = [(0.50 + 0.1 * u, 0.40 + 0.3 * v) for u, v in TARGET_FRACS]   # ~0.1-wide region
    cal, cfg = _run_calibration(tiny)
    st = cal.state
    assert st.step != Step.DONE and st.attempts == 1 and "further" in st.message
    cal2, cfg2 = _run_calibration(tiny + tiny)          # second small attempt
    assert cal2.state.step == Step.DONE and cal2.state.expanded
    c = cfg2.cursor
    assert c.region_x1 - c.region_x0 >= 0.22 - 1e-9


def test_next_target_needs_hand_travel():
    """A hand parked in one spot must not satisfy all four targets."""
    parked = [(0.5, 0.5)] * 4
    cal, _ = _run_calibration(parked)
    assert cal.state.step == Step.TARGETS and cal.state.target == 1


def test_mirrored_samples_rejected():
    backwards = [(0.7 - 0.4 * u, 0.2 + 0.4 * v) for u, v in TARGET_FRACS]
    with pytest.raises(ValueError):
        region_from_samples(backwards)
