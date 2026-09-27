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
