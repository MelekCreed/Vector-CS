import numpy as np
import pytest

from vector.core.filters import (
    CriticalSpring, History, Hysteresis, OneEuroFilter, VelocityEstimator, regression_velocity, Sample,
)


def test_one_euro_suppresses_jitter_at_rest():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.02)
    out = [f(np.array([100.0, 100.0]) + rng.normal(0, 2.0, 2), i / 30) for i in range(120)]
    raw_std = 2.0
    # At 30 Hz the filter alone leaves ~40% of the jitter; the cursor's dead zone
    # removes the rest. Guard the filter's share of the job here.
    assert np.std(np.array(out[30:]), axis=0).max() < raw_std * 0.5


def test_one_euro_tracks_fast_motion_with_low_lag():
    slow = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    adaptive = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    for i in range(60):
        x = 20.0 * i                     # 600 units/s ramp
        a = slow(x, i / 30)
        b = adaptive(x, i / 30)
    assert (x - b) < (x - a) * 0.5       # speed-adaptive cutoff reduces lag


def test_one_euro_ignores_duplicate_timestamps():
    f = OneEuroFilter()
    f(1.0, 0.0)
    assert f(50.0, 0.0) == pytest.approx(1.0)


def test_regression_velocity_exact_for_linear_motion():
    s = [Sample(t, np.array([3.0 * t, -2.0 * t + 1])) for t in np.linspace(0, 0.1, 5)]
    assert regression_velocity(s) == pytest.approx([3.0, -2.0])


def test_velocity_estimator_window_and_noise():
    rng = np.random.default_rng(1)
    v = VelocityEstimator(window_s=0.15)
    for i in range(30):
        t = i / 30
        est = v.update(np.array([5.0 * t, 0.0]) + rng.normal(0, 0.005, 2), t)
    assert est[0] == pytest.approx(5.0, rel=0.1)
    assert abs(est[1]) < 0.5


def test_velocity_estimator_single_sample_is_zero():
    v = VelocityEstimator()
    assert np.allclose(v.update([1, 1], 0.0), 0)


def test_history_at_and_span():
    h = History(span_s=0.5)
    for i in range(10):
        h.push(i * 0.1, i)
    assert len(h) == 6                   # 0.4..0.9 kept (span 0.5)
    assert h.at(0.65) == 6
    assert h.at(-1) == 4                 # before buffer: oldest


def test_critical_spring_converges_without_overshoot():
    s = CriticalSpring(frequency_hz=10)
    s.reset([0.0])
    xs = [s.step([100.0], 1 / 120)[0] for _ in range(120)]
    assert xs[-1] == pytest.approx(100.0, abs=0.01)
    assert max(xs) <= 100.0 + 1e-9
    assert all(b >= a - 1e-9 for a, b in zip(xs, xs[1:]))   # monotonic


def test_critical_spring_is_frame_rate_independent():
    a, b = CriticalSpring(8), CriticalSpring(8)
    a.reset([0.0]); b.reset([0.0])
    for _ in range(60):
        a.step([10.0], 1 / 60)
    for _ in range(240):
        b.step([10.0], 1 / 240)
    assert a.pos[0] == pytest.approx(b.pos[0], rel=1e-6)


def test_hysteresis_low_is_on():
    h = Hysteresis(enter=0.3, exit=0.45, low_is_on=True)
    seq = [0.5, 0.35, 0.29, 0.4, 0.44, 0.46, 0.35]
    assert [h(x) for x in seq] == [False, False, True, True, True, False, False]
