import math

import numpy as np
import pytest

from vector.learned.dataset import FEATURES, canonical_world, clip_sequence, expected_engine_event, load_dataset
from vector.sim.synth_clips import make_clip, make_clips
from vector.sim.synthetic_hand import HandPose, build_world


def test_canonical_world_is_scale_and_roll_invariant():
    a = canonical_world(build_world(HandPose("point")))
    b = canonical_world(build_world(HandPose("point", roll=0.6)) * 1.7)
    assert np.allclose(a, b, atol=1e-6)


def test_clip_sequence_shape_and_time_resampling():
    clip = make_clip("palm_swipe_right", np.random.default_rng(0), 0)
    seq = clip_sequence(clip, T=32)
    assert seq.shape == (32, FEATURES) and np.isfinite(seq).all()
    vx = seq[:, 63]                                     # palm velocity x (hand-lengths/s)
    assert vx.max() > 1.5                               # the swipe is visible in the features


def test_dataset_groups_sessions():
    ds = load_dataset(clips=make_clips(per_label=4, sessions=2, seed=1))
    assert ds.X.shape[0] == 24 and len(ds.labels) == 6
    assert len(np.unique(ds.sessions)) == 2


def test_label_to_engine_event_mapping():
    assert expected_engine_event("palm_swipe_left") == ("swipe", {"family": "palm", "direction": "left"})
    assert expected_engine_event("two_finger_swipe_right")[1]["family"] == "two_finger"
    assert expected_engine_event("fist_hold") == ("hold", {"pose": "fist"})
    assert expected_engine_event("circle") is None


def test_models_train_and_predict_smoke():
    pytest.importorskip("torch")
    from vector.learned.model import predict, train
    ds = load_dataset(clips=make_clips(per_label=6, sessions=2, seed=2))
    for kind in ("gru", "tcn"):
        m = train(kind, ds.X, ds.y, len(ds.labels), epochs=15)
        p = predict(m, ds.X[:5])
        assert p.shape == (5, len(ds.labels)) and np.allclose(p.sum(1), 1, atol=1e-5)
