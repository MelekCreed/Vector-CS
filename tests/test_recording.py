from vector.config import Config
from vector.pipeline import GesturePipeline
from vector.recording import Recorder, list_clips, load_clip, replay
from vector.sim.scenario import Scenario
from vector.sim.synthetic_hand import HandPose
from tests.test_engine import DESK


def _swipe_scenario():
    sc = Scenario()
    sc.key("R", 0, HandPose("open", x=0.3)).key("R", 0.5, HandPose("open", x=0.3))
    sc.key("R", 0.72, HandPose("open", x=0.7)).key("R", 1.2, HandPose("open", x=0.7))
    return sc


def test_record_save_load_replay_roundtrip(tmp_path):
    sc = _swipe_scenario()
    pipe = GesturePipeline(Config(), DESK)
    rec = Recorder(tmp_path)
    rec.start("palm swipe right!", sc.aspect, {"note": "test"})
    for t, dets in sc.frames(t0=100.0):
        pipe.process(dets, t, sc.aspect)
        rec.add(t, dets, pipe.features)
    path = rec.stop()
    assert path.parent.name == "palm_swipe_right_"
    assert list_clips(tmp_path)["palm_swipe_right_"] == [path]

    clip = load_clip(path)
    assert clip.label == "palm swipe right!" and clip.frames[0]["t"] == 0.0
    assert len(clip.frames) == len(list(sc.frames()))
    f0 = clip.frames[5]["features"][0]
    assert set(f0) >= {"velocity", "palm_normal", "pinch_ratio", "extension"}

    events = replay(clip, Config(), DESK)
    swipes = [e for e in events if e.kind == "swipe"]
    assert [(e.data["family"], e.data["direction"]) for e in swipes] == [("palm", "right")]


def test_stop_without_frames_saves_nothing(tmp_path):
    rec = Recorder(tmp_path)
    rec.start("x", 4 / 3)
    assert rec.stop() is None
