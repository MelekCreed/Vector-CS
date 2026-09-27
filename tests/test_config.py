import json

import pytest

from vector import config as C


def test_defaults_validate():
    cfg = C.Config()
    C.validate(cfg)
    assert cfg.threshold("app_switch") == 0.80
    assert cfg.threshold("something_new") == cfg.confidence.default_threshold


def test_partial_override_deep_merges():
    cfg, unknown = C.from_dict({
        "cursor": {"sensitivity": 1.4},
        "confidence": {"thresholds": {"media": 0.9}},
        "bindings": {"fist_hold": "keys:ctrl+shift+m"},
    })
    assert unknown == []
    assert cfg.cursor.sensitivity == 1.4
    assert cfg.cursor.spring_hz == C.CursorConfig().spring_hz        # untouched default kept
    assert cfg.confidence.thresholds["media"] == 0.9
    assert cfg.confidence.thresholds["app_switch"] == 0.80           # merged, not replaced
    assert cfg.bindings["fist_hold"] == "keys:ctrl+shift+m"
    assert cfg.bindings["palm_swipe_left"] == "app_previous"


def test_int_is_accepted_for_float_but_bool_is_not():
    cfg, _ = C.from_dict({"cursor": {"sensitivity": 2}})
    assert cfg.cursor.sensitivity == 2.0 and isinstance(cfg.cursor.sensitivity, float)
    with pytest.raises(C.ConfigError):
        C.from_dict({"cursor": {"sensitivity": True}})


def test_unknown_keys_reported():
    _, unknown = C.from_dict({"cursor": {"sensitivty": 1.2}, "bogus": 1})
    assert "cursor.sensitivty" in unknown and "bogus" in unknown


@pytest.mark.parametrize("bad", [
    {"cursor": {"region_x0": 0.9, "region_x1": 0.1}},
    {"gesture": {"pinch_enter": 0.5, "pinch_exit": 0.3}},
    {"dominant_hand": "Middle"},
    {"confidence": {"thresholds": {"media": 1.5}}},
    {"cursor": "fast"},
])
def test_invalid_configs_rejected(bad):
    with pytest.raises(C.ConfigError):
        C.from_dict(bad)


def test_save_writes_only_diff_and_roundtrips(tmp_path):
    cfg = C.Config()
    cfg.cursor.sensitivity = 1.3
    cfg.dominant_hand = "Left"
    path = C.save(cfg, tmp_path / "user_config.json")
    on_disk = json.loads(path.read_text())
    assert on_disk == {"cursor": {"sensitivity": 1.3}, "dominant_hand": "Left"}
    again = C.load(path)
    assert again.cursor.sensitivity == 1.3 and again.dominant_hand == "Left"


def test_missing_file_gives_defaults(tmp_path):
    assert C.load(tmp_path / "nope.json") == C.Config()


def test_corrupt_file_raises(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{not json")
    with pytest.raises(C.ConfigError):
        C.load(p)
