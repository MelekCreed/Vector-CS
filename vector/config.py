"""Typed configuration with JSON overrides.

Defaults live in the dataclasses below (single source of truth). A user file
(``%APPDATA%/Vector/user_config.json`` by default) stores only the values that
differ, and is deep-merged on top at load time. Unknown keys are reported, not
silently ignored, so typos in hand-edited configs surface immediately.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

log = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    index: int = 0
    backend: str = "msmf"          # msmf | dshow | any
    width: int = 640
    height: int = 480
    fps: int = 30
    mirror: bool = True            # selfie view: moving right moves the cursor right


@dataclass
class TrackerConfig:
    model_path: str = "models/hand_landmarker.task"
    max_hands: int = 2
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    lost_grace_s: float = 0.25     # keep a hand "alive" through brief dropouts


@dataclass
class CursorConfig:
    # Region of the (mirrored, normalised) camera image mapped to the desktop.
    region_x0: float = 0.22
    region_y0: float = 0.18
    region_x1: float = 0.78
    region_y1: float = 0.68
    mode: str = "hybrid"           # hybrid | absolute
    sensitivity: float = 1.0
    # One-Euro filter (jitter at rest vs lag while moving).
    min_cutoff: float = 1.2
    beta: float = 0.02
    d_cutoff: float = 1.0
    # Speed-dependent gain (hybrid mode). Speeds in screen-heights / s.
    gain_slow: float = 0.55
    gain_fast: float = 1.5
    speed_slow: float = 0.15
    speed_fast: float = 1.6
    drift_correction: float = 2.5  # 1/s pull toward absolute mapping while moving
    dead_zone_px: float = 3.0
    # Actuator spring (critically damped); higher = snappier, lower = silkier.
    spring_hz: float = 14.0
    drive_os_cursor: bool = True
    pinch_rewind_s: float = 0.10   # target lookup uses cursor from before pinch onset


@dataclass
class GestureConfig:
    # Pose hysteresis on pinch ratio (thumb-index distance / palm size).
    pinch_enter: float = 0.28
    pinch_exit: float = 0.42
    pose_smoothing: float = 0.45   # EMA alpha for pose scores
    pose_enter: float = 0.62
    pose_exit: float = 0.42
    pose_confirm_frames: int = 3
    click_max_s: float = 0.28
    click_max_move: float = 0.035  # hand-lengths of palm travel
    grab_hold_s: float = 0.30
    drag_start_move: float = 0.05
    throw_min_speed: float = 2.4   # hand-lengths / s at release
    throw_window_s: float = 0.12   # velocity averaged over this window before release
    throw_project_s: float = 0.28
    throw_down_speed_factor: float = 1.3   # minimize needs a more deliberate throw
    swipe_min_speed: float = 2.2
    swipe_min_distance: float = 0.9
    swipe_max_duration_s: float = 0.6
    swipe_direction_ratio: float = 1.8  # |dx| must beat |dy| by this factor
    scroll_gain: float = 900.0     # wheel units per hand-length
    scroll_inertia_decay: float = 4.5
    scroll_dead_zone: float = 0.25 # hand-lengths / s
    hold_still_speed: float = 0.6  # below this a hand counts as "still"
    fist_hold_s: float = 0.45
    three_hold_s: float = 0.7
    volume_step: float = 0.35      # hand-lengths per volume step
    carousel_enabled: bool = True
    carousel_hold_s: float = 0.9   # still open palm this long opens the app carousel
    carousel_step: float = 0.45    # hand-lengths of sideways travel per app
    resize_min_w: int = 360
    resize_min_h: int = 240


@dataclass
class ConfidenceConfig:
    default_threshold: float = 0.70
    thresholds: dict[str, float] = field(default_factory=lambda: {
        "app_switch": 0.80,
        "media": 0.80,
        "throw": 0.72,
        "click": 0.70,
        "grab": 0.65,
        "activate": 0.85,
        "sleep": 0.85,
        "draw_toggle": 0.80,
    })


@dataclass
class CooldownConfig:
    swipe_s: float = 0.7
    media_s: float = 1.0
    throw_s: float = 0.4
    activate_s: float = 1.5
    draw_toggle_s: float = 1.2


@dataclass
class ActivationConfig:
    require_activation: bool = True
    start_active: bool = False
    wake_hold_s: float = 0.6
    sleep_hold_s: float = 0.8
    auto_sleep_s: float = 0.0      # 0 = never auto-sleep when hands leave


@dataclass
class SafetyConfig:
    failsafe_double_tap_key: str = "esc"
    failsafe_window_s: float = 0.45
    rearm_hotkey: str = "ctrl+alt+v"
    blocked_classes: list[str] = field(default_factory=lambda: [
        "Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd",
        "Windows.UI.Core.CoreWindow", "XamlExplorerHostIslandWindow",
        "TopLevelWindowForOverflowXamlIsland", "NotifyIconOverflowWindow",
    ])
    blocked_processes: list[str] = field(default_factory=lambda: [
        "LockApp.exe", "ShellExperienceHost.exe", "SearchHost.exe",
        "StartMenuExperienceHost.exe",
    ])


@dataclass
class DrawConfig:
    colors: list[str] = field(default_factory=lambda: [
        "#6EE7F9", "#F472B6", "#FDE68A", "#A7F3D0", "#FFFFFF",
    ])
    thickness: float = 5.0
    eraser_radius: float = 42.0
    pen_down_pose: str = "pinch"   # pinch | point
    min_point_spacing_px: float = 2.5
    export_dir: str = "~/Pictures/Vector"


@dataclass
class OverlayConfig:
    enabled: bool = True
    fps: int = 60
    accent: str = "#6EE7F9"
    warn: str = "#F59E0B"
    danger: str = "#F43F5E"
    show_skeleton: bool = True
    show_hints: bool = True
    skeleton_opacity: float = 0.55
    hud_corner: str = "top-right"


@dataclass
class MonitorConfig:
    throw_to_monitors: bool = True
    snap_on_throw: bool = True


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    cursor: CursorConfig = field(default_factory=CursorConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    cooldown: CooldownConfig = field(default_factory=CooldownConfig)
    activation: ActivationConfig = field(default_factory=ActivationConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    draw: DrawConfig = field(default_factory=DrawConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    monitors: MonitorConfig = field(default_factory=MonitorConfig)
    dominant_hand: str = "Right"
    debug: bool = False
    calibrated: bool = False
    # gesture -> action. Actions: see vector.intent.actions. "keys:ctrl+shift+t" sends a chord.
    bindings: dict[str, str] = field(default_factory=lambda: {
        "palm_swipe_left": "app_previous",
        "palm_swipe_right": "app_next",
        "two_finger_swipe_left": "context_back",
        "two_finger_swipe_right": "context_forward",
        "three_finger_swipe_left": "media_previous",
        "three_finger_swipe_right": "media_next",
        "fist_hold": "media_play_pause",
        "shaka_vertical": "volume",
        "three_finger_hold": "toggle_draw",
        "middle_pinch": "right_click",
    })

    def threshold(self, action_class: str) -> float:
        return self.confidence.thresholds.get(action_class, self.confidence.default_threshold)


class ConfigError(ValueError):
    pass


def _coerce(value: Any, typ: Any, path: str) -> Any:
    origin = getattr(typ, "__origin__", None)
    if dataclasses.is_dataclass(typ):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected object, got {type(value).__name__}")
        return value
    if typ is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected number, got {value!r}")
        return float(value)
    if typ is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected integer, got {value!r}")
        return value
    if typ is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true/false, got {value!r}")
        return value
    if typ is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected string, got {value!r}")
        return value
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected list, got {value!r}")
        return list(value)
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected object, got {value!r}")
        return dict(value)
    return value


def _apply(obj: Any, data: dict[str, Any], path: str, unknown: list[str]) -> None:
    hints = get_type_hints(type(obj))
    for key, value in data.items():
        if key not in hints:
            unknown.append(f"{path}{key}")
            continue
        typ = hints[key]
        value = _coerce(value, typ, f"{path}{key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            _apply(current, value, f"{path}{key}.", unknown)
        elif isinstance(current, dict) and key in ("thresholds", "bindings"):
            merged = dict(current)
            merged.update(value)       # partial override keeps untouched defaults
            setattr(obj, key, merged)
        else:
            setattr(obj, key, value)


def from_dict(data: dict[str, Any]) -> tuple[Config, list[str]]:
    cfg = Config()
    unknown: list[str] = []
    _apply(cfg, data, "", unknown)
    validate(cfg)
    return cfg, unknown


def validate(cfg: Config) -> None:
    c = cfg.cursor
    if not (0 <= c.region_x0 < c.region_x1 <= 1 and 0 <= c.region_y0 < c.region_y1 <= 1):
        raise ConfigError("cursor region must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    if c.mode not in ("hybrid", "absolute"):
        raise ConfigError(f"cursor.mode must be hybrid|absolute, got {c.mode!r}")
    g = cfg.gesture
    if g.pinch_enter >= g.pinch_exit:
        raise ConfigError("gesture.pinch_enter must be below gesture.pinch_exit (hysteresis)")
    if g.pose_exit >= g.pose_enter:
        raise ConfigError("gesture.pose_exit must be below gesture.pose_enter (hysteresis)")
    if cfg.dominant_hand not in ("Left", "Right"):
        raise ConfigError("dominant_hand must be Left or Right")
    if cfg.draw.pen_down_pose not in ("pinch", "point"):
        raise ConfigError("draw.pen_down_pose must be pinch|point")
    for name, value in cfg.confidence.thresholds.items():
        if not 0.0 <= value <= 1.0:
            raise ConfigError(f"confidence.thresholds.{name} must be within [0, 1]")


def to_dict(cfg: Config) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def diff_from_defaults(cfg: Config) -> dict[str, Any]:
    """Only the values that differ from defaults — what gets written to disk."""

    def _diff(a: Any, b: Any) -> Any:
        if isinstance(a, dict) and isinstance(b, dict):
            out = {}
            for k, v in a.items():
                if k not in b:
                    out[k] = v
                    continue
                d = _diff(v, b[k])
                if d is not _SAME:
                    out[k] = d
            return out if out else _SAME
        return _SAME if a == b else a

    result = _diff(to_dict(cfg), to_dict(Config()))
    return {} if result is _SAME else result


_SAME = object()


def default_user_path() -> Path:
    base = os.environ.get("VECTOR_CONFIG_DIR") or os.path.join(
        os.environ.get("APPDATA", str(Path.home())), "Vector")
    return Path(base) / "user_config.json"


def load(path: Path | None = None) -> Config:
    path = path or default_user_path()
    if not path.exists():
        return Config()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON ({exc})") from exc
    cfg, unknown = from_dict(data)
    for key in unknown:
        log.warning("config: unknown key %r in %s (ignored)", key, path)
    return cfg


def save(cfg: Config, path: Path | None = None) -> Path:
    path = path or default_user_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(diff_from_defaults(cfg), indent=2), encoding="utf-8")
    os.replace(tmp, path)          # atomic: a crash never leaves half a config
    return path
