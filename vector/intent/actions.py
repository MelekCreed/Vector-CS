"""Gesture -> action binding resolution and confidence-threshold classes."""

from __future__ import annotations

from vector.config import Config
from vector.gestures.events import GestureEvent

ACTIONS = {
    "app_next", "app_previous", "context_back", "context_forward", "browser_back",
    "browser_forward", "media_play_pause", "media_next", "media_previous", "volume",
    "volume_up", "volume_down", "volume_mute", "toggle_draw", "right_click", "none",
    "maximize_foreground", "minimize_foreground", "draw_undo", "draw_color", "draw_clear",
}

_CLASS_BY_ACTION = {
    "app_next": "app_switch", "app_previous": "app_switch",
    "context_back": "app_switch", "context_forward": "app_switch",
    "browser_back": "app_switch", "browser_forward": "app_switch",
    "media_play_pause": "media", "media_next": "media", "media_previous": "media",
    "volume": "media", "volume_up": "media", "volume_down": "media", "volume_mute": "media",
    "toggle_draw": "draw_toggle", "right_click": "click",
}

_CLASS_BY_EVENT = {
    "click": "click", "right_click": "click", "pinch_start": "grab", "drag_start": "grab",
    "grab": "grab", "resize_start": "grab", "throw": "throw", "wake": "activate",
    "sleep": "sleep", "volume_step": "media",
}


def binding_key(ev: GestureEvent) -> str | None:
    if ev.kind == "swipe":
        return f"{ev.data['family']}_swipe_{ev.data['direction']}"
    if ev.kind == "hold":
        return f"{ev.data['pose']}_hold"
    if ev.kind == "volume_step":
        return "shaka_vertical"
    if ev.kind == "right_click":
        return "middle_pinch"
    return None


def action_for(ev: GestureEvent, cfg: Config) -> str | None:
    key = binding_key(ev)
    return cfg.bindings.get(key) if key else None


def is_valid_action(action: str) -> bool:
    return action in ACTIONS or action.startswith("keys:")


def threshold_class_for(ev: GestureEvent, cfg: Config) -> str:
    action = action_for(ev, cfg)
    if action:
        return _CLASS_BY_ACTION.get(action, "shortcut")
    return _CLASS_BY_EVENT.get(ev.kind, "default")
