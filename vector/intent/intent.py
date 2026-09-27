"""Intent engine: gesture events + snapshot -> desktop commands + feedback.

This is where gestures acquire *meaning* in context: what window is under
the cursor, whether the foreground app is a browser, whether draw mode is on,
which monitor a throw is aimed at. It also applies the last line of
accidental-action defence: per-action confidence thresholds and cooldowns,
and target revalidation right before a command is issued.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from vector.config import Config
from vector.core.geometry import Rect
from vector.desktop.commands import Command, Motion
from vector.desktop.monitors import VirtualDesktop
from vector.gestures.events import GestureEvent, HandMode, Snapshot, SystemState
from vector.intent import throw as throw_mod
from vector.intent.actions import action_for, is_valid_action, threshold_class_for
from vector.intent.resize import resize_rect

log = logging.getLogger(__name__)

BROWSERS = {"chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe",
            "vivaldi.exe", "arc.exe", "explorer.exe"}
COOLDOWN_CLASS = {"app_switch": "swipe_s", "media": "media_s", "throw": "throw_s",
                  "draw_toggle": "draw_toggle_s"}


@dataclass
class HistoryItem:
    t: float
    gesture: str
    action: str
    confidence: float
    accepted: bool
    reason: str = ""


@dataclass
class Feedback:
    """What the overlay should show, beyond the raw snapshot."""
    hover_frame: Rect | None = None
    hover_title: str = ""
    lock_frame: Rect | None = None
    lock_state: str = ""                # pinch | grab | drag | resize | ""
    lock_title: str = ""
    throw_trail: tuple | None = None    # (t, start_xy, end_xy, label)
    toast: tuple | None = None          # (t, text)
    draw_ops: list = field(default_factory=list)
    history: list[HistoryItem] = field(default_factory=list)


@dataclass
class _Lock:
    interaction: int
    hwnd: int | None
    frame: Rect | None
    title: str = ""
    state: str = "pinch"
    resize_start: Rect | None = None
    resize_p0: tuple | None = None


class IntentEngine:
    def __init__(self, cfg: Config, desktop: VirtualDesktop, backend,
                 submit: Callable[[Command], None], generation: Callable[[], int],
                 on_toggle_draw: Callable[[float], None] | None = None,
                 clock: Callable[[], float] = time.perf_counter):
        self.cfg, self.desktop, self.backend = cfg, desktop, backend
        self.submit, self.generation = submit, generation
        self.on_toggle_draw = on_toggle_draw
        self.clock = clock
        self.fb = Feedback()
        self.history: deque[HistoryItem] = deque(maxlen=40)
        self._lock: _Lock | None = None
        self._last_fire: dict[str, float] = {}
        self._hover_t = -1.0
        self._switch_session: tuple[float, list[int], int] | None = None
        self.last_motion: Motion | None = None

    # ------------------------------------------------------------ helpers
    def _cmd(self, kind: str, ev_t: float, interaction: int = 0, ttl: float = 0.35, **args) -> None:
        self.submit(Command(kind, self.generation(), ev_t, ttl, interaction, args))

    def _record(self, ev: GestureEvent, action: str, ok: bool, reason: str = "") -> None:
        item = HistoryItem(ev.t, _gesture_name(ev), action, ev.confidence.value, ok, reason)
        self.history.append(item)
        if ok:
            self.fb.toast = (ev.t, _pretty(action, ev))

    def _gate(self, ev: GestureEvent, action: str) -> bool:
        """Confidence threshold + cooldown for a discrete action."""
        cls = threshold_class_for(ev, self.cfg)
        need = self.cfg.threshold(cls)
        if ev.confidence.value < need:
            self._record(ev, action, False, f"confidence {ev.confidence.value:.2f} < {need:.2f}"
                         f" (weakest: {ev.confidence.weakest()})")
            return False
        cd_attr = COOLDOWN_CLASS.get(cls)
        if cd_attr:
            cd = getattr(self.cfg.cooldown, cd_attr)
            last = self._last_fire.get(cls, -1e9)
            if ev.t - last < cd:
                self._record(ev, action, False, f"cooldown {cls}")
                return False
            self._last_fire[cls] = ev.t
        return True

    # --------------------------------------------------------------- main
    def handle(self, events: list[GestureEvent], snap: Snapshot) -> Feedback:
        self.fb.draw_ops = []
        for ev in events:
            try:
                self._handle_event(ev, snap)
            except Exception:  # never let one bad event kill the pipeline
                log.exception("intent failed for %r", ev)
        self._continuous(snap)
        self.fb.history = list(self.history)
        return self.fb

    def _handle_event(self, ev: GestureEvent, snap: Snapshot) -> None:
        k = ev.kind
        if k in ("wake", "sleep"):
            self._record(ev, k, True)
            if k == "sleep":
                self._release_lock(ev.t)
            return
        if k == "pinch_start":
            hwnd = self.backend.window_at(*ev.data["point"])
            inf = self.backend.info(hwnd) if hwnd else None
            self._lock = _Lock(ev.interaction, hwnd, inf.frame if inf else None,
                               inf.title if inf else "", "pinch")
            return
        if k == "grab":
            if self._lock:
                self._lock.state = "grab"
            return
        if k == "click":
            if self._gate(ev, "click"):
                self._cmd("click", ev.t, ev.interaction, point=ev.data["point"], button="left")
                self._record(ev, "click", True)
            self._lock = None
            return
        if k == "right_click":
            if self._gate(ev, "right_click"):
                self._cmd("click", ev.t, ev.interaction, point=ev.data["point"], button="right")
                self._record(ev, "right_click", True)
            return
        if k == "release":
            self._lock = None
            return
        if k == "drag_start":
            self._drag_start(ev)
            return
        if k == "drag_end":
            self._cmd("drag_end", ev.t, ev.interaction, ttl=5.0)
            self._lock = None
            return
        if k == "throw":
            self._throw(ev)
            return
        if k == "resize_start":
            self._resize_start(ev)
            return
        if k == "resize_end":
            self._cmd("resize_end", ev.t, ev.interaction, ttl=5.0)
            if self._lock:
                self._lock.state = "grab"
            return
        if k == "cancel":
            self._release_lock(ev.t)
            self._record(ev, "cancel", True, ev.data.get("reason", ""))
            return
        if k in ("stroke_begin", "stroke_point", "stroke_end", "erase_point"):
            self.fb.draw_ops.append((k, ev.data.get("point")))
            return
        if k in ("swipe", "hold", "volume_step"):
            self._bound_action(ev, snap)

    # ------------------------------------------------------- window drags
    def _drag_start(self, ev: GestureEvent) -> None:
        lock = self._lock
        if lock is None or lock.interaction != ev.interaction and not ev.data.get("rebase"):
            return
        hwnd = lock.hwnd
        if hwnd is None or not self.backend.is_valid_target(hwnd):
            self._record(ev, "drag", False, "no movable window under cursor")
            return
        if ev.confidence.value < self.cfg.threshold("grab"):
            self._record(ev, "drag", False, "low confidence")
            return
        lock.state = "drag"
        lock.interaction = ev.interaction
        self._cmd("drag_begin", ev.t, ev.interaction, ttl=1.0, hwnd=hwnd,
                  anchor=np.asarray(ev.data["anchor"]), point=np.asarray(ev.data["point"]))
        if not ev.data.get("rebase"):
            self._record(ev, "grab window", True)

    def _throw(self, ev: GestureEvent) -> None:
        lock = self._lock
        self._cmd("drag_end", ev.t, ev.interaction, ttl=5.0)
        self._lock = None
        if lock is None or lock.hwnd is None or not self.backend.is_valid_target(lock.hwnd):
            return
        if not self._gate(ev, "throw"):
            return
        inf = self.backend.info(lock.hwnd)
        if inf is None:
            return
        m = self.cfg.monitors
        dec = throw_mod.decide(inf.frame, inf.maximized, ev.data["release_point"],
                               ev.data["velocity_px"], self.desktop,
                               self.cfg.gesture.throw_project_s, m.throw_to_monitors, m.snap_on_throw)
        if dec.action == "none":
            return
        h = lock.hwnd
        if dec.action in ("snap_left", "snap_right", "monitor"):
            self._cmd("set_frame", ev.t, ev.interaction, ttl=1.0, hwnd=h, rect=dec.rect,
                      maximize_after=dec.maximize_after)
        else:
            self._cmd(dec.action, ev.t, ev.interaction, ttl=1.0, hwnd=h)
        start = tuple(ev.data["release_point"])
        end = dec.projected or start
        self.fb.throw_trail = (ev.t, start, end, dec.action.replace("_", " "))
        self._record(ev, f"throw -> {dec.action}", True)

    def _resize_start(self, ev: GestureEvent) -> None:
        lock = self._lock
        if lock is None or lock.hwnd is None or not self.backend.is_valid_target(lock.hwnd):
            return
        inf = self.backend.info(lock.hwnd)
        if inf is None or not inf.resizable:
            self._record(ev, "resize", False, "window is not resizable")
            return
        self._cmd("drag_end", ev.t, ev.interaction, ttl=5.0)
        self._cmd("resize_begin", ev.t, ev.interaction, ttl=1.0, hwnd=lock.hwnd)
        lock.state, lock.resize_start = "resize", inf.frame
        lock.resize_p0 = tuple(np.asarray(p) for p in ev.data["points"])
        self._record(ev, "resize", True)

    def _release_lock(self, t: float) -> None:
        if self._lock is not None:
            self._cmd("drag_end", t, self._lock.interaction, ttl=5.0)
            self._cmd("resize_end", t, self._lock.interaction, ttl=5.0)
        self._lock = None

    # ---------------------------------------------------- bound actions
    def _bound_action(self, ev: GestureEvent, snap: Snapshot) -> None:
        action = self._draw_mode_override(ev) if snap.draw_mode else None
        action = action or action_for(ev, self.cfg)
        if not action or action == "none":
            self._record(ev, "unbound", False, "no binding")
            return
        if not is_valid_action(action):
            self._record(ev, action, False, "unknown action in config")
            return
        if ev.kind == "volume_step":
            # Steps are rate-limited by hand travel, not by the media cooldown.
            if ev.confidence.value >= self.cfg.threshold("media"):
                self._cmd("keys", ev.t, chord="volume_up" if ev.data["direction"] > 0 else "volume_down")
                self._record(ev, "volume +" if ev.data["direction"] > 0 else "volume -", True)
            return
        if not self._gate(ev, action):
            return
        self._execute(action, ev)

    def _draw_mode_override(self, ev: GestureEvent) -> str | None:
        if ev.kind == "swipe" and ev.data["family"] == "two_finger":
            return "draw_undo" if ev.data["direction"] == "left" else "draw_color"
        if ev.kind == "hold" and ev.data["pose"] == "fist":
            return "draw_clear"
        return None

    def _execute(self, action: str, ev: GestureEvent) -> None:
        t = ev.t
        if action.startswith("draw_"):
            self.fb.draw_ops.append((action, None))
        elif action in ("app_next", "app_previous"):
            self._switch_app(+1 if action == "app_next" else -1, t, ev.interaction)
        elif action in ("context_back", "context_forward"):
            fg = self.backend.foreground()
            proc = self.backend.process_name(fg).lower() if fg else ""
            back = action == "context_back"
            if proc in BROWSERS:
                chord = "browser_back" if back else "browser_forward"
            else:
                chord = "media_previous" if back else "media_next"
            self._cmd("keys", t, chord=chord)
            action = f"{action} ({chord})"
        elif action in ("browser_back", "browser_forward", "media_play_pause", "media_next",
                        "media_previous", "volume_up", "volume_down", "volume_mute"):
            self._cmd("keys", t, chord=action)
        elif action == "toggle_draw":
            if self.on_toggle_draw:
                self.on_toggle_draw(t)
        elif action == "right_click":
            pass
        elif action in ("maximize_foreground", "minimize_foreground"):
            fg = self.backend.foreground()
            if fg:
                self._cmd(action.split("_")[0], t, hwnd=fg)
        elif action.startswith("keys:"):
            self._cmd("keys", t, chord=action[5:])
        self._record(ev, action, True)

    def _switch_app(self, step: int, t: float, interaction: int) -> None:
        """Consecutive swipes within a short session walk a *frozen* z-order
        list (like holding Alt and pressing Tab repeatedly); a fresh swipe
        after a pause starts again from the current z-order."""
        sess = self._switch_session
        if sess is None or t - sess[0] > 2.5:
            windows = self.backend.switchable()
            if len(windows) < 2:
                return
            idx = 0
        else:
            _, windows, idx = sess
        windows = [w for w in windows if self.backend.is_valid_target(w)]
        if len(windows) < 2:
            self._switch_session = None
            return
        idx = (idx + (1 if step > 0 else -1)) % len(windows)
        self._switch_session = (t, windows, idx)
        self._cmd("focus", t, interaction, hwnd=windows[idx])

    # -------------------------------------------------------- continuous
    def _continuous(self, snap: Snapshot) -> None:
        lock = self._lock
        resize_rect_ = None
        if lock is not None and lock.state == "resize" and snap.resize_points is not None \
                and lock.resize_start is not None:
            mon = self.desktop.monitor_for_rect(lock.resize_start)
            g = self.cfg.gesture
            resize_rect_ = resize_rect(lock.resize_start, lock.resize_p0, snap.resize_points,
                                       mon.work, g.resize_min_w, g.resize_min_h)
        drives = bool(snap.confidence.get("drives_cursor")) and snap.cursor is not None
        self.last_motion = Motion(
            t_capture=snap.t, cursor=snap.cursor, drive_cursor=drives and self.cfg.cursor.drive_os_cursor,
            drag_point=snap.drag_point if lock is not None and lock.state == "drag" else None,
            resize_rect=resize_rect_,
            scroll_velocity=snap.scroll_velocity if snap.system == SystemState.ACTIVE else 0.0)

        # Hover target (throttled: EnumWindows walk is ~1 ms).
        fb = self.fb
        if lock is not None:
            fb.lock_state, fb.lock_title = lock.state, lock.title
            fb.lock_frame = resize_rect_ or lock.frame
        else:
            fb.lock_state, fb.lock_frame, fb.lock_title = "", None, ""
        pointing = snap.primary_mode == HandMode.POINTING and snap.cursor_visible
        if pointing and snap.cursor is not None and snap.t - self._hover_t > 0.08:
            self._hover_t = snap.t
            hwnd = self.backend.window_at(*snap.cursor)
            inf = self.backend.info(hwnd) if hwnd else None
            fb.hover_frame = inf.frame if inf and not inf.maximized else (inf.frame if inf else None)
            fb.hover_title = inf.title if inf else ""
        elif not pointing and lock is None:
            fb.hover_frame, fb.hover_title = None, ""


def _gesture_name(ev: GestureEvent) -> str:
    if ev.kind == "swipe":
        return f"{ev.data['family']} swipe {ev.data['direction']}"
    if ev.kind == "hold":
        return f"{ev.data['pose']} hold"
    return ev.kind.replace("_", " ")


def _pretty(action: str, ev: GestureEvent) -> str:
    names = {
        "app_next": "Next app", "app_previous": "Previous app",
        "media_play_pause": "Play / Pause", "media_next": "Next track",
        "media_previous": "Previous track", "toggle_draw": "Draw mode",
        "click": "Click", "right_click": "Right click", "wake": "Active", "sleep": "Sleeping",
        "draw_undo": "Undo", "draw_color": "Color", "draw_clear": "Clear",
        "grab window": "Grab", "resize": "Resize",
    }
    if action.startswith("throw -> "):
        return {"snap_left": "Snap left", "snap_right": "Snap right", "maximize": "Maximize",
                "minimize": "Minimize", "restore": "Restore", "monitor": "Move to display"
                }.get(action[9:], action)
    if action.startswith("context_"):
        inner = action.split("(")[-1].rstrip(")")
        return {"browser_back": "Back", "browser_forward": "Forward",
                "media_previous": "Previous track", "media_next": "Next track"}.get(inner, inner)
    if action.startswith("keys:"):
        return action[5:].upper()
    return names.get(action, action.replace("_", " ").capitalize())
