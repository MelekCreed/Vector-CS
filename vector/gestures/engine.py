"""Temporal gesture engine.

Consumes per-frame HandFeatures, maintains per-hand state machines plus one
*global* interaction owner, and emits discrete GestureEvents and a continuous
Snapshot. Nothing here touches the OS: the intent layer decides what events
mean on the desktop, and the actuator executes them.

Accidental-action defences, all layered:
* pose hysteresis + time-based confirmation (poses.py)
* SLEEPING / ACTIVE / DISABLED gating, wake needs a held, still, upright palm
* after waking, a *neutral* pose is required before any command (the wake
  palm can't immediately become a swipe)
* one interaction owner at a time: while dragging, no swipe/hold/sleep fires
* clicks and throws only on an *observed* release; tracking loss cancels
* swipes: distance + peak speed + direction + cooldown + return-stroke lockout
* holds: pose + stillness + duration, fire once per entry
* every event carries a confidence; the intent layer applies thresholds
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from vector.config import Config
from vector.core.filters import EMA, History, VelocityEstimator
from vector.cursor.mapper import CursorMapper
from vector.gestures.confidence import Confidence, combine, duration_factor, velocity_margin
from vector.gestures.events import GestureEvent, HandMode, HandView, Snapshot, SystemState
from vector.core.filters import regression_velocity
from vector.gestures.poses import PinchLatch, Pose, PoseState, PoseTracker, pinch_closeness, pose_scores
from vector.gestures.swipe import SwipeDetector
from vector.vision.features import HandFeatures

SWIPE_POSES = {Pose.OPEN_PALM: "palm", Pose.TWO: "two_finger", Pose.THREE: "three_finger"}
HOLD_POSES = {Pose.FIST: "fist", Pose.THREE: "three_finger"}
PINCH_MODES = (HandMode.PINCH_STARTED, HandMode.GRABBING, HandMode.DRAGGING)


@dataclass
class HandCtx:
    track_id: int
    f: HandFeatures
    poses: PoseTracker
    pinch: PinchLatch
    vel: VelocityEstimator = field(default_factory=lambda: VelocityEstimator(0.1))
    scale: EMA = field(default_factory=lambda: EMA(0.15))
    pos_hl: History = field(default_factory=lambda: History(span_s=1.0))   # palm, hand-lengths
    swipe: SwipeDetector | None = None
    swipe_family: str | None = None
    pose: PoseState | None = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    speed: float = 0.0
    pinch_on: bool = False
    pinch_rose: bool = False
    pinch_fell: bool = False
    hold_start: float | None = None
    hold_pose: Pose | None = None
    hold_fired: bool = False
    pose_hist: list = field(default_factory=list)


@dataclass
class _Interaction:
    kind: str                  # "pinch" | "drag" | "resize" | "scroll" | "volume" | "draw" | "erase"
    id: int
    hand: int
    t0: float
    anchor_px: np.ndarray | None = None       # desktop px where the interaction locked on
    palm0: np.ndarray | None = None           # palm (hand-lengths) at start, for movement
    mapper0: np.ndarray | None = None         # mapper output at start (for drag deltas)
    other: int | None = None                  # second hand (resize)
    drag_t0: float | None = None
    vol_anchor: float = 0.0
    extra: dict = field(default_factory=dict)


class GestureEngine:
    def __init__(self, cfg: Config, mapper: CursorMapper):
        self.cfg = cfg
        self.mapper = mapper
        g = cfg.gesture
        a = cfg.activation
        self.system = (SystemState.ACTIVE if (a.start_active or not a.require_activation)
                       else SystemState.SLEEPING)
        self.draw_mode = False
        self.hands: dict[int, HandCtx] = {}
        self.inter: _Interaction | None = None
        self._next_inter = 1
        self.neutral_required = False
        self._neutral_since: float | None = None
        self._wake_start: dict[int, float] = {}
        self._sleep_start: float | None = None
        self._last_event_t: dict[str, float] = {}
        self._scroll_v = 0.0
        self._scroll_inertia = False
        self._drag_hist: History[np.ndarray] = History(span_s=0.6)
        self._last_seen_hand_t = 0.0
        self._cursor_lock: np.ndarray | None = None
        self.primary_id: int | None = None
        self._g = g

    # ------------------------------------------------------------------ utils
    def _new_ctx(self, f: HandFeatures) -> HandCtx:
        g = self._g
        ctx = HandCtx(
            track_id=f.track_id, f=f,
            poses=PoseTracker(alpha=g.pose_smoothing, enter=g.pose_enter, exit=g.pose_exit,
                              confirm_s=max(0.03, g.pose_confirm_frames / 30.0)),
            pinch=PinchLatch(g.pinch_enter, g.pinch_exit))
        return ctx

    def _swipe_detector(self) -> SwipeDetector:
        g, c = self._g, self.cfg.cooldown
        return SwipeDetector(g.swipe_min_speed, g.swipe_min_distance, g.swipe_max_duration_s,
                             g.swipe_direction_ratio, cooldown=c.swipe_s)

    def _tracking(self, ctx: HandCtx) -> float:
        f = ctx.f
        age = min(1.0, 0.4 + f.age_s / 0.5)
        return max(0.05, f.presence) * age * (0.0 if f.stale else 1.0) or 0.05

    def _emit(self, events: list, kind: str, t: float, conf: Confidence, hand: int | None,
              **data) -> GestureEvent:
        ev = GestureEvent(kind, t, conf, hand, self.inter.id if self.inter else 0, data)
        events.append(ev)
        self._last_event_t[kind] = t
        return ev

    def _start(self, kind: str, ctx: HandCtx, t: float, **kw) -> _Interaction:
        self.inter = _Interaction(kind, self._next_inter, ctx.track_id, t, **kw)
        self._next_inter += 1
        return self.inter

    def _palm_hl(self, ctx: HandCtx) -> np.ndarray:
        s = ctx.scale.value or ctx.f.hand_scale
        return ctx.f.palm_center / max(1e-3, s)

    # ------------------------------------------------------------- public API
    def set_system(self, state: SystemState, t: float, events: list | None = None) -> list:
        events = [] if events is None else events
        if state != SystemState.ACTIVE and self.inter is not None:
            self._cancel(events, t, reason=state.value.lower())
        self.system = state
        self._scroll_v = 0.0
        self._scroll_inertia = False
        self._wake_start.clear()
        self._sleep_start = None
        if state == SystemState.ACTIVE:
            self.neutral_required = True
            self._neutral_since = None
        return events

    def toggle_draw(self, t: float) -> None:
        self.draw_mode = not self.draw_mode
        if self.inter and self.inter.kind in ("draw", "erase", "drag", "pinch", "scroll"):
            self._cancel([], t, "mode change")

    def update(self, feats: list[HandFeatures], t: float) -> tuple[list[GestureEvent], Snapshot]:
        events: list[GestureEvent] = []
        self._sync_hands(feats, t, events)
        self._choose_primary(t)

        if self.system == SystemState.SLEEPING:
            self._update_wake(t, events)
        elif self.system == SystemState.ACTIVE:
            if self.cfg.activation.require_activation:
                self._update_sleep(t, events)
            if self.system == SystemState.ACTIVE:
                self._update_neutral(t)
                if not self.neutral_required:
                    self._update_primary(t, events)
                self._update_scroll_inertia(t)
        return events, self._snapshot(t)

    # ---------------------------------------------------------- hand tracking
    def _sync_hands(self, feats: list[HandFeatures], t: float, events: list) -> None:
        seen = set()
        for f in feats:
            seen.add(f.track_id)
            ctx = self.hands.get(f.track_id)
            if ctx is None:
                ctx = self.hands[f.track_id] = self._new_ctx(f)
            ctx.f = f
            if f.stale:
                # Tracking dropout: freeze the hand's state; never transition on
                # stale data (no clicks, no throws, no pose changes).
                ctx.pinch_rose = ctx.pinch_fell = False
                continue
            s = ctx.scale(f.hand_scale)
            pos_hl = f.palm_center / max(1e-3, s)
            ctx.pos_hl.push(t, pos_hl)
            ctx.velocity = ctx.vel.update(f.palm_center, t) / max(1e-3, s)
            ctx.speed = float(np.linalg.norm(ctx.velocity))
            f.velocity, f.speed = ctx.velocity, ctx.speed
            ctx.pose = ctx.poses.update(pose_scores(f), t)
            ctx.pose_hist = (ctx.pose_hist + [ctx.pose.pose])[-8:]
            was = ctx.pinch_on
            # Pinch may not *begin* while another pose is firmly held (e.g. a
            # fist, where the thumb rests near the index finger).
            allow = ctx.pose.pose in (Pose.NONE, Pose.POINT, Pose.OPEN_PALM, Pose.TWO)
            ctx.pinch_on = ctx.pinch.update(f.pinch_ratio, t, allowed=allow)
            ctx.pinch_rose = ctx.pinch_on and not was
            ctx.pinch_fell = was and not ctx.pinch_on
        for tid in list(self.hands):
            if tid not in seen:
                self._hand_lost(tid, t, events)
        if seen:
            self._last_seen_hand_t = t

    def _hand_lost(self, tid: int, t: float, events: list) -> None:
        self.hands.pop(tid, None)
        self._wake_start.pop(tid, None)
        if self.inter and (self.inter.hand == tid or self.inter.other == tid):
            if self.inter.kind == "resize" and self.inter.other == tid and self.inter.hand in self.hands:
                self._end_resize(events, t, keep_drag=True)
            else:
                self._cancel(events, t, reason="hand lost")
        if self.primary_id == tid:
            self.primary_id = None
            self.mapper.reset()

    def _choose_primary(self, t: float) -> None:
        if self.inter is not None and self.inter.hand in self.hands:
            self.primary_id = self.inter.hand          # never switch hands mid-interaction
            return
        dom = [c for c in self.hands.values() if c.f.label == self.cfg.dominant_hand]
        if dom:
            pid = max(dom, key=lambda c: c.f.age_s).track_id
        elif len(self.hands) == 1:
            only = next(iter(self.hands.values()))
            pid = only.track_id if only.f.age_s > 0.3 else None
        else:
            pid = None
        if pid != self.primary_id:
            self.mapper.reset()
        self.primary_id = pid

    @property
    def primary(self) -> HandCtx | None:
        return self.hands.get(self.primary_id) if self.primary_id is not None else None

    # ------------------------------------------------------- wake / sleep
    def _palm_up_still(self, ctx: HandCtx) -> float:
        """0..1 how much this hand looks like a deliberate 'stop' palm."""
        if ctx.f.stale or ctx.pose is None:
            return 0.0
        s = ctx.pose.scores.get(Pose.OPEN_PALM, 0.0)
        facing = np.clip((ctx.f.palm_facing - 0.3) / 0.4, 0, 1)
        upright = np.clip((ctx.f.upright - 0.4) / 0.4, 0, 1)
        still = np.clip((self._g.hold_still_speed * 1.5 - ctx.speed) / self._g.hold_still_speed, 0, 1)
        return float(min(s / max(self._g.pose_enter, 1e-3), 1.0) * facing * upright * still)

    def _update_wake(self, t: float, events: list) -> None:
        a = self.cfg.activation
        for tid, ctx in self.hands.items():
            q = self._palm_up_still(ctx)
            if q > 0.6 and ctx.pose and ctx.pose.pose == Pose.OPEN_PALM:
                start = self._wake_start.setdefault(tid, t)
                if t - start >= a.wake_hold_s:
                    conf = combine({"geometry": q, "duration": 1.0, "tracking": self._tracking(ctx),
                                    "stability": self._stability(ctx, Pose.OPEN_PALM)})
                    if conf.value >= self.cfg.threshold("activate") and self._cool("wake", t, self.cfg.cooldown.activate_s):
                        self.set_system(SystemState.ACTIVE, t)
                        self._emit(events, "wake", t, conf, tid)
                        return
            else:
                self._wake_start.pop(tid, None)

    def _update_sleep(self, t: float, events: list) -> None:
        if self.inter is not None or len(self.hands) < 2:
            self._sleep_start = None
            return
        qs = [self._palm_up_still(c) for c in self.hands.values()]
        poses_ok = all(c.pose and c.pose.pose == Pose.OPEN_PALM for c in self.hands.values())
        if poses_ok and min(qs) > 0.6:
            if self._sleep_start is None:
                self._sleep_start = t
            elif t - self._sleep_start >= self.cfg.activation.sleep_hold_s:
                conf = combine({"geometry": min(qs), "duration": 1.0,
                                "tracking": min(self._tracking(c) for c in self.hands.values())})
                if conf.value >= self.cfg.threshold("sleep") and self._cool("sleep", t, self.cfg.cooldown.activate_s):
                    self.set_system(SystemState.SLEEPING, t)
                    self._emit(events, "sleep", t, conf, None)
        else:
            self._sleep_start = None

    def _update_neutral(self, t: float) -> None:
        if not self.neutral_required:
            return
        p = self.primary
        if not self.hands:
            self.neutral_required = False              # hands left: that's neutral
            return
        palms = any(c.pose and c.pose.pose == Pose.OPEN_PALM for c in self.hands.values())
        if p is not None and not palms and not p.pinch_on:
            if self._neutral_since is None:
                self._neutral_since = t
            elif t - self._neutral_since >= 0.15:
                self.neutral_required = False
                for c in self.hands.values():
                    if c.swipe:
                        c.swipe.reset()
        else:
            self._neutral_since = None

    # ----------------------------------------------------- primary state
    def _stability(self, ctx: HandCtx, pose: Pose) -> float:
        h = ctx.pose_hist
        return sum(1 for p in h if p == pose) / len(h) if h else 0.0

    def _cool(self, key: str, t: float, cooldown: float) -> bool:
        return t - self._last_event_t.get(key, -1e9) >= cooldown

    def _cursor_point(self, ctx: HandCtx, source: str) -> np.ndarray:
        """Hand point in normalised image coords (what the cursor region uses)."""
        f = ctx.f
        p = {"index": f.index_tip, "palm": f.palm_center, "pinch": f.pinch_point}[source]
        return np.array([p[0] / f.aspect, p[1]])

    def _update_primary(self, t: float, events: list) -> None:
        ctx = self.primary
        # Continuing interactions whose hand is momentarily stale just hold still.
        if ctx is None or ctx.f.stale or ctx.pose is None:
            return
        it = self.inter
        mode_pinch = it is not None and it.kind in ("pinch", "drag")

        # ---- cursor
        if self.draw_mode and self.cfg.draw.pen_down_pose == "pinch":
            src = "pinch"
        else:
            src = "palm" if (mode_pinch or (it and it.kind == "resize")) else "index"
        cursor = self.mapper.update(self._cursor_point(ctx, src), t, source=src,
                                    engaged=mode_pinch or bool(it and it.kind in ("resize", "draw")))

        if it is not None and it.kind == "resize":
            self._update_resize(ctx, t, events)
            return
        if mode_pinch:
            self._update_pinch(ctx, t, events, cursor)
            return
        if self.draw_mode:
            self._update_draw(ctx, t, events, cursor)
            return

        pose = ctx.pose.pose
        # ---- pinch begins (index pinch)
        if ctx.pinch_rose:
            self._end_passive(t)
            rewind = self.mapper.position_at(t - self.cfg.cursor.pinch_rewind_s)
            anchor = rewind if rewind is not None else cursor
            conf = combine({"geometry": pinch_closeness(ctx.f.pinch_ratio),
                            "tracking": self._tracking(ctx),
                            "stability": 1.0 - 0.5 * self._stability(ctx, Pose.FIST)})
            self._start("pinch", ctx, t, anchor_px=anchor, palm0=self._palm_hl(ctx), mapper0=cursor)
            self._cursor_lock = anchor.copy()
            self._emit(events, "pinch_start", t, conf, ctx.track_id, point=anchor.copy())
            return

        # ---- right click: middle-finger pinch, once per pose entry
        if pose == Pose.MIDDLE_PINCH and ctx.pose.held(t) < 0.05 and self._cool("right_click", t, 0.5):
            conf = combine({"geometry": ctx.pose.score, "tracking": self._tracking(ctx),
                            "stability": self._stability(ctx, Pose.MIDDLE_PINCH)})
            self._emit(events, "right_click", t, conf, ctx.track_id, point=cursor.copy())

        # ---- scroll (two fingers, vertical)
        if pose == Pose.TWO and ctx.pose.held(t) >= 0.12:
            if it is None or it.kind != "scroll":
                self._end_passive(t)
                self._start("scroll", ctx, t)
                self._scroll_inertia = False
            self._update_scroll(ctx, t)
        elif it is not None and it.kind == "scroll":
            self.inter = None
            self._scroll_inertia = abs(self._scroll_v) > 60      # let it coast

        # ---- volume (shaka + vertical)
        if pose == Pose.SHAKA and ctx.pose.held(t) >= 0.15:
            if it is None or it.kind != "volume":
                self._end_passive(t)
                self._start("volume", ctx, t, vol_anchor=float(self._palm_hl(ctx)[1]))
            self._update_volume(ctx, t, events)
        elif it is not None and it.kind == "volume":
            self.inter = None

        # ---- swipes (palm / two / three)
        self._update_swipes(ctx, t, events)
        # ---- holds (fist / three)
        self._update_holds(ctx, t, events)

    def _end_passive(self, t: float) -> None:
        if self.inter is not None and self.inter.kind in ("scroll", "volume"):
            self.inter = None
        self._scroll_v = 0.0
        self._scroll_inertia = False

    # ----------------------------------------------------------- pinch / drag
    def _update_pinch(self, ctx: HandCtx, t: float, events: list, cursor: np.ndarray) -> None:
        it = self.inter
        g = self._g
        held = t - it.t0
        moved = float(np.linalg.norm(self._palm_hl(ctx) - it.palm0))
        drag_point = it.anchor_px + (cursor - it.mapper0)
        if it.kind == "drag":
            self._drag_hist.push(t, drag_point)

        if ctx.pinch_on:
            if it.kind == "pinch":
                if moved > g.drag_start_move:
                    it.kind, it.drag_t0 = "drag", t
                    self._cursor_lock = None
                    self._drag_hist.clear()
                    self._drag_hist.push(t, drag_point)
                    self._emit(events, "drag_start", t, combine({"tracking": self._tracking(ctx)}),
                               ctx.track_id, point=drag_point.copy(), anchor=it.anchor_px.copy())
                elif held >= g.grab_hold_s and not it.extra.get("grabbed"):
                    it.extra["grabbed"] = True
                    self._emit(events, "grab", t, combine({"tracking": self._tracking(ctx),
                                                           "duration": 1.0}),
                               ctx.track_id, point=it.anchor_px.copy())
            # A second hand pinching while we hold a window starts a resize.
            other = self._other_pinching(ctx)
            if other is not None and (it.kind == "drag" or it.extra.get("grabbed")):
                self._begin_resize(ctx, other, t, events)
            return

        if not ctx.pinch_fell:
            return
        # Observed release (hand is tracked and not stale).
        if it.kind == "pinch":
            if held <= g.click_max_s and moved <= g.click_max_move:
                conf = combine({"geometry": 1.0, "tracking": self._tracking(ctx),
                                "duration": 1.0 if held > 0.04 else 0.5})
                self._emit(events, "click", t, conf, ctx.track_id, point=it.anchor_px.copy())
            else:
                self._emit(events, "release", t, combine({"tracking": 1.0}), ctx.track_id)
        else:
            throw = self._evaluate_throw(ctx, t)
            if throw is not None:
                self._emit(events, "throw", t, throw.pop("confidence"), ctx.track_id, **throw)
            else:
                self._emit(events, "drag_end", t, combine({"tracking": 1.0}), ctx.track_id,
                           point=drag_point.copy())
        self.inter = None
        self._cursor_lock = None

    def _evaluate_throw(self, ctx: HandCtx, t: float) -> dict | None:
        g = self._g
        it = self.inter
        if it.kind != "drag" or it.drag_t0 is None or t - it.drag_t0 < 0.12:
            return None
        # Evaluate motion *before* the fingers started opening (last frame
        # excluded): opening the pinch jostles landmarks and would fake speed.
        t_end = t - 0.034
        w = g.throw_window_s
        samples = [s for s in ctx.pos_hl.since(t_end - w) if s.t <= t_end]
        if len(samples) < 3:
            return None
        mid = len(samples) // 2
        v_all = regression_velocity(samples)
        v_a = regression_velocity(samples[:mid + 1])
        v_b = regression_velocity(samples[mid:])
        speed = float(np.linalg.norm(v_all))
        disp = float(np.linalg.norm(samples[-1].value - samples[0].value))
        if speed < g.throw_min_speed * 0.8 or disp < 0.12:
            return None
        sustained = min(np.linalg.norm(v_a), np.linalg.norm(v_b)) / max(1e-6, speed)
        cos = float(np.dot(v_a, v_b) / (np.linalg.norm(v_a) * np.linalg.norm(v_b) + 1e-9))
        angle = math.atan2(v_all[1], v_all[0])
        down = v_all[1] > 0 and abs(v_all[1]) > abs(v_all[0])
        need = g.throw_min_speed * (1.3 if down else 1.0)       # minimize needs more intent
        px = [s for s in self._drag_hist.since(t_end - w)]
        v_px = regression_velocity(px) if len(px) >= 2 else np.zeros(2)
        conf = combine({
            "velocity": velocity_margin(speed, need, 0.3),
            "direction": max(0.0, cos),
            "stability": min(1.0, sustained / 0.6),
            "duration": duration_factor(t - it.drag_t0, 0.2),
            "tracking": self._tracking(ctx),
        })
        if speed < need:
            return None
        rel = self._drag_hist.latest.value if self._drag_hist.latest else it.anchor_px
        return {"confidence": conf, "velocity_px": v_px, "velocity_hl": v_all, "speed": speed,
                "angle": angle, "release_point": np.asarray(rel).copy()}

    def throw_preview(self, t: float) -> np.ndarray | None:
        """While dragging fast, the px velocity a release right now would throw with."""
        it, ctx = self.inter, self.primary
        if it is None or it.kind != "drag" or ctx is None:
            return None
        if ctx.speed < self._g.throw_min_speed:
            return None
        px = self._drag_hist.since(t - self._g.throw_window_s)
        return regression_velocity(px) if len(px) >= 2 else None

    # ----------------------------------------------------------------- resize
    def _other_pinching(self, ctx: HandCtx) -> HandCtx | None:
        for c in self.hands.values():
            if c is not ctx and not c.f.stale and c.pinch_rose:
                return c
        return None

    def _begin_resize(self, ctx: HandCtx, other: HandCtx, t: float, events: list) -> None:
        it = self.inter
        p1, p2 = self._pinch_px(ctx), self._pinch_px(other)
        it.kind, it.other = "resize", other.track_id
        it.extra["p0"] = (p1, p2)
        conf = combine({"tracking": min(self._tracking(ctx), self._tracking(other)),
                        "geometry": pinch_closeness(other.f.pinch_ratio)})
        self._emit(events, "resize_start", t, conf, ctx.track_id, points=(p1, p2))

    def _pinch_px(self, ctx: HandCtx) -> np.ndarray:
        """Absolute desktop mapping of a hand's pinch point (no hybrid state,
        so it works for the secondary hand too)."""
        return self.mapper.to_desktop(self._cursor_point(ctx, "pinch"))

    def _update_resize(self, ctx: HandCtx, t: float, events: list) -> None:
        it = self.inter
        other = self.hands.get(it.other)
        if other is None or other.f.stale:
            return
        if not ctx.pinch_on or not other.pinch_on:
            self._end_resize(events, t, keep_drag=ctx.pinch_on)
            return
        it.extra["points"] = (self._pinch_px(ctx), self._pinch_px(other))

    def _end_resize(self, events: list, t: float, keep_drag: bool) -> None:
        it = self.inter
        self._emit(events, "resize_end", t, combine({"tracking": 1.0}), it.hand)
        ctx = self.hands.get(it.hand)
        if keep_drag and ctx is not None and ctx.pinch_on:
            # Rebase the drag so the window doesn't jump back to the pre-resize anchor.
            cur = self.mapper.position if self.mapper.position is not None else it.anchor_px
            self._start("drag", ctx, t, anchor_px=cur.copy(),
                        palm0=self._palm_hl(ctx), mapper0=cur)
            self.inter.drag_t0 = t
            self._drag_hist.clear()
            self._emit(events, "drag_start", t, combine({"tracking": 1.0}), ctx.track_id,
                       point=cur.copy(), anchor=cur.copy(), rebase=True)
        else:
            self.inter = None

    # ----------------------------------------------------------------- scroll
    def _update_scroll(self, ctx: HandCtx, t: float) -> None:
        g = self._g
        vx, vy = float(ctx.velocity[0]), float(ctx.velocity[1])
        target = 0.0
        if abs(vy) > g.scroll_dead_zone and abs(vy) > 0.8 * abs(vx):
            # Hand up (vy < 0) scrolls up (positive wheel). Mild acceleration.
            target = -vy * g.scroll_gain * (1.0 + 0.35 * abs(vy))
        self._scroll_v += 0.5 * (target - self._scroll_v)

    def _update_scroll_inertia(self, t: float) -> None:
        if not self._scroll_inertia:
            if self.inter is None or self.inter.kind != "scroll":
                self._scroll_v = 0.0
            return
        dt = 1 / 30
        self._scroll_v *= math.exp(-self._g.scroll_inertia_decay * dt)
        if abs(self._scroll_v) < 25:
            self._scroll_v, self._scroll_inertia = 0.0, False

    # ----------------------------------------------------------------- volume
    def _update_volume(self, ctx: HandCtx, t: float, events: list) -> None:
        it = self.inter
        y = float(self._palm_hl(ctx)[1])
        dy = it.vol_anchor - y               # up = positive
        step = self._g.volume_step
        while abs(dy) >= step:
            direction = 1 if dy > 0 else -1
            conf = combine({"geometry": ctx.pose.score, "tracking": self._tracking(ctx),
                            "stability": self._stability(ctx, Pose.SHAKA)})
            self._emit(events, "volume_step", t, conf, ctx.track_id, direction=direction)
            it.vol_anchor -= direction * step
            dy = it.vol_anchor - y

    # ----------------------------------------------------------------- swipes
    def _update_swipes(self, ctx: HandCtx, t: float, events: list) -> None:
        pose = ctx.pose.pose
        fam = SWIPE_POSES.get(pose)
        if self.inter is not None and self.inter.kind not in ("scroll",):
            fam = None
        if fam != ctx.swipe_family:
            if ctx.swipe is None:
                ctx.swipe = self._swipe_detector()
            ctx.swipe.cancel_stroke()        # cooldown/lockout state survives pose changes
            ctx.swipe_family = fam
        if fam is None:
            return
        # The pose must be established before a stroke may begin.
        res = ctx.swipe.update(self._palm_hl(ctx), ctx.velocity, t,
                               geometry=ctx.pose.score, tracking=self._tracking(ctx),
                               can_start=ctx.pose.held(t) >= 0.12)
        if res is not None:
            if fam == "two_finger" and self.inter and self.inter.kind == "scroll":
                self._scroll_v = 0.0         # a sideways flick shouldn't also scroll
            self._emit(events, "swipe", t, res.confidence, ctx.track_id,
                       family=fam, direction=res.direction, distance=res.distance,
                       peak_speed=res.peak_speed)

    # ------------------------------------------------------------------ holds
    def _update_holds(self, ctx: HandCtx, t: float, events: list) -> None:
        pose = ctx.pose.pose
        g = self._g
        still = ctx.speed < g.hold_still_speed
        if pose in HOLD_POSES and still and self.inter is None:
            if ctx.hold_pose != pose:
                ctx.hold_pose, ctx.hold_start, ctx.hold_fired = pose, t, False
            need = g.fist_hold_s if pose == Pose.FIST else g.three_hold_s
            if not ctx.hold_fired and t - ctx.hold_start >= need:
                ctx.hold_fired = True
                conf = combine({"geometry": ctx.pose.score, "duration": 1.0,
                                "stability": self._stability(ctx, pose),
                                "tracking": self._tracking(ctx),
                                "velocity": 1.0 - min(1.0, ctx.speed / g.hold_still_speed) * 0.5})
                self._emit(events, "hold", t, conf, ctx.track_id, pose=HOLD_POSES[pose])
        elif pose != ctx.hold_pose or not still:
            if pose != ctx.hold_pose:
                ctx.hold_pose, ctx.hold_fired = None, False
            ctx.hold_start = t if pose in HOLD_POSES else None

    def _hold_progress(self, ctx: HandCtx, t: float) -> tuple[float, str]:
        if ctx.hold_pose in HOLD_POSES and ctx.hold_start is not None and not ctx.hold_fired:
            need = self._g.fist_hold_s if ctx.hold_pose == Pose.FIST else self._g.three_hold_s
            return min(1.0, (t - ctx.hold_start) / need), HOLD_POSES[ctx.hold_pose]
        return 0.0, ""

    # ------------------------------------------------------------------- draw
    def _update_draw(self, ctx: HandCtx, t: float, events: list, cursor: np.ndarray) -> None:
        it = self.inter
        pose = ctx.pose.pose
        pen = ctx.pinch_on if self.cfg.draw.pen_down_pose == "pinch" else pose == Pose.POINT
        if pen:
            if it is None or it.kind != "draw":
                self._start("draw", ctx, t)
                self._emit(events, "stroke_begin", t, combine({"tracking": self._tracking(ctx)}),
                           ctx.track_id, point=cursor.copy())
            else:
                self._emit(events, "stroke_point", t, combine({"tracking": 1.0}), ctx.track_id,
                           point=cursor.copy())
            return
        if it is not None and it.kind == "draw":
            self._emit(events, "stroke_end", t, combine({"tracking": 1.0}), ctx.track_id)
            self.inter = None
        if pose == Pose.OPEN_PALM and ctx.pose.held(t) > 0.15:
            if self.inter is None:
                self._start("erase", ctx, t)
            eraser = self.mapper.to_desktop(self._cursor_point(ctx, "palm"))
            self.inter.extra["eraser"] = eraser
            self._emit(events, "erase_point", t, combine({"tracking": 1.0}), ctx.track_id,
                       point=eraser)
        elif self.inter is not None and self.inter.kind == "erase":
            self.inter = None
        if self.inter is None:
            self._update_swipes(ctx, t, events)
            self._update_holds(ctx, t, events)

    # ----------------------------------------------------------------- cancel
    def _cancel(self, events: list, t: float, reason: str) -> None:
        it = self.inter
        if it is None:
            return
        if it.kind == "draw":
            self._emit(events, "stroke_end", t, combine({"tracking": 1.0}), it.hand)
        self._emit(events, "cancel", t, combine({"tracking": 1.0}), it.hand,
                   what=it.kind, reason=reason)
        self.inter = None
        self._cursor_lock = None
        self._scroll_v = 0.0
        self._scroll_inertia = False

    # --------------------------------------------------------------- snapshot
    def _mode_of(self, ctx: HandCtx) -> HandMode:
        it = self.inter
        if it is not None and (it.hand == ctx.track_id or it.other == ctx.track_id):
            return {
                "pinch": HandMode.GRABBING if it.extra.get("grabbed") else HandMode.PINCH_STARTED,
                "drag": HandMode.DRAGGING, "resize": HandMode.RESIZING,
                "scroll": HandMode.SCROLLING, "volume": HandMode.VOLUME,
                "draw": HandMode.DRAWING, "erase": HandMode.ERASING,
            }[it.kind]
        if ctx.pose is None:
            return HandMode.HOVERING
        if ctx.pose.pose == Pose.POINT:
            return HandMode.POINTING
        return HandMode.HOVERING

    def _snapshot(self, t: float) -> Snapshot:
        views = []
        for ctx in self.hands.values():
            prog, kind = self._hold_progress(ctx, t)
            views.append(HandView(
                track_id=ctx.track_id, label=ctx.f.label, mode=self._mode_of(ctx),
                pose=ctx.pose.pose.value if ctx.pose else "none",
                pose_score=ctx.pose.score if ctx.pose else 0.0, pinch=ctx.pinch_on,
                pinch_ratio=ctx.f.pinch_ratio, speed=ctx.speed, velocity=ctx.velocity.copy(),
                iso=ctx.f.iso, aspect=ctx.f.aspect, primary=ctx.track_id == self.primary_id,
                stale=ctx.f.stale, palm_facing=ctx.f.palm_facing,
                hold_progress=prog, hold_kind=kind))
        p = self.primary
        primary_mode = self._mode_of(p) if p is not None else HandMode.IDLE
        cursor = self._cursor_lock if self._cursor_lock is not None else self.mapper.position
        drives = primary_mode in (HandMode.POINTING, HandMode.PINCH_STARTED, HandMode.GRABBING,
                                  HandMode.DRAGGING, HandMode.DRAWING)
        it = self.inter
        drag_point = None
        if it is not None and it.kind == "drag" and self._drag_hist.latest is not None:
            drag_point = self._drag_hist.latest.value.copy()
        wake = max((min(1.0, (t - s) / self.cfg.activation.wake_hold_s)
                    for s in self._wake_start.values()), default=0.0)
        sleep = (min(1.0, (t - self._sleep_start) / self.cfg.activation.sleep_hold_s)
                 if self._sleep_start is not None else 0.0)
        return Snapshot(
            t=t, system=self.system, draw_mode=self.draw_mode, hands=views,
            cursor=None if cursor is None else np.asarray(cursor).copy(),
            cursor_visible=p is not None and self.system == SystemState.ACTIVE and not self.neutral_required,
            primary_mode=primary_mode, interaction=it.id if it else 0,
            owner=it.kind if it else None, drag_point=drag_point,
            resize_points=it.extra.get("points") if it and it.kind == "resize" else None,
            scroll_velocity=self._scroll_v if self.system == SystemState.ACTIVE else 0.0,
            wake_progress=wake, sleep_progress=sleep, neutral_required=self.neutral_required,
            throw_preview=self.throw_preview(t),
            confidence={"drives_cursor": float(drives and self.system == SystemState.ACTIVE
                                               and not self.neutral_required)})

