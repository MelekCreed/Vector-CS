"""Actuator: the only component that touches the OS, and the last safety gate.

Runs its own ~120 Hz thread:
* drains the discrete command queue, dropping commands from an older
  activation *generation* (system was disabled/slept since) or past expiry
* turns 30 Hz camera-rate targets into smooth 120 Hz motion with critically
  damped springs (cursor, dragged window, resize rectangle)
* integrates scroll velocity into wheel events (with sub-notch precision)
* watchdog: if vision stops delivering, motion stops; a long stall ends drags
All OS calls go through an injectable ``ops`` object (Win32Ops in the app,
a recording fake in tests).
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from vector.config import Config
from vector.core.filters import CriticalSpring
from vector.core.geometry import Rect
from vector.desktop.commands import Command, Mailbox, Motion

log = logging.getLogger(__name__)

STALE_MOTION_S = 0.4       # stop continuous motion if vision is this late
WATCHDOG_END_DRAG_S = 1.5  # end a drag if vision has been silent this long
SCROLL_MIN_DELTA = 30      # wheel units per event (120 = one notch)


class Win32Ops:
    """Real OS operations."""

    def __init__(self):
        from vector.desktop import input as I
        from vector.desktop import windows as W
        self.I, self.W = I, W

    def info(self, hwnd):
        return self.W.info(hwnd) if hwnd and self.W.is_window(hwnd) else None

    def set_frame(self, hwnd, rect: Rect, insets=None, move_only=False):
        return self.W.set_frame(hwnd, rect, insets, move_only)

    def move_cursor(self, x, y):
        self.I.move_cursor(x, y)

    def cursor_pos(self):
        return self.I.cursor_pos()

    def click(self, button="left"):
        self.I.click(button)

    def wheel(self, delta):
        self.I.wheel(delta)

    def press_chord(self, chord):
        self.I.press_chord(chord)

    def focus(self, hwnd):
        return self.W.focus(hwnd)

    def maximize(self, hwnd):
        self.W.maximize(hwnd)

    def minimize(self, hwnd):
        self.W.minimize(hwnd)

    def restore(self, hwnd):
        self.W.restore(hwnd)

    def release_all_buttons(self):
        self.I.release_all_buttons()


@dataclass
class _Drag:
    hwnd: int
    anchor: np.ndarray            # desktop px the hand locked on at
    start: Rect | None            # window frame when the drag began (None while restoring)
    insets: tuple | None
    restoring: bool = False
    restore_deadline: float = 0.0
    anchor_frac: float = 0.5
    last_sent: Rect | None = None
    insets_t: float = 0.0


@dataclass
class _Pending:
    """A follow-up that must wait for an async window state change."""
    hwnd: int
    rect: Rect | None
    maximize_after: bool
    deadline: float


@dataclass
class ActuatorStats:
    commands: int = 0
    dropped_generation: int = 0
    dropped_expired: int = 0
    last_cmd_latency_ms: float = 0.0     # camera frame -> OS call
    last_cmd_exec_ms: float = 0.0
    tick_hz: float = 0.0
    motion_age_ms: float = 0.0
    recent: collections.deque = field(default_factory=lambda: collections.deque(maxlen=64))


class Actuator:
    def __init__(self, cfg: Config, ops, mailbox: Mailbox | None = None, rate_hz: float = 120.0,
                 clock=time.perf_counter):
        self.cfg = cfg
        self.ops = ops
        self.mailbox = mailbox or Mailbox()
        self.rate_hz = rate_hz
        self.clock = clock
        self._q: collections.deque[Command] = collections.deque(maxlen=64)
        self._qlock = threading.Lock()
        self._gen = 0
        self._armed = False
        self._latched = False            # emergency stop: arm() refuses until unlatch()
        self._tick_lock = threading.RLock()
        self._cursor = CriticalSpring(cfg.cursor.spring_hz)
        self._drag_spring = CriticalSpring(cfg.cursor.spring_hz)
        self._resize_spring = CriticalSpring(cfg.cursor.spring_hz * 0.8)
        self._cursor_idle_since: float | None = None
        self._last_cursor: np.ndarray | None = None
        self.drag: _Drag | None = None
        self.resize_hwnd: int | None = None
        self._resize_last: Rect | None = None
        self._pending: list[_Pending] = []
        self._scroll_acc = 0.0
        self._thread: threading.Thread | None = None
        self._running = False
        self.stats = ActuatorStats()
        self._last_tick: float | None = None

    # ------------------------------------------------------------ control
    @property
    def generation(self) -> int:
        return self._gen

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def latched(self) -> bool:
        return self._latched

    def arm(self) -> bool:
        """Returns False (and stays disarmed) while an emergency stop is latched,
        so no other thread can race the failsafe back into an armed state."""
        with self._tick_lock:
            if self._latched:
                return False
            self._armed = True
            return True

    def unlatch(self) -> None:
        with self._tick_lock:
            self._latched = False

    def disarm(self, reason: str = "", latch: bool = False) -> None:
        """Instantly stop everything: bump the generation (invalidating every
        queued or in-flight command), drop motion, release buttons. Serialised
        with tick() so a half-finished tick can't move a window afterwards."""
        with self._tick_lock:
            self._disarm_locked(reason, latch)

    def _disarm_locked(self, reason: str, latch: bool) -> None:
        if latch:
            self._latched = True
        self._gen += 1
        self._armed = False
        with self._qlock:
            self._q.clear()
        self.drag = None
        self.resize_hwnd = None
        self._pending.clear()
        self._scroll_acc = 0.0
        self.mailbox.clear()
        try:
            self.ops.release_all_buttons()
        except Exception:  # pragma: no cover
            log.exception("release_all_buttons failed")
        if reason:
            log.warning("actuator disarmed: %s", reason)

    def submit(self, cmd: Command) -> None:
        with self._qlock:
            self._q.append(cmd)

    # ------------------------------------------------------------- thread
    def start(self) -> "Actuator":
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="actuator", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        period = 1.0 / self.rate_hz
        n, t_rate = 0, self.clock()
        while self._running:
            t0 = self.clock()
            try:
                self.tick(t0)
            except Exception:
                log.exception("actuator tick failed")
            n += 1
            if t0 - t_rate >= 1.0:
                self.stats.tick_hz = n / (t0 - t_rate)
                n, t_rate = 0, t0
            # Disarmed and idle: tick at 20 Hz so we don't steal CPU from inference.
            idle = not self._armed and not self._q and not self._pending
            p = 0.05 if idle else period
            time.sleep(max(0.0, p - (self.clock() - t0)))

    # --------------------------------------------------------------- tick
    def tick(self, now: float) -> None:
        with self._tick_lock:
            self._tick(now)

    def _tick(self, now: float) -> None:
        dt = 1 / self.rate_hz if self._last_tick is None else min(0.05, max(1e-4, now - self._last_tick))
        self._last_tick = now
        self._drain(now)
        self._process_pending(now)
        m = self.mailbox.get()
        if not self._armed or m is None:
            self._scroll_acc = 0.0
            return
        age = now - m.t_capture
        self.stats.motion_age_ms = age * 1000
        if age > STALE_MOTION_S:
            self._scroll_acc = 0.0
            if age > WATCHDOG_END_DRAG_S and (self.drag or self.resize_hwnd):
                log.warning("watchdog: vision stalled %.1fs, ending drag/resize", age)
                self.drag, self.resize_hwnd = None, None
            return
        self._update_cursor(m, now, dt)
        self._update_drag(m, now, dt)
        self._update_resize(m, dt)
        self._update_scroll(m, dt)

    def _drain(self, now: float) -> None:
        with self._qlock:
            cmds = list(self._q)
            self._q.clear()
        for c in cmds:
            if c.generation != self._gen:
                self.stats.dropped_generation += 1
                continue
            if not self._armed and c.kind not in ("drag_end", "resize_end"):
                self.stats.dropped_generation += 1
                continue
            if c.expired(now):
                self.stats.dropped_expired += 1
                log.debug("dropped expired %s (%.0f ms old)", c.kind, (now - c.t_capture) * 1000)
                continue
            t0 = self.clock()
            try:
                self._execute(c, now)
            except Exception:
                log.exception("command %s failed", c.kind)
            s = self.stats
            s.commands += 1
            s.last_cmd_exec_ms = (self.clock() - t0) * 1000
            s.last_cmd_latency_ms = (self.clock() - c.t_capture) * 1000
            s.recent.append((now, c.kind, s.last_cmd_latency_ms))

    def _execute(self, c: Command, now: float) -> None:
        a, ops = c.args, self.ops
        k = c.kind
        if k == "click":
            p = np.asarray(a["point"], dtype=float)
            ops.move_cursor(*p)
            self._cursor.reset(p)
            ops.click(a.get("button", "left"))
        elif k == "drag_begin":
            self._begin_drag(a["hwnd"], np.asarray(a["anchor"], float), np.asarray(a["point"], float), now)
        elif k == "drag_end":
            self.drag = None
        elif k == "resize_begin":
            inf = ops.info(a["hwnd"])
            if inf is not None:
                self.resize_hwnd = a["hwnd"]
                self._resize_insets = inf.insets
                self._resize_spring.reset(np.array([inf.frame.x, inf.frame.y, inf.frame.w, inf.frame.h]))
                self._resize_last = inf.frame
        elif k == "resize_end":
            self.resize_hwnd = None
        elif k == "set_frame":
            self._set_frame_when_restored(a["hwnd"], a["rect"], a.get("maximize_after", False), now)
        elif k in ("maximize", "minimize", "restore"):
            getattr(ops, k)(a["hwnd"])
        elif k == "focus":
            ops.focus(a["hwnd"])
        elif k == "keys":
            ops.press_chord(a["chord"])
        elif k == "release_all":
            ops.release_all_buttons()
        else:
            log.warning("unknown command %s", k)

    # --------------------------------------------------------------- drag
    def _begin_drag(self, hwnd: int, anchor: np.ndarray, point: np.ndarray, now: float) -> None:
        inf = self.ops.info(hwnd)
        if inf is None:
            return
        self.ops.focus(hwnd)
        self._drag_spring.reset(point)
        if inf.maximized:
            # Like Windows: dragging a maximized window restores it under the
            # hand, keeping the grab point at the same relative x position.
            frac = (anchor[0] - inf.frame.x) / max(1.0, inf.frame.w)
            self.ops.restore(hwnd)
            self.drag = _Drag(hwnd, anchor, None, None, restoring=True,
                              restore_deadline=now + 0.6, anchor_frac=float(np.clip(frac, 0.1, 0.9)))
            return
        # Offsets are measured from the *current* hand point so the window
        # never jumps when the drag threshold is crossed.
        self.drag = _Drag(hwnd, point.copy(), inf.frame, inf.insets, insets_t=now)

    def _update_drag(self, m: Motion, now: float, dt: float) -> None:
        d = self.drag
        if d is None or m.drag_point is None:
            return
        target = np.asarray(m.drag_point, dtype=float)
        p = self._drag_spring.step(target, dt)
        if d.restoring:
            inf = self.ops.info(d.hwnd)
            if inf is None:
                self.drag = None
                return
            if inf.maximized and now < d.restore_deadline:
                return
            fr = inf.frame
            start = Rect(p[0] - d.anchor_frac * fr.w, p[1] - 16, fr.w, fr.h)
            d.start, d.insets, d.anchor, d.restoring, d.insets_t = start, inf.insets, p.copy(), False, now
        if now - d.insets_t > 0.5:           # borders can change with DPI / state
            inf = self.ops.info(d.hwnd)
            if inf is None:
                self.drag = None
                return
            d.insets, d.insets_t = inf.insets, now
        frame = d.start.translated(*(p - d.anchor)).rounded()
        if d.last_sent is None or abs(frame.x - d.last_sent.x) >= 1 or abs(frame.y - d.last_sent.y) >= 1:
            self.ops.set_frame(d.hwnd, frame, d.insets, move_only=True)
            d.last_sent = frame

    # ------------------------------------------------------------- resize
    def _update_resize(self, m: Motion, dt: float) -> None:
        if self.resize_hwnd is None or m.resize_rect is None:
            return
        r = m.resize_rect
        v = self._resize_spring.step(np.array([r.x, r.y, r.w, r.h]), dt)
        rect = Rect(*v).rounded()
        last = self._resize_last
        if last is None or any(abs(a - b) >= 1 for a, b in zip((rect.x, rect.y, rect.w, rect.h),
                                                                (last.x, last.y, last.w, last.h))):
            self.ops.set_frame(self.resize_hwnd, rect, self._resize_insets)
            self._resize_last = rect

    # ------------------------------------------------------------- cursor
    def _update_cursor(self, m: Motion, now: float, dt: float) -> None:
        if not m.drive_cursor or m.cursor is None:
            if self._cursor_idle_since is None:
                self._cursor_idle_since = now
            return
        target = np.asarray(m.cursor, dtype=float)
        if self._cursor_idle_since is not None and now - self._cursor_idle_since > 0.3:
            self._cursor.reset(target)      # resuming after a pause: don't glide from far away
        self._cursor_idle_since = None
        p = self._cursor.step(target, dt)
        if self._last_cursor is None or np.abs(p - self._last_cursor).max() >= 0.5:
            self.ops.move_cursor(*p)
            self._last_cursor = p

    @property
    def cursor_position(self) -> np.ndarray | None:
        return None if self._cursor.pos is None else self._cursor.pos.copy()

    # ------------------------------------------------------------- scroll
    def _update_scroll(self, m: Motion, dt: float) -> None:
        v = m.scroll_velocity
        if v == 0.0:
            self._scroll_acc = 0.0
            return
        self._scroll_acc += v * dt
        if abs(self._scroll_acc) >= SCROLL_MIN_DELTA:
            n = int(self._scroll_acc)
            self.ops.wheel(n)
            self._scroll_acc -= n

    # ------------------------------------------------ async follow-ups
    def _set_frame_when_restored(self, hwnd: int, rect: Rect, maximize_after: bool, now: float) -> None:
        inf = self.ops.info(hwnd)
        if inf is None:
            return
        if inf.maximized or inf.minimized:
            self.ops.restore(hwnd)
            self._pending.append(_Pending(hwnd, rect, maximize_after, now + 0.6))
            return
        self.ops.set_frame(hwnd, rect, inf.insets)
        if maximize_after:
            self._pending.append(_Pending(hwnd, None, True, now + 0.6))

    def _process_pending(self, now: float) -> None:
        keep = []
        for p in self._pending:
            inf = self.ops.info(p.hwnd)
            if inf is None:
                continue
            if p.rect is not None:
                if (inf.maximized or inf.minimized) and now < p.deadline:
                    keep.append(p)
                    continue
                self.ops.set_frame(p.hwnd, p.rect, inf.insets)
                if p.maximize_after:
                    keep.append(_Pending(p.hwnd, None, True, now + 0.6))
            elif p.maximize_after:
                self.ops.maximize(p.hwnd)
        self._pending = keep
