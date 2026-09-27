"""Demo mode: on-screen captions that walk through the showcase sequence and
advance automatically when the matching gesture actually happens — so a
recording explains itself without narration."""

from __future__ import annotations

from dataclasses import dataclass

from vector.gestures.events import HandMode, SystemState

STEPS = [
    ("Raise your hand", "wake"),
    ("Point at a window", "hover"),
    ("Pinch to grab it", "pinch"),
    ("Drag it", "drag"),
    ("Throw it to the other display", "throw"),
    ("Grab another window", "pinch2"),
    ("Two hands — stretch it", "resize"),
    ("Swipe to switch apps", "swipe_palm"),
    ("Two fingers to scroll", "scroll"),
    ("Three fingers — draw mode", "draw_on"),
    ("Draw in the air", "stroke"),
    ("Fist to clear", "clear"),
    ("Both palms — sleep", "sleep"),
]


@dataclass
class DemoView:
    index: int
    total: int
    caption: str
    done: bool
    advanced_at: float


class DemoScript:
    def __init__(self):
        self.i = 0
        self.advanced_at = 0.0
        self._stroke_points = 0

    def _hit(self, key: str, events, fb, snap) -> bool:
        kinds = {e.kind for e in events}
        if key == "wake":
            return "wake" in kinds or (snap.system == SystemState.ACTIVE and self.i == 0)
        if key == "hover":
            return fb.hover_frame is not None and snap.primary_mode == HandMode.POINTING
        if key in ("pinch", "pinch2"):
            return fb.lock_state in ("pinch", "grab", "drag")
        if key == "drag":
            return "drag_start" in kinds
        if key == "throw":
            return "throw" in kinds
        if key == "resize":
            return "resize_start" in kinds
        if key == "swipe_palm":
            return any(e.kind == "swipe" and e.data.get("family") == "palm" for e in events)
        if key == "scroll":
            return abs(snap.scroll_velocity) > 300
        if key == "draw_on":
            return snap.draw_mode
        if key == "stroke":
            self._stroke_points += sum(e.kind == "stroke_point" for e in events)
            return self._stroke_points > 40 and "stroke_end" in kinds
        if key == "clear":
            return any(op == "draw_clear" for op, _ in fb.draw_ops)
        if key == "sleep":
            return "sleep" in kinds
        return False

    def update(self, events, fb, snap) -> None:
        if self.i >= len(STEPS):
            return
        if self._hit(STEPS[self.i][1], events, fb, snap):
            self.i += 1
            self.advanced_at = snap.t

    def view(self) -> DemoView:
        done = self.i >= len(STEPS)
        cap = "" if done else STEPS[self.i][0]
        return DemoView(self.i, len(STEPS), cap, done, self.advanced_at)
