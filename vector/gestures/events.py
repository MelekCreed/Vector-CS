"""Engine output types: discrete events + a continuous per-frame snapshot."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vector.gestures.confidence import Confidence


class HandMode(str, enum.Enum):
    IDLE = "IDLE"
    HOVERING = "HOVERING"
    POINTING = "POINTING"
    PINCH_STARTED = "PINCH_STARTED"
    GRABBING = "GRABBING"
    DRAGGING = "DRAGGING"
    THROWING = "THROWING"
    RESIZING = "RESIZING"
    SCROLLING = "SCROLLING"
    DRAWING = "DRAWING"
    ERASING = "ERASING"
    VOLUME = "VOLUME"
    CAROUSEL = "CAROUSEL"


class SystemState(str, enum.Enum):
    SLEEPING = "SLEEPING"
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"     # emergency stop; only a keyboard re-arm leaves it


@dataclass
class GestureEvent:
    kind: str                       # e.g. "click", "grab", "throw", "swipe", "hold", "wake"
    t: float
    confidence: Confidence
    hand_id: int | None = None
    interaction: int = 0            # interaction id (drag/resize lifecycles share one)
    data: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<{self.kind} c={self.confidence.value:.2f} {self.data}>"


@dataclass
class HandView:
    """Per-hand info for the overlay / debug HUD."""
    track_id: int
    label: str
    mode: HandMode
    pose: str
    pose_score: float
    pinch: bool
    pinch_ratio: float
    speed: float                    # hand-lengths / s
    velocity: np.ndarray
    iso: np.ndarray                 # (21,2) landmarks, isotropic image coords
    aspect: float
    primary: bool
    stale: bool
    palm_facing: float
    hold_progress: float = 0.0      # 0..1 for hold gestures (wake/fist/three)
    hold_kind: str = ""


@dataclass
class Snapshot:
    t: float
    system: SystemState
    draw_mode: bool
    hands: list[HandView]
    cursor: np.ndarray | None       # desktop px
    cursor_visible: bool
    primary_mode: HandMode
    interaction: int
    owner: str | None               # what currently owns input ("drag", "resize", ...)
    drag_point: np.ndarray | None = None       # desktop px the dragged window follows
    resize_points: tuple[np.ndarray, np.ndarray] | None = None  # desktop px of both pinches
    scroll_velocity: float = 0.0    # wheel units / s (positive = up)
    carousel_offset: int | None = None           # apps moved from the start selection
    wake_progress: float = 0.0
    sleep_progress: float = 0.0
    neutral_required: bool = False
    throw_preview: np.ndarray | None = None    # px/s velocity while a throw looks likely
    confidence: dict[str, float] = field(default_factory=dict)
