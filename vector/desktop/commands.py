"""Commands flowing from the intent layer to the actuator.

Discrete commands are stamped with the activation *generation* and an
expiry. The actuator refuses anything from an older generation (the system
was disabled / put to sleep since) or anything too old to still reflect the
user's intent. Continuous motion goes through a latest-value mailbox instead
of a queue, so a slow consumer never replays stale positions.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vector.core.geometry import Rect


@dataclass
class Command:
    kind: str                   # click | drag_begin | drag_end | resize_begin | resize_end |
                                # set_frame | maximize | minimize | restore | focus | keys |
                                # app_switch | release_all
    generation: int
    t_capture: float            # camera timestamp of the frame that caused it
    ttl_s: float = 0.35
    interaction: int = 0
    args: dict[str, Any] = field(default_factory=dict)

    def expired(self, now: float) -> bool:
        return now - self.t_capture > self.ttl_s


@dataclass
class Motion:
    t_capture: float
    cursor: np.ndarray | None = None
    drive_cursor: bool = False
    drag_point: np.ndarray | None = None
    resize_rect: Rect | None = None
    scroll_velocity: float = 0.0


class Mailbox:
    def __init__(self):
        self._lock = threading.Lock()
        self._value: Motion | None = None

    def put(self, m: Motion) -> None:
        with self._lock:
            self._value = m

    def get(self) -> Motion | None:
        with self._lock:
            return self._value

    def clear(self) -> None:
        with self._lock:
            self._value = None


def now() -> float:
    return time.perf_counter()
