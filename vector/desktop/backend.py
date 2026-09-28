"""Window-system backend interface: the real Win32 one and an in-memory fake
used by tests to exercise intent logic (targeting, snapping, switching)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from vector.core.geometry import Rect
from vector.desktop import windows as W


class Win32Backend:
    def __init__(self, policy: W.WindowPolicy):
        self.policy = policy

    def window_at(self, x: float, y: float) -> int | None:
        return W.window_at(x, y, self.policy)

    def info(self, hwnd: int) -> W.WindowInfo | None:
        if not hwnd or not W.is_window(hwnd):
            return None
        return W.info(hwnd)

    def is_valid_target(self, hwnd: int) -> bool:
        return bool(hwnd) and W.is_window(hwnd) and self.policy.is_candidate(hwnd)

    def foreground(self) -> int | None:
        return W.foreground()

    def switchable(self) -> list[int]:
        return W.switchable_windows(self.policy)

    def process_name(self, hwnd: int) -> str:
        return W.process_name(W.pid_of(hwnd)) if hwnd else ""

    def icon(self, hwnd: int) -> int:
        return W.icon_handle(hwnd)


@dataclass
class FakeWindow:
    hwnd: int
    title: str
    frame: Rect
    process: str = "app.exe"
    maximized: bool = False
    minimized: bool = False
    resizable: bool = True


@dataclass
class FakeBackend:
    """Z-ordered list of fake windows, topmost first."""
    windows: list[FakeWindow] = field(default_factory=list)
    log: list[tuple] = field(default_factory=list)

    def _get(self, hwnd):
        return next((w for w in self.windows if w.hwnd == hwnd), None)

    def window_at(self, x, y):
        for w in self.windows:
            if not w.minimized and w.frame.contains(x, y):
                return w.hwnd
        return None

    def info(self, hwnd):
        w = self._get(hwnd)
        if w is None:
            return None
        return W.WindowInfo(hwnd=w.hwnd, title=w.title, class_name="Fake", pid=1,
                            process=w.process, frame=w.frame, rect=w.frame,
                            maximized=w.maximized, minimized=w.minimized,
                            resizable=w.resizable, topmost=False)

    def is_valid_target(self, hwnd):
        return self._get(hwnd) is not None

    def foreground(self):
        return self.windows[0].hwnd if self.windows else None

    def switchable(self):
        return [w.hwnd for w in self.windows]

    def process_name(self, hwnd):
        w = self._get(hwnd)
        return w.process if w else ""

    def icon(self, hwnd):
        return 0

    # mutation helpers used by a fake actuator
    def focus(self, hwnd):
        w = self._get(hwnd)
        if w:
            self.windows.remove(w)
            self.windows.insert(0, replace(w, minimized=False))
            self.log.append(("focus", hwnd))
