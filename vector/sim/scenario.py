"""Scripted hand-motion scenarios for tests, benchmarks and tuning.

A scenario is a list of keyframes per hand; positions interpolate linearly,
pose parameters (pose name, pinch gap, rotations) switch at keyframes. Noise
and frame drops can be injected to mimic a real webcam.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from vector.sim.synthetic_hand import HandPose, build_detection
from vector.vision.tracker import Detection


@dataclass
class Key:
    t: float
    pose: HandPose | None            # None = hand not visible from here on


@dataclass
class Scenario:
    fps: float = 30.0
    noise: float = 0.0015            # landmark noise in normalised units (~1px @640)
    drop_rate: float = 0.0           # probability a hand is missed in a frame
    seed: int = 0
    aspect: float = 4 / 3
    hands: dict[str, list[Key]] = field(default_factory=dict)

    def key(self, hand: str, t: float, pose: HandPose | None) -> "Scenario":
        self.hands.setdefault(hand, []).append(Key(t, pose))
        self.hands[hand].sort(key=lambda k: k.t)
        return self

    @property
    def duration(self) -> float:
        return max((k.t for ks in self.hands.values() for k in ks), default=0.0)

    def pose_at(self, hand: str, t: float) -> HandPose | None:
        ks = self.hands.get(hand, [])
        prev = None
        for k in ks:
            if k.t <= t:
                prev = k
            else:
                if prev is None or prev.pose is None:
                    return None
                if k.pose is None:
                    return prev.pose
                a = (t - prev.t) / (k.t - prev.t)
                p0, p1 = prev.pose, k.pose
                return replace(p0, x=p0.x + (p1.x - p0.x) * a, y=p0.y + (p1.y - p0.y) * a,
                               pinch_gap=p0.pinch_gap + (p1.pinch_gap - p0.pinch_gap) * a
                               if p0.pose == p1.pose else p0.pinch_gap,
                               roll=p0.roll + (p1.roll - p0.roll) * a,
                               scale=p0.scale + (p1.scale - p0.scale) * a)
        return prev.pose if prev else None

    def frames(self, t0: float = 0.0):
        """Yield (t, detections) at the scenario frame rate."""
        rng = np.random.default_rng(self.seed)
        n = int(self.duration * self.fps) + 1
        for i in range(n):
            t = i / self.fps
            dets: list[Detection] = []
            for hand in self.hands:
                hp = self.pose_at(hand, t)
                if hp is None or (self.drop_rate and rng.random() < self.drop_rate):
                    continue
                dets.append(build_detection(hp, self.aspect, self.noise, rng))
            yield t0 + t, dets


def run(pipeline, scenario: Scenario, t0: float = 0.0):
    """Feed a scenario through a GesturePipeline. Returns (events, snapshots)."""
    events, snaps = [], []
    for t, dets in scenario.frames(t0):
        ev, snap = pipeline.process(dets, t, scenario.aspect)
        events.extend(ev)
        snaps.append(snap)
    return events, snaps
