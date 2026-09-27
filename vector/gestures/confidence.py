"""Gesture confidence.

A confidence is a *weighted geometric mean* of named factors in [0, 1], so a
single bad factor (e.g. flickering tracking) pulls the whole score down.
It is a ranking/gating heuristic, not a calibrated probability, and it never
replaces hard gates (hand tracked, identity stable, target valid) which are
checked separately by the engine and intent layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

DEFAULT_WEIGHTS = {
    "geometry": 1.0,     # how well the hand shape matches the pose
    "stability": 1.0,    # fraction of recent frames agreeing on the pose
    "duration": 0.8,     # held long enough relative to requirement
    "velocity": 1.0,     # margin above a motion threshold
    "tracking": 1.0,     # detector presence x handedness certainty
    "context": 0.7,      # target / mode appropriateness
    "direction": 0.8,    # motion direction consistency
}


@dataclass
class Confidence:
    value: float
    factors: dict[str, float] = field(default_factory=dict)

    def __float__(self) -> float:
        return self.value

    def weakest(self) -> str | None:
        return min(self.factors, key=self.factors.get) if self.factors else None

    def describe(self) -> str:
        parts = " ".join(f"{k[:4]}={v:.2f}" for k, v in self.factors.items())
        return f"{self.value:.2f} [{parts}]"


def combine(factors: dict[str, float], weights: dict[str, float] | None = None) -> Confidence:
    weights = weights or DEFAULT_WEIGHTS
    num = den = 0.0
    clean = {}
    for name, v in factors.items():
        v = min(1.0, max(1e-3, float(v)))
        clean[name] = v
        w = weights.get(name, 1.0)
        num += w * math.log(v)
        den += w
    value = math.exp(num / den) if den else 0.0
    return Confidence(value, clean)


def velocity_margin(speed: float, threshold: float, softness: float = 0.35) -> float:
    """~0.5 exactly at threshold, ->1 comfortably above, ->0 below."""
    if threshold <= 0:
        return 1.0
    x = (speed / threshold - 1.0) / softness
    return 1.0 / (1.0 + math.exp(-4.0 * x))


def duration_factor(held_s: float, required_s: float) -> float:
    if required_s <= 0:
        return 1.0
    return min(1.0, max(0.0, held_s / required_s))
