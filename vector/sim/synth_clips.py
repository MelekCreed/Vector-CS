"""Synthetic labelled clips (for smoke-testing the learned pipeline only —
never a substitute for real recordings when judging accuracy)."""

from __future__ import annotations

import numpy as np

from vector.recording import Clip
from vector.sim.scenario import Scenario
from vector.sim.synthetic_hand import HandPose

LABELS = ["palm_swipe_left", "palm_swipe_right", "two_finger_swipe_left",
          "two_finger_swipe_right", "fist_hold", "none"]


def make_clip(label: str, rng: np.random.Generator, session: int) -> Clip:
    sc = Scenario(noise=float(rng.uniform(0.001, 0.003)), drop_rate=0.03, seed=int(rng.integers(1e9)))
    x0, y = rng.uniform(0.35, 0.65), rng.uniform(0.4, 0.65)
    roll = rng.uniform(-0.25, 0.25)
    if "swipe" in label:
        pose = "open" if label.startswith("palm") else "two"
        d = 1 if label.endswith("right") else -1
        dist = rng.uniform(0.25, 0.4)
        dur = rng.uniform(0.15, 0.3)
        hold = rng.uniform(0.35, 0.6)
        start = x0 - d * dist / 2
        sc.key("R", 0, HandPose(pose, x=start, y=y, roll=roll))
        sc.key("R", hold, HandPose(pose, x=start, y=y, roll=roll))
        sc.key("R", hold + dur, HandPose(pose, x=start + d * dist, y=y + rng.uniform(-.03, .03), roll=roll))
        sc.key("R", hold + dur + 0.35, HandPose(pose, x=start + d * dist, y=y, roll=roll))
    elif label == "fist_hold":
        sc.key("R", 0, HandPose("point", x=x0, y=y, roll=roll))
        sc.key("R", 0.2, HandPose("fist", x=x0, y=y, roll=roll))
        sc.key("R", rng.uniform(1.0, 1.3), HandPose("fist", x=x0 + rng.uniform(-.01, .01), y=y, roll=roll))
    else:  # none: aimless movement
        t = 0.0
        for _ in range(4):
            sc.key("R", t, HandPose(str(rng.choice(["relaxed", "open", "point"])),
                                    x=rng.uniform(.3, .7), y=rng.uniform(.35, .7), roll=roll))
            t += rng.uniform(0.3, 0.5)
    frames = []
    for t, dets in sc.frames():
        frames.append({"t": t, "hands": [{"image": d.image.tolist(), "world": d.world.tolist(),
                                          "label": d.label, "score": d.score} for d in dets]})
    return Clip(label, sc.aspect, frames, {"session": f"synthetic-{session}"})


def make_clips(per_label: int = 20, sessions: int = 4, seed: int = 0) -> dict[str, list[Clip]]:
    rng = np.random.default_rng(seed)
    out: dict[str, list[Clip]] = {lab: [] for lab in LABELS}
    for lab in LABELS:
        for i in range(per_label):
            out[lab].append(make_clip(lab, rng, i % sessions))
    return out
