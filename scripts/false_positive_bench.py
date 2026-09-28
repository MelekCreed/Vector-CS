"""Measure accidental-command rate on randomised 'everyday' hand motion.

Random walks through relaxed / open / point / fist / V poses at moderate and
fast speeds, with landmark noise and dropouts. Counts commands that would
pass the configured confidence thresholds. Usage:
    python scripts/false_positive_bench.py [--seeds 20]
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vector.config import Config  # noqa: E402
from vector.core.geometry import Rect  # noqa: E402
from vector.desktop.monitors import Monitor, VirtualDesktop  # noqa: E402
from vector.intent.actions import threshold_class_for  # noqa: E402
from vector.pipeline import GesturePipeline  # noqa: E402
from vector.sim.scenario import Scenario, run  # noqa: E402
from vector.sim.synthetic_hand import HandPose  # noqa: E402

COMMANDS = {"click", "right_click", "throw", "swipe", "hold", "volume_step", "drag_start",
            "pinch_start", "resize_start", "wake", "sleep", "carousel_select"}
DESK = VirtualDesktop([Monitor("M", Rect(0, 0, 1920, 1080), Rect(0, 0, 1920, 1040), primary=True)])


def everyday_scenario(seed: int, step_range=(0.3, 0.6), keys=50,
                      poses=("relaxed", "relaxed", "open", "relaxed", "point", "fist", "two")):
    rng = np.random.default_rng(seed)
    sc = Scenario(noise=0.003, drop_rate=0.05, seed=seed)
    t = 0.0
    for _ in range(keys):
        sc.key("R", t, HandPose(pose=str(rng.choice(poses)), x=rng.uniform(.25, .75),
                                y=rng.uniform(.2, .8), roll=rng.uniform(-.6, .6)))
        t += rng.uniform(*step_range)
    return sc


def accidental(events, cfg: Config):
    out = []
    for e in events:
        if e.kind not in COMMANDS:
            continue
        cls = threshold_class_for(e, cfg)
        if e.confidence.value >= cfg.threshold(cls):
            out.append(e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    cfg = Config()
    total_s, counts = 0.0, collections.Counter()
    t0 = time.perf_counter()
    for seed in range(args.seeds):
        for steps in ((0.3, 0.6), (0.15, 0.35)):
            sc = everyday_scenario(seed, steps)
            pipe = GesturePipeline(cfg, DESK)
            pipe.engine.system = pipe.engine.system.ACTIVE
            ev, _ = run(pipe, sc)
            total_s += sc.duration
            for e in accidental(ev, cfg):
                counts[f"{e.kind}:{e.data.get('family') or e.data.get('pose') or ''}"] += 1
    n = sum(counts.values())
    print(f"simulated {total_s / 60:.1f} min of everyday motion "
          f"({time.perf_counter() - t0:.1f}s wall)")
    print(f"accidental commands above threshold: {n} "
          f"({n / (total_s / 3600):.1f}/hour) {dict(counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
