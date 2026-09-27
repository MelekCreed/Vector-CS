"""Benchmark: deterministic temporal engine vs learned temporal models.

Both are scored on the *same* clips:
  * deterministic — replay each clip through the full engine; correct if the
    expected event fires (and, for 'none' clips, if no command fires)
  * learned (GRU / TCN) — leave-one-session-out cross-validation, so a model
    is always tested on a recording session it never trained on

    python scripts/benchmark_learned.py                 # your recordings (data/gestures)
    python scripts/benchmark_learned.py --synthetic     # pipeline smoke test only

Adoption rule (from the project brief): only switch a gesture to the learned
model if it beats the deterministic engine on held-out sessions.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vector.config import Config  # noqa: E402
from vector.core.geometry import Rect  # noqa: E402
from vector.desktop.monitors import Monitor, VirtualDesktop  # noqa: E402
from vector.learned.dataset import expected_engine_event, load_dataset  # noqa: E402
from vector.recording import load_clip, replay  # noqa: E402

DESK = VirtualDesktop([Monitor("M", Rect(0, 0, 1920, 1080), Rect(0, 0, 1920, 1040), primary=True)])
COMMANDS = {"click", "throw", "swipe", "hold", "right_click", "volume_step"}


def deterministic_scores(clips: dict) -> dict[str, tuple[int, int]]:
    cfg = Config()
    out = {}
    for label, items in clips.items():
        exp = expected_engine_event(label)
        if exp is None and label not in ("none", "idle", "background"):
            continue                          # custom gesture: engine has no opinion
        ok = 0
        for item in items:
            clip = item if not isinstance(item, Path) else load_clip(item)
            events = [e for e in replay(clip, cfg, DESK) if e.kind in COMMANDS
                      and e.confidence.value >= 0.7]
            if exp is None:
                ok += not events
            else:
                kind, data = exp
                hits = [e for e in events if e.kind == kind
                        and all(e.data.get(k) == v for k, v in data.items())]
                ok += bool(hits) and len(events) == len(hits)
        out[label] = (ok, len(items))
    return out


def learned_scores(ds, kind: str, epochs: int):
    from vector.learned.model import predict, train
    groups = np.unique(ds.sessions)
    if len(groups) < 2:
        folds = [(np.arange(len(ds.y)) % 5 != k, np.arange(len(ds.y)) % 5 == k) for k in range(5)]
        split = "5-fold (only one session recorded)"
    else:
        folds = [(ds.sessions != g, ds.sessions == g) for g in groups]
        split = f"leave-one-session-out ({len(groups)} sessions)"
    correct = np.zeros(len(ds.labels))
    total = np.zeros(len(ds.labels))
    infer_ms = []
    for tr, te in folds:
        if te.sum() == 0 or len(np.unique(ds.y[tr])) < 2:
            continue
        m = train(kind, ds.X[tr], ds.y[tr], len(ds.labels), epochs=epochs)
        t0 = time.perf_counter()
        pred = predict(m, ds.X[te]).argmax(1)
        infer_ms.append((time.perf_counter() - t0) * 1000 / te.sum())
        for yi, pi in zip(ds.y[te], pred):
            total[yi] += 1
            correct[yi] += yi == pi
    return {lab: (int(c), int(n)) for lab, c, n in zip(ds.labels, correct, total)}, split, \
        float(np.mean(infer_ms)) if infer_ms else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="use generated clips (smoke test)")
    ap.add_argument("--data", default="data/gestures")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--models", default="gru,tcn")
    args = ap.parse_args()

    if args.synthetic:
        from vector.sim.synth_clips import make_clips
        clips = make_clips()
        print("SYNTHETIC DATA — validates the pipeline, says nothing about real-world accuracy\n")
    else:
        from vector.recording import list_clips
        clips = list_clips(Path(args.data))
        if not clips:
            print(f"no clips in {args.data}. Record some (debug window / Ctrl+Alt+R), "
                  "or run with --synthetic for a smoke test.")
            return 1
    ds = load_dataset(clips=clips)
    print(f"{len(ds.y)} clips, {len(ds.labels)} labels, sequence {ds.X.shape[1]}x{ds.X.shape[2]}")

    det = deterministic_scores(clips)
    results = {"deterministic": det}
    splits = {}
    for kind in [k for k in args.models.split(",") if k]:
        try:
            res, split, ms = learned_scores(ds, kind, args.epochs)
        except SystemExit as exc:
            print(exc)
            break
        results[kind] = res
        splits[kind] = (split, ms)

    names = list(results)
    print(f"\n{'label':<26}" + "".join(f"{n:>16}" for n in names))
    for lab in ds.labels:
        row = f"{lab:<26}"
        for n in names:
            c, t = results[n].get(lab, (0, 0))
            row += f"{(f'{c}/{t} ({c / t:.0%})' if t else 'n/a'):>16}"
        print(row)
    common = [lab for lab in ds.labels if lab in det]
    print()
    for n in names:
        c = sum(results[n].get(lab, (0, 0))[0] for lab in common)
        t = sum(results[n].get(lab, (0, 0))[1] for lab in common)
        extra = f"  [{splits[n][0]}, {splits[n][1]:.2f} ms/seq]" if n in splits else ""
        print(f"{n:<14} accuracy on engine-covered labels: {c}/{t} = {c / max(t, 1):.1%}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
