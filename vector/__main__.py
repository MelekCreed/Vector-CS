"""Command line entry point.

    python -m vector                 run (calibrates on first launch)
    python -m vector --debug         run with the developer window
    python -m vector --demo          run with demo captions for recording videos
    python -m vector calibrate       force recalibration
    python -m vector replay CLIP     run a recorded clip through the engine, print events
    python -m vector clips           list recorded gesture clips
    python -m vector config          print the effective config and its path
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vector", description="Hand-gesture control for Windows")
    ap.add_argument("command", nargs="?", default="run",
                    choices=["run", "calibrate", "replay", "clips", "config"])
    ap.add_argument("clip", nargs="?", help="clip path for 'replay'")
    ap.add_argument("--config", help="user config JSON (default %%APPDATA%%/Vector/user_config.json)")
    ap.add_argument("--debug", action="store_true", help="open the developer window")
    ap.add_argument("--demo", action="store_true", help="show demo-sequence captions")
    ap.add_argument("--active", action="store_true", help="start ACTIVE (skip the wake gesture)")
    ap.add_argument("--quit-after", type=float, default=0.0, help="exit after N seconds (smoke tests)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname).1s %(name)s: %(message)s", datefmt="%H:%M:%S")
    from vector.app import load_config
    cfg, cfg_path = load_config(args.config)
    if args.debug:
        cfg.debug = True
    if args.active:
        cfg.activation.start_active = True

    if args.command == "config":
        from vector import config as C
        print(f"# {cfg_path} ({'exists' if cfg_path.exists() else 'not created yet'})")
        print(json.dumps(C.to_dict(cfg), indent=2))
        return 0
    if args.command == "clips":
        from vector.recording import list_clips
        clips = list_clips()
        if not clips:
            print("no clips yet — record some from the debug window (Ctrl+Alt+R)")
        for label, paths in clips.items():
            print(f"{label:<24} {len(paths):>4} clips")
        return 0
    if args.command == "replay":
        if not args.clip:
            ap.error("replay needs a clip path")
        from vector.desktop.dpi import make_process_dpi_aware
        from vector.desktop.monitors import VirtualDesktop
        from vector.recording import load_clip, replay
        make_process_dpi_aware()
        clip = load_clip(Path(args.clip))
        events = replay(clip, cfg, VirtualDesktop.from_system())
        print(f"clip '{clip.label}': {len(clip.frames)} frames")
        for e in events:
            if e.kind not in ("stroke_point", "erase_point"):
                print(f"  {e.t:7.3f}s  {e.kind:<14} c={e.confidence.value:.2f}  "
                      f"{ {k: v for k, v in e.data.items() if not hasattr(v, 'shape')} }")
        return 0

    from vector.app import run
    return run(cfg, cfg_path, calibrate=args.command == "calibrate", demo=args.demo,
               debug_window=args.debug, quit_after=args.quit_after)


if __name__ == "__main__":
    sys.exit(main())
