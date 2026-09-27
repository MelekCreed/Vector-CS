"""Download the MediaPipe hand landmarker model (not committed to the repo)."""

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
       "hand_landmarker/float16/latest/hand_landmarker.task")
DEST = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


def main() -> int:
    if DEST.exists() and DEST.stat().st_size > 1_000_000:
        print(f"model already present: {DEST}")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}")
    tmp = DEST.with_suffix(".part")
    urllib.request.urlretrieve(URL, tmp)
    tmp.replace(DEST)
    print(f"saved {DEST} ({DEST.stat().st_size / 1e6:.1f} MB, sha256 "
          f"{hashlib.sha256(DEST.read_bytes()).hexdigest()[:16]}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
