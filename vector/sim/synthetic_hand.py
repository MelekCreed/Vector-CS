"""Kinematic synthetic hand producing MediaPipe-layout landmarks.

Used by the test-suite and benchmarks to drive the full gesture pipeline
with scripted, reproducible hand motion (including noise and dropouts)
without a webcam. Geometry follows MediaPipe conventions for a *mirrored*
image: x right, y down, z away from the camera; world units are metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from vector.vision.tracker import Detection

# Right hand, palm facing the camera, fingers up. Thumb on the image-left.
_MCP = {
    "index": (-0.026, -0.085), "middle": (-0.006, -0.090),
    "ring": (0.014, -0.086), "pinky": (0.032, -0.076),
}
_SEG = {
    "index": (0.040, 0.025, 0.020), "middle": (0.045, 0.028, 0.021),
    "ring": (0.042, 0.026, 0.020), "pinky": (0.032, 0.020, 0.018),
}
_SPLAY = {"index": -0.10, "middle": -0.02, "ring": 0.06, "pinky": 0.14}
_THUMB_CMC = np.array([-0.028, -0.022, -0.005])
_THUMB_SEG = (0.035, 0.030, 0.026)

POSES: dict[str, dict[str, float]] = {
    # curl per finger: 0 straight .. 1 fully curled
    "open":   {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0},
    "fist":   {"thumb": 1.0, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
    "point":  {"thumb": 1.0, "index": 0.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
    "two":    {"thumb": 1.0, "index": 0.0, "middle": 0.0, "ring": 1.0, "pinky": 1.0},
    "three":  {"thumb": 1.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 1.0},
    "shaka":  {"thumb": 0.0, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 0.0},
    "relaxed": {"thumb": 0.35, "index": 0.35, "middle": 0.4, "ring": 0.45, "pinky": 0.5},
    # pinch variants are built specially (thumb tip meets a fingertip)
    "pinch":  {"thumb": 0.0, "index": 0.35, "middle": 0.15, "ring": 0.2, "pinky": 0.25},
    "middle_pinch": {"thumb": 0.0, "index": 0.1, "middle": 0.4, "ring": 0.25, "pinky": 0.3},
}


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _finger(name: str, curl: float) -> list[np.ndarray]:
    bx, by = _MCP[name]
    p = np.array([bx, by, 0.0])
    pts = [p.copy()]
    splay = _SPLAY[name]
    angles = (curl * math.radians(75), curl * math.radians(100), curl * math.radians(70))
    theta = 0.0
    for seg, ang in zip(_SEG[name], angles):
        theta += ang
        # Direction: up (-y), splayed in x, bending toward the camera (-z) = palm side.
        d = np.array([math.sin(splay), -math.cos(theta), -math.sin(theta)])
        d /= np.linalg.norm(d)
        p = p + seg * d
        pts.append(p.copy())
    return pts


def _thumb(curl: float, target: np.ndarray | None = None) -> list[np.ndarray]:
    cmc = _THUMB_CMC.copy()
    if target is not None:
        # Chain from CMC to the target tip with a gentle outward bow.
        pts = [cmc]
        for k, frac in enumerate((0.38, 0.70, 1.0), start=1):
            q = cmc + (target - cmc) * frac
            bow = math.sin(frac * math.pi) * 0.012
            q = q + np.array([-bow, 0.0, -bow * 0.5])
            pts.append(q)
        pts[-1] = target.copy()
        return pts
    open_dir = np.array([-0.62, -0.72, -0.30])
    tucked_tip = np.array([0.0, -0.045, -0.035])     # folded across the palm
    pts = [cmc]
    p = cmc
    d0 = open_dir / np.linalg.norm(open_dir)
    for i, seg in enumerate(_THUMB_SEG):
        tgt_dir = tucked_tip - p
        tgt_dir /= np.linalg.norm(tgt_dir) + 1e-9
        d = (1 - curl) * d0 + curl * tgt_dir
        d /= np.linalg.norm(d)
        p = p + seg * (1 - 0.35 * curl) * d
        pts.append(p.copy())
    return pts


@dataclass
class HandPose:
    pose: str = "open"
    x: float = 0.5                    # palm centre, normalised image coords
    y: float = 0.5
    scale: float = 2.6                # image-height units per metre (distance to camera)
    roll: float = 0.0                 # radians, image-plane rotation
    yaw: float = 0.0                  # radians, turn about vertical axis (palm away at pi)
    pitch: float = 0.0
    label: str = "Right"
    curl_override: dict[str, float] = field(default_factory=dict)
    pinch_gap: float = 0.0            # metres between thumb tip and target fingertip


def build_world(hp: HandPose) -> np.ndarray:
    curls = dict(POSES[hp.pose if hp.pose in POSES else "open"])
    curls.update(hp.curl_override)
    pts = np.zeros((21, 3))
    pts[0] = (0.0, 0.0, 0.0)
    fingers = {n: _finger(n, curls[n]) for n in ("index", "middle", "ring", "pinky")}
    base = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
    for n, chain in fingers.items():
        pts[base[n]:base[n] + 4] = chain
    if hp.pose in ("pinch", "middle_pinch"):
        tip = fingers["index" if hp.pose == "pinch" else "middle"][-1]
        gap = np.array([-hp.pinch_gap * 0.7, hp.pinch_gap * 0.7, 0.0])
        pts[1:5] = _thumb(0.0, target=tip + gap)
    else:
        pts[1:5] = _thumb(curls["thumb"])
    # Centre on the wrist (MediaPipe world landmarks are hand-centred).
    rot = _rot_z(hp.roll) @ _rot_y(hp.yaw) @ _rot_x(hp.pitch)
    pts = pts @ rot.T
    if hp.label == "Left":
        pts[:, 0] *= -1
    return pts


def build_detection(hp: HandPose, aspect: float = 4 / 3, noise: float = 0.0,
                    rng: np.random.Generator | None = None, score: float = 0.97) -> Detection:
    world = build_world(hp)
    palm_idx = [0, 5, 9, 13, 17]
    centre = world[palm_idx].mean(axis=0)
    img = np.zeros((21, 3))
    iso_x = hp.x * aspect + (world[:, 0] - centre[0]) * hp.scale
    img[:, 0] = iso_x / aspect
    img[:, 1] = hp.y + (world[:, 1] - centre[1]) * hp.scale
    img[:, 2] = world[:, 2] * hp.scale
    if noise and rng is not None:
        img[:, :2] += rng.normal(0, noise, (21, 2))
        world = world + rng.normal(0, noise * 0.03, world.shape)
    return Detection(image=img, world=world, label=hp.label, score=score)
