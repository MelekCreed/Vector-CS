"""Landmark processor: turns 21 raw landmarks into scale-invariant features.

Two coordinate systems are used deliberately:
* **world** landmarks (metric, hand-centred) for *shape*: finger curl, pinch
  distance, palm orientation. These don't change with distance to the camera.
* **image** landmarks (x scaled by aspect ratio so units are isotropic) for
  *position and motion*: cursor, velocity, hand-to-hand distance.
Velocities are expressed in *hand-lengths per second* so gesture thresholds
work the same whether you sit 40 cm or 1.5 m from the camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vector.vision import landmarks as L

NOMINAL_PALM_M = 0.09   # wrist -> middle MCP of an adult hand, metres


@dataclass
class HandObservation:
    """One hand in one frame, straight out of the tracker."""
    track_id: int
    label: str                      # "Left" | "Right" (user's actual hand)
    label_score: float
    image: np.ndarray               # (21,3) normalised image coords (x,y in [0,1], z rel.)
    world: np.ndarray               # (21,3) metres
    t: float
    aspect: float                   # width / height of the source image
    stale: bool = False             # carried through a brief tracking dropout
    label_confidence: float = 1.0   # how settled the handedness vote is (0..1)
    age_s: float = 0.0              # how long this identity has been tracked


@dataclass
class HandFeatures:
    track_id: int
    label: str
    t: float
    presence: float
    iso: np.ndarray                 # (21,2) isotropic image coords
    palm_center: np.ndarray         # (2,) iso
    index_tip: np.ndarray           # (2,) iso
    pinch_point: np.ndarray         # (2,) iso, midpoint of thumb+index tips
    hand_scale: float               # palm length in iso units (foreshortening-robust)
    extension: dict[str, float]     # 0 = curled .. 1 = straight
    pinch_ratio: float              # thumb-index tip distance / palm length (world)
    middle_pinch_ratio: float
    palm_normal: np.ndarray         # (3,) unit, world
    palm_facing: float              # +1 palm to camera, -1 back of hand to camera
    upright: float                  # +1 fingers up, -1 fingers down
    roll: float                     # radians, rotation of the palm in the image plane
    finger_spread: float            # mean angle between adjacent fingers (radians)
    stale: bool = False
    age_s: float = 0.0
    aspect: float = 4 / 3           # image width / height (iso x = norm x * aspect)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))  # hand-lengths / s
    speed: float = 0.0


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * 0


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


# Joint chains for the batched bend computation (wrist -> ... -> tip).
_LONG_CHAINS = np.array([[L.WRIST, *L.FINGERS[f]] for f in ("index", "middle", "ring", "pinky")])
_THUMB_CHAIN = np.array(L.FINGERS["thumb"])


def _bends(world: np.ndarray, chains: np.ndarray) -> np.ndarray:
    """Angle (rad) between consecutive segments along each chain, batched."""
    pts = world[chains]                          # (n, k, 3)
    seg = np.diff(pts, axis=1)                   # (n, k-1, 3)
    seg /= np.linalg.norm(seg, axis=2, keepdims=True) + 1e-9
    cos = np.einsum("nki,nki->nk", seg[:, :-1], seg[:, 1:])
    return np.arccos(np.clip(cos, -1.0, 1.0))    # (n, k-2)


def finger_extensions(world: np.ndarray) -> dict[str, float]:
    """Straightness from bend angles along each finger chain: 0 rad of total
    bend = fully straight (1.0); ~pi of bend = fully curled (0.0)."""
    b = _bends(world, _LONG_CHAINS)              # (4, 3): mcp, pip, dip
    # MCP bend counts less: fingers are often slightly bent at the knuckle while extended.
    total = 0.5 * b[:, 0] + b[:, 1] + b[:, 2]
    ext = np.clip(1.0 - (total - 0.35) / 2.0, 0.0, 1.0)
    out = dict(zip(("index", "middle", "ring", "pinky"), map(float, ext)))

    tb = _bends(world, _THUMB_CHAIN[None])[0]
    straight = _clip01(1.0 - float(tb.sum()) / 1.6)
    # A straight thumb can still be folded across the palm, so also require
    # the tip to be far from the pinky knuckle (tucked ~1x palm width, out ~2x).
    palm_w = float(np.linalg.norm(world[L.INDEX_MCP] - world[L.PINKY_MCP])) + 1e-9
    away = float(np.linalg.norm(world[L.THUMB_TIP] - world[L.PINKY_MCP])) / palm_w
    out["thumb"] = straight ** 0.5 * _clip01((away - 1.15) / 0.6)
    return out


def finger_extension(world: np.ndarray, finger: str) -> float:
    return finger_extensions(world)[finger]


def compute(obs: HandObservation) -> HandFeatures:
    img, world = obs.image, obs.world
    iso = np.column_stack([img[:, 0] * obs.aspect, img[:, 1]])

    palm_len_w = float(np.linalg.norm(world[L.MIDDLE_MCP] - world[L.WRIST])) or NOMINAL_PALM_M
    # Pixels-per-metre from the *least* foreshortened palm segment: segments
    # tilted away from the camera look short in the image but not in world space.
    a_idx, b_idx = [0, 0, 0, 5], [5, 9, 17, 17]
    wl = np.linalg.norm(world[a_idx] - world[b_idx], axis=1)
    il = np.linalg.norm(iso[a_idx] - iso[b_idx], axis=1)
    ok = wl > 1e-4
    hand_scale = float((il[ok] / wl[ok]).max() * palm_len_w) if ok.any() else 0.2

    ext = finger_extensions(world)
    pinch = float(np.linalg.norm(world[L.THUMB_TIP] - world[L.INDEX_TIP]) / palm_len_w)
    mpinch = float(np.linalg.norm(world[L.THUMB_TIP] - world[L.MIDDLE_TIP]) / palm_len_w)

    n = _unit(np.cross(world[L.INDEX_MCP] - world[L.WRIST], world[L.PINKY_MCP] - world[L.WRIST]))
    # With a mirrored image, x right / y down / z away from camera: for a right
    # hand showing its palm, index->pinky is left->right so the cross product
    # points away from the camera (+z). The left hand is the mirror image.
    facing = float(n[2] if obs.label == "Right" else -n[2])

    up = _unit(iso[L.MIDDLE_MCP] - iso[L.WRIST])
    upright = float(-up[1])
    across = iso[L.PINKY_MCP] - iso[L.INDEX_MCP]
    roll = float(np.arctan2(across[1], across[0]))

    dirs = world[[8, 12, 16, 20]] - world[[5, 9, 13, 17]]
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
    spread = float(np.arccos(np.clip(np.einsum("ij,ij->i", dirs[:-1], dirs[1:]), -1, 1)).mean())

    palm_center = iso[list(L.PALM)].mean(axis=0)
    return HandFeatures(
        track_id=obs.track_id, label=obs.label, t=obs.t, presence=obs.label_score, iso=iso,
        palm_center=palm_center, index_tip=iso[L.INDEX_TIP].copy(),
        pinch_point=(iso[L.THUMB_TIP] + iso[L.INDEX_TIP]) / 2,
        hand_scale=hand_scale, extension=ext, pinch_ratio=pinch, middle_pinch_ratio=mpinch,
        palm_normal=n, palm_facing=facing, upright=upright, roll=roll, finger_spread=spread,
        stale=obs.stale, age_s=obs.age_s, aspect=obs.aspect)
