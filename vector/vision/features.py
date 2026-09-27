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
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))  # hand-lengths / s
    speed: float = 0.0


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * 0


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _unit(a), _unit(b)
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def finger_extension(world: np.ndarray, finger: str) -> float:
    """Straightness from the bend angles along the finger chain.
    0 rad of total bend = fully straight (1.0); ~pi of bend = fully curled (0.0)."""
    j = L.FINGERS[finger]
    if finger == "thumb":
        chain = [L.WRIST if False else j[0], j[1], j[2], j[3]]
        bends = [_angle(world[chain[i + 1]] - world[chain[i]], world[chain[i + 2]] - world[chain[i + 1]])
                 for i in range(2)]
        straight = 1.0 - np.clip(sum(bends) / 1.6, 0.0, 1.0)
        # A straight thumb can still be tucked across the palm; require it to point away.
        palm_w = np.linalg.norm(world[L.INDEX_MCP] - world[L.PINKY_MCP]) + 1e-9
        away = np.linalg.norm(world[L.THUMB_TIP] - world[L.INDEX_MCP]) / palm_w
        return float(straight * np.clip((away - 0.55) / 0.5, 0.0, 1.0))
    chain = [L.WRIST, j[0], j[1], j[2], j[3]]
    bends = [_angle(world[chain[i + 1]] - world[chain[i]], world[chain[i + 2]] - world[chain[i + 1]])
             for i in range(3)]
    # MCP bend counts less: fingers are often slightly bent at the knuckle while extended.
    total = 0.5 * bends[0] + bends[1] + bends[2]
    return float(1.0 - np.clip((total - 0.35) / 2.0, 0.0, 1.0))


def compute(obs: HandObservation) -> HandFeatures:
    img, world = obs.image, obs.world
    iso = np.column_stack([img[:, 0] * obs.aspect, img[:, 1]])

    palm_len_w = float(np.linalg.norm(world[L.MIDDLE_MCP] - world[L.WRIST])) or NOMINAL_PALM_M
    # Pixels-per-metre from the *least* foreshortened palm segment: segments
    # tilted away from the camera look short in the image but not in world space.
    segs = [(L.WRIST, L.INDEX_MCP), (L.WRIST, L.MIDDLE_MCP), (L.WRIST, L.PINKY_MCP),
            (L.INDEX_MCP, L.PINKY_MCP)]
    ratios = []
    for a, b in segs:
        wl = np.linalg.norm(world[a] - world[b])
        if wl > 1e-4:
            ratios.append(np.linalg.norm(iso[a] - iso[b]) / wl)
    hand_scale = float(max(ratios) * palm_len_w) if ratios else 0.2

    ext = {f: finger_extension(world, f) for f in L.FINGER_NAMES}
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

    dirs = [_unit(world[L.FINGERS[f][3]] - world[L.FINGERS[f][0]]) for f in ("index", "middle", "ring", "pinky")]
    spread = float(np.mean([_angle(dirs[i], dirs[i + 1]) for i in range(3)]))

    palm_center = iso[list(L.PALM)].mean(axis=0)
    return HandFeatures(
        track_id=obs.track_id, label=obs.label, t=obs.t, presence=obs.label_score, iso=iso,
        palm_center=palm_center, index_tip=iso[L.INDEX_TIP].copy(),
        pinch_point=(iso[L.THUMB_TIP] + iso[L.INDEX_TIP]) / 2,
        hand_scale=hand_scale, extension=ext, pinch_ratio=pinch, middle_pinch_ratio=mpinch,
        palm_normal=n, palm_facing=facing, upright=upright, roll=roll, finger_spread=spread,
        stale=obs.stale)
