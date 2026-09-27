# Handoff 001 — Vector architecture critique

## Objective
Vector: control the REAL Windows desktop with webcam hand gestures (point/cursor, pinch-click,
pinch-grab-drag windows, throw-to-snap / throw-to-monitor, two-hand resize, palm swipe app switch,
two-finger scroll, media controls, air drawing overlay, sleep/active gating, failsafe, calibration,
debug HUD, gesture recording, optional learned model). Must feel like a polished trackpad and
must never fire accidental commands.

## Constraints / measured environment
- Windows 11, i7-1355U, Iris Xe, 1 monitor 1920x1080 @125% scaling, integrated webcam.
- Webcam capped at 30 fps (MSMF 640x480 = 30.5fps; DSHOW 720p = 10fps).
- Python 3.13 venv, mediapipe 1.0.1 (Tasks API only; mp.solutions removed), HandLandmarker VIDEO
  mode ~17-23 ms/frame for 2 hands on CPU. Measured: inference RELEASES the GIL (a 240Hz ticker
  thread saw max 10ms gaps while inference ran).
- PySide6 6.11 for overlay/debug UI; pywin32 + ctypes for Win32.

## Decisions (proposed)
1. Threads, not processes: CameraThread (latest-frame, drop stale) -> VisionThread (flip, infer,
   features, temporal gesture engine, intent) -> command queue -> ActuatorThread (120 Hz loop:
   critically-damped spring interpolation of cursor + dragged-window rect between 30 Hz samples;
   executes discrete commands; SetWindowPos with SWP_ASYNCWINDOWPOS) ; Qt main thread renders
   overlay at 60 Hz reading snapshots; LL keyboard hook thread for ESC-ESC failsafe + hotkeys.
2. Process is Per-Monitor-DPI-Aware v2; all Win32 math in physical virtual-desktop pixels. One
   overlay window per monitor (click-through: WS_EX_LAYERED|TRANSPARENT|NOACTIVATE), converting
   physical -> that screen's logical coords by its devicePixelRatio.
3. Features from MediaPipe WORLD landmarks (metric, scale-invariant): finger curl from joint
   angles, pinch ratio = thumb-index tip distance / palm size, palm normal, roll; velocities
   normalized by hand size in image (hand-lengths/s) so thresholds are distance-independent.
   Frame is mirrored before inference so handedness labels match the user; persistent hand IDs by
   nearest-neighbour + sticky handedness voting.
4. Pose layer: continuous per-frame scores (POINT, PINCH, MIDDLE_PINCH, OPEN_PALM, FIST, V,
   THREE, SHAKA) -> temporal layer: EMA-smoothed scores, enter/exit hysteresis, confirm frames.
5. Per-hand state machine: IDLE, HOVERING, POINTING, PINCH_STARTED, GRABBING, DRAGGING, THROWING,
   SCROLLING, DRAWING, SWIPING(+cooldown) ; two-hand RESIZING. Emits events with confidence =
   weighted geometric mean of (pose geometry, stability, duration, velocity margin, tracking
   presence, target context). Intent engine maps events -> commands via configurable bindings,
   gated by ACTIVE/SLEEPING/DISABLED and per-action thresholds + cooldowns.
6. Pinch: target lock uses cursor position from ~100 ms BEFORE pinch onset (index tip moves while
   closing the pinch); during pinch the drag follows the palm centre (stable under finger
   articulation). Quick pinch w/o movement = left click; hold or move > threshold = grab/drag.
   Middle-finger pinch = right click.
7. Cursor: One-Euro filter -> hybrid mapping (relative deltas with speed-dependent gain curve +
   slow drift correction toward the absolute calibrated mapping so hand<->screen stays anchored)
   -> dead zone -> 120 Hz spring in actuator.
8. Throw: on pinch release with speed > threshold: project release point along velocity for
   ~0.25s; if it lands on another monitor -> move there (keep relative rect / maximized state);
   else dominant direction: left/right = snap half of work area (using DWM extended frame bounds
   to compensate invisible borders), up = maximize, down = minimize.
9. Sleep/wake: SLEEPING->ACTIVE: single upright open palm facing camera held still 0.6s.
   ACTIVE->SLEEPING: two open palms held 0.8s. ESC ESC (LL hook, ignores injected keys) =
   DISABLED (sticky; re-arm with Ctrl+Alt+V), releases held buttons, cancels grabs.
10. Window targeting: EnumWindows z-order walk (skip own overlays, cloaked, tool, shell classes,
   blocklist), not WindowFromPoint. Foreground via SetForegroundWindow (+Alt-key trick fallback).
11. Testing: pure-logic modules with injected timestamps; synthetic kinematic hand generator for
   pose/state-machine sequence tests; replay of recorded landmark sessions; fake monitor layouts
   for multi-monitor/DPI math; real Win32 tests against a spawned test window.
12. Learned model: record landmark sequences (JSONL/npz) -> optional PyTorch GRU/TCN, benchmark vs
   deterministic engine on the same replay set; only adopt if better.

## Uncertainties / risks
- Qt translucent full-screen raster overlay repaint cost at 60 Hz on Iris Xe.
- SetForegroundWindow restrictions; UIPI blocking elevated windows.
- Pinch detection robustness when the hand is edge-on to the camera.
- Handedness flicker; two-hand resize when hands cross.

## Response wanted
Critique: risks, recommendation, alternatives considered, confidence level. Point out concrete
failure modes in the gesture/intent design (false positives especially) and Win32/DPI pitfalls.
Be concise (<= 700 words). Do NOT modify any files.
