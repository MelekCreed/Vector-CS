# Decisions log

## 2026-09-27 — Architecture (handoff 001; Claude proposed, Codex critiqued)
Writer: Claude Code. Consultant: Codex CLI (fresh `codex exec` session, read-only).

Adopted from Codex critique:
- Commands carry interaction id, activation generation, capture time and expiry; the actuator
  drops stale/cancelled commands. Failsafe/sleep bumps the generation. Motion = latest-value
  mailbox; discrete actions = bounded queue. Vision watchdog cancels drags if tracking stalls.
- One global interaction arbiter owns the hands; modes are exclusive with explicit precedence.
  Wake requires a neutral pose before commands become eligible. Confidence is a soft score;
  tracking / identity / target checks are hard gates.
- Click only on observed pinch release (never on tracking loss). Index/middle pinch exclusive.
  Target lock requires hover stability; drag deltas rebased at acquisition.
- Throw requires an established drag, sustained velocity, min displacement, directional
  consistency and an observed release. Destination is previewed while dragging.
- Durations are measured in elapsed time, not frame counts.
- Hybrid cursor drift-correction only while moving and never while engaged (pinched).
- Overlay: one window per monitor; physical -> local = (p - monitor_origin_physical) / DPR.
- window_at stops at occluding protected windows (skips only own/invisible/cloaked/click-through).
- Full ctypes prototypes; use a single correct INPUT union. Foreground: PowerToys-style empty
  mouse SendInput before SetForegroundWindow (no Alt keystroke — it can open menus).
- Negative-behaviour replay tests (waving, face touch, dropouts, crossings) are first-class.

Partial disagreement (resolved by the user's spec, not synthesised):
- Codex: disable throw-down-to-minimize initially. The spec explicitly requires it, so it stays
  enabled but with a stricter speed threshold (`throw_down_speed_factor`) and is configurable.
