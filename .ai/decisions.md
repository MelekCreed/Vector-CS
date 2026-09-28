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

## 2026-09-28 — Implementation decisions (Claude, writer)
- Swipes require pause-then-flick (>=100 ms sustained stillness within 0.35 s before the
  stroke) and the stroke-minimum pose score; measured false swipes 24 -> 1 per ~12 simulated
  minutes of adversarial motion (scripts/false_positive_bench.py).
- While pinched, the pointer is `fingertip_at_onset + (palm - palm_at_onset)`; a decaying
  source-switch offset caused a ~400 px jump at drag start (found via offscreen HUD render,
  guarded by test_pinch_onset_does_not_move_cursor_or_window).
- Emergency stop latches inside the actuator (arm() refuses until explicit unlatch) and
  tick/disarm are serialised: closes a race where the vision thread could re-arm after Esc Esc.
- Learned models (GRU/TCN) are optional; adoption only if they beat the deterministic engine on
  leave-one-session-out real recordings. Synthetic smoke run: engine 120/120, GRU/TCN 118/120,
  so nothing adopted. Real-data benchmark pending the user's recordings.
- Codex `exec review --base` cannot take a custom prompt; the handoff lives in
  .ai/handoffs/002-final-review.md inside the repo for the reviewer to read.

## 2026-09-28 — Final review (handoff 002; Codex reviewed, Claude fixed)
Codex `exec review --base review-base` (fresh read-only session) reported 9 findings; all
accepted, no rebuttal round needed. Fixes + one regression test each
(tests/test_review_regressions.py, verified failing on the pre-fix commit):
- [P1] stale re-arm could undo a newer emergency stop -> stop generation check
- [P1] held sleep palms re-woke the system -> wake blocked until palms drop
- [P2] right click scored first-frame evidence -> emit once evidence matures
- [P2] resize ending without drag kept cursor/target locks -> release event + clear
- [P2] draw-mode eraser blocked sleep -> passive interactions don't block sleep
- [P2] draw toggle dropped teardown events -> queued pending events
- [P2] depth change manufactured travel -> integrate scale-normalised displacement
- [P2] middle_pinch / shaka bindings ignored -> routed through binding resolution
- [P2] spline recomputed under lock every repaint -> vectorised + cached outside lock
