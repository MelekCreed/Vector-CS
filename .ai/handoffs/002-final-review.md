# Handoff 002 — Final review of Vector v0.1

## Objective
Final pre-release review. Writer: Claude Code. Reviewer: Codex (read-only, fresh session).

## Constraints
Real desktop control: correctness of safety gates matters more than style. No destructive actions.

## Scope
Everything since the foundation commit (review-base), plus foundation Win32 files
vector/desktop/windows.py, vector/desktop/input.py, vector/desktop/monitors.py.

## Decisions already made (see .ai/decisions.md)
Your handoff-001 critique was adopted (generation-stamped commands, interaction owner,
observed-release clicks, occluder-aware hit test, PowerToys-style foregrounding, etc.).

## Tests and results
129 tests pass (unit, synthetic-hand scenario, real Win32 against a spawned window).
False-positive bench: 1 accidental command / 11.5 simulated minutes. App smoke-run OK.

## Focus (in priority order)
1. Safety invariants: can any path execute an OS action while SLEEPING/DISABLED, after
   Esc Esc, or from stale data? (runtime.py VisionWorker arm/disarm sync, actuator.py
   generation/expiry/watchdog, safety.py hook, intent.py gating)
2. Threading races (vision thread vs Qt thread vs hook thread vs actuator), canvas lock.
3. Gesture engine state-machine bugs (engine.py): stuck states, interaction never released,
   resize/drag rebase, draw mode, anchor handling.
4. Win32 pitfalls (ctypes prototypes, DPI, SetWindowPos async assumptions).
5. Anything claimed in README.md that the code does not do.

## Response wanted
Concrete findings with file:line, severity (high/medium/low), failure scenario, and a
suggested fix. Then: risks, recommendation, alternatives considered, confidence level.
Do not modify files.
