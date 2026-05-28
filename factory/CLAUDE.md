# Factory operating contract

This directory is a **local, scenario-first work factory**. Read this file before doing anything
in it. Everything below is non-negotiable.

## Core rules

1. **Scenarios, not tests.** Correctness is judged by observable behavior against the plain-language
   "satisfied when" of each scenario. The judge exercises the real program. Tests are gameable
   (`return true` passes a narrow test); behavior is not.

2. **The holdout is sealed.** `factory/scenarios/holdout/` is the only thing that decides "done".
   You — the builder — MUST NEVER read, list, write, copy, or otherwise observe files under it.
   A guardrail hook will block such attempts; do not try to work around it. The judge (a separate
   process with its own permissions) reads holdout, not you.

3. **`.factory/` is off-limits.** `factory/.factory/` holds the guardrail and loop wiring.
   Do not edit it as part of solving a task.

4. **Never commit without human approval.** The loop stages changes and stops for human review.
   You do not commit, push, force-push, rewrite history, or delete files outside the working tree.

5. **Memory trail.** Before stopping each session, append to `factory/memory/log.md`:
   - what changed,
   - what is still unknown (write `UNKNOWN: ...` rather than guess),
   - what you would do next.
   This is how the next iteration picks up where you left off.

6. **One small step.** Each iteration should be the smallest change that plausibly moves the
   judge score forward. Do not refactor speculatively. Do not add scope. Failing scenarios + judge
   notes are your only feedback signal — follow them.

## Files you may read

- `factory/CLAUDE.md` (this file)
- `factory/scenarios/` (everything EXCEPT `holdout/`)
- `factory/tasks/`
- `factory/memory/`
- The source code being modified

## Files you must not touch

- `factory/scenarios/holdout/**`
- `factory/.factory/**`
- `factory/runs/state.json` (the loop manages it)

## When in doubt

Stop and write `UNKNOWN: <question>` in `factory/memory/log.md`. The human will resolve it.
