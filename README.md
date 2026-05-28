# local-factory

A small, filesystem-driven build loop for LLM coding agents. Correctness is
judged by **observable program behavior** against sealed *holdout* scenarios,
not by reading source or running brittle tests. The builder agent never sees
the holdout; a separate judge agent does, and scores the program's behavior
on a 0.0–1.0 satisfaction scale. Failures feed back into an append-only
memory log that informs the next iteration.

The loop is intentionally local, intentionally Markdown-driven, and
intentionally **never commits**. A human reviews the staged diff at the end
of each task and decides whether to commit.

## Status and scope

Currently built on [Claude Code](https://docs.claude.com/claude-code)'s
headless `claude -p` mode and its `PreToolUse` / `Stop` hooks. The
scenario/holdout *pattern* is agent-agnostic, and the agent-CLI surface in
`factory/loop.sh` is small (roughly 30 lines of `claude -p` invocations
and JSON-envelope parsing), so a port to a different agent CLI is
tractable — but no such port exists yet. If you arrived expecting a
vendor-neutral tool: today this is Claude Code only.

The seed task in `factory/tasks/greet.md` has been run end-to-end once and
converged in a single iteration. The feedback loop — failures driving the
next iteration through `memory/log.md` — has not yet been observed
empirically, because the seed task was too easy. Consider the project
*alpha*: the bones are real, the load-bearing parts are tested, the
hard-task evidence is still ahead.

## Why

Two problems with the usual "LLM writes the code, tests verify it" loop:

1. **Tests are gameable.** `def test_foo(): assert True` passes. An agent
   optimizing for a green test suite can reach pass without solving the
   problem. Behavior — exit codes, stdout, file contents, side effects — is
   harder to fake than a narrow test.
2. **Visible specs invite overfitting.** If the agent can read every
   criterion in advance, it will hard-code for those criteria. Sealing a
   subset of scenarios from the agent is the only way to know the work
   generalizes beyond what was negotiated in the prompt.

Scenarios are written in plain language ("satisfied when X happens when you
run Y"). A judge LLM, with permission to read the holdout and exercise the
real program, decides how well behavior matches each scenario.

## How it works

```
       ┌──── factory/tasks/<name>.md            (prose task spec)
       │
       │       factory/scenarios/*.md           (visible scenarios)
       │       factory/scenarios/holdout/*.md   (judge-only scenarios)
       │
       ▼
  ┌─────────────┐    edits     ┌────────────────────┐
  │   BUILDER   │─────────────▶│   working tree     │
  │ FACTORY_    │              │  (staged, not      │
  │ PHASE=      │              │   committed)       │
  │ builder     │              └────────────────────┘
  └─────────────┘                        │
       ▲                                 │ VALIDATE_CMD
       │ feedback                        ▼
       │ (failing                ┌────────────────┐
       │  scenarios +            │  real program  │
       │  judge notes)           │   exercised    │
       │                         └────────────────┘
       │                                 │
       │                                 ▼
  ┌─────────────┐                ┌────────────────┐
  │   memory    │◀───────────────│     JUDGE      │
  │   log.md    │  failing,      │ FACTORY_PHASE  │
  │ (append-    │  notes         │     =judge     │
  │  only)      │                │ (may read      │
  └─────────────┘                │  holdout)      │
                                 └────────────────┘
                                         │
                                         ▼
                                 {min_satisfaction,
                                  failing, notes}
                                         │
                       ┌─────────────────┴─────────────────┐
                       │                                   │
                  ≥ THRESHOLD                         < THRESHOLD
                       │                                   │
                       ▼                                   ▼
                  write summary,                    log feedback,
                  exit (no commit)                  next iteration
```

Each iteration:

1. **Builder** edits the working tree with restricted tool permissions
   (`FACTORY_PHASE=builder`). The guardrail hook blocks any attempt to read
   the holdout or the loop wiring.
2. **Halt detection** — if the builder's `claude -p` call returned a
   rate-limit or billing error, state is saved and the loop exits cleanly.
   Never fakes a scenario failure for a billing event.
3. **Stage, do not commit.** `git add -A` plus a `git diff --cached --stat`
   summary. The loop never runs `git commit`.
4. **Judge** runs `VALIDATE_CMD`, reads the holdout, optionally reads the
   working tree, and returns a strict JSON object:
   `{"min_satisfaction": 0.0..1.0, "failing": [...], "notes": "..."}`.
   `min_satisfaction` is the *minimum* across all holdout scenarios — one
   fully-failing scenario drives it to zero.
5. **Converge or feed back.** If the score clears `THRESHOLD`, write a
   summary to `factory/runs/summaries/` and stop. Otherwise append the
   judge's `failing` + `notes` to `factory/memory/log.md` and loop.

## Safety boundaries

Enforced by `factory/.factory/guardrail.sh` (registered in
`.claude/settings.json` as a `PreToolUse` hook). Exit status 2 from the
hook blocks the tool call and surfaces the reason to the agent.

| Rule | Scope | Bypass |
|------|-------|--------|
| Writes to `factory/scenarios/holdout/**` are blocked | all phases | `factory/.bootstrap_unlock` sentinel (holdout only) |
| Writes to `factory/.factory/**` are blocked | all phases | **none** |
| Destructive commands are blocked (`rm -rf`, `git push --force`, `git reset --hard`, `git rebase -i`, `git commit --amend`, `git clean -f`, history rewrites) | all phases | **none** |
| Reads of `factory/scenarios/holdout/**` are blocked | `FACTORY_PHASE=builder` | judge phase permitted |
| Reads of `factory/.factory/**` are blocked | `FACTORY_PHASE=builder` | judge phase permitted |

The judge phase can read holdout (so it can score against it) but still
cannot write there. `.factory/` is always sealed for writes because it
holds `VALIDATE_CMD` and other loop wiring — leaking it to the builder
would reveal the judge's probe inputs and invite overfitting.

The loop **never commits**. The author reviews `factory/runs/summaries/`
and the staged diff, then commits manually.

## State and memory

| File | Purpose |
|------|---------|
| `factory/runs/state.json` | `{iter, last_model, halted_reason, last_score, task}` — the loop's resume point. |
| `factory/runs/summaries/<task>.md` | Written on convergence: judge notes + staged diff. |
| `factory/memory/log.md` | Append-only chronicle. Dated section per day (the Stop hook adds it). Per-iteration entries from builder ("what I changed") and judge ("failing scenarios + notes"). |
| `factory/memory/decisions.md` | ADR-style notes on assumptions and trade-offs. |

`halted_reason` is one of `""` (in progress), `"rate_limit"`,
`"billing_error"`, `"max_iters"`, `"approved_stop"` (converged, waiting for
human commit), `"builder_error"`, `"judge_parse_error"`. No `spent_usd`
field — the agent plan's own credit is the spend ceiling; halts surface
through the explicit reason instead of dollar estimation.

## Configuration

`factory/.factory/config` is shell-sourced by the loop:

```sh
VALIDATE_CMD="..."   # command the judge runs to exercise the program
MAX_ITERS=8          # cap on builder→judge iterations
THRESHOLD=0.9        # judge score required to converge
```

`VALIDATE_CMD` is captured (stdout + stderr) and shown to the judge in its
prompt; the builder never sees it. If your task is purely static (e.g.
generated files), leave it empty and rely on the judge's `Read`/`Bash`
permissions.

## Running

```sh
# One task. Loops until converged, MAX_ITERS, or a halt.
./factory/loop.sh <task-name>

# Single iteration then exit (useful for testing changes to the loop).
./factory/loop.sh <task-name> --one-iter
```

The loop expects `factory/tasks/<task-name>.md` to exist and at least one
scenario file under `factory/scenarios/`. Holdout scenarios go in
`factory/scenarios/holdout/`.

### Adding a task

1. Write `factory/tasks/<name>.md` describing what success looks like, in
   prose. Be specific about observable behavior.
2. Write 2–3 visible scenarios under `factory/scenarios/` covering the
   obvious cases. Each has a `**Satisfied when:**` clause in plain language.
3. Write 2–3 holdout scenarios under `factory/scenarios/holdout/` covering
   *distinct, harder* cases — failure modes the visible scenarios don't.
   The builder must not see these; the judge must be able to score them
   without ambiguity.
4. Edit `VALIDATE_CMD` in `factory/.factory/config` so it exercises the
   relevant program paths.

To seed the holdout, create an empty `factory/.bootstrap_unlock` file
first, write the scenarios, then delete the sentinel. The hook prints a
warning when the sentinel is honored so you can see it in real time.

### Worked example: `greet`

The repo ships with a small seed task (`factory/tasks/greet.md`) — a Python
CLI that prints `Hello, <NAME>!`. The first run of the loop converged in
one iteration with `min_satisfaction = 1.0`; see
`factory/runs/summaries/greet.md` for the judge's notes.

## Caveats and known weak spots

- **Hook live-load.** `.claude/settings.json` is read at Claude Code session
  start. Mid-session edits to the file do not retroactively activate hooks
  in the running session — but fresh `claude -p` subprocesses (which is
  where the loop runs anyway) load it. If you're editing the guardrail and
  testing it in your interactive session, restart.
- **Single-iteration convergence is a weak proof of the feedback loop.** The
  seed task converged on iter 1, so the failure-feedback-improvement cycle
  hasn't been observed empirically. The first task that doesn't pass at
  iter 1 will be the real credibility test.
- **Halt detection is untested in production.** Rate-limit and billing-error
  detection works against a synthetic envelope, but the real exhaustion
  event hasn't fired yet against this code. A misclassification would
  result in the loop continuing instead of pausing — fixable when observed
  but worth knowing.
- **`jq` is not used.** The loop parses JSON with `python3`, so `python3`
  is a hard dependency. `bash`, `git`, and the `claude` CLI are the others.
- **Cost.** Each iteration is two `claude -p` calls (builder + judge). With
  `MAX_ITERS=8`, a non-converging task is up to 16 calls.

## Dependencies

- `bash`
- `git`
- `python3` (3.8+)
- `claude` CLI (Claude Code, headless `claude -p` mode)

No third-party Python packages.

## License

MIT. See [LICENSE](LICENSE).

## Repository layout

```
.
├── .claude/
│   └── settings.json              hook registration (PreToolUse + Stop)
├── README.md
└── factory/
    ├── CLAUDE.md                  operating contract for any agent
    ├── loop.sh                    build/judge driver
    ├── .factory/
    │   ├── config                 VALIDATE_CMD, MAX_ITERS, THRESHOLD
    │   ├── guardrail.sh           PreToolUse hook
    │   └── on_stop.sh             Stop hook
    ├── scenarios/
    │   ├── *.md                   visible scenarios
    │   └── holdout/
    │       └── *.md               sealed scenarios (judge-only)
    ├── tasks/
    │   └── *.md                   one prose task per file
    ├── memory/
    │   ├── log.md                 append-only chronicle
    │   └── decisions.md           ADR-style notes
    └── runs/
        ├── state.json             resume state (gitignored)
        └── summaries/
            └── *.md               per-task convergence summaries
```
