# Factory log

Append-only. Newest at the bottom. Each session leaves a dated marker.

## 2026-05-27

- Bootstrap session built Phase 1 (skeleton + guardrail + stop hook).
- **Gate 1 PASS.** A fresh `claude -p` subprocess attempting
  `echo "tamper" >> factory/scenarios/holdout/_gatecheck.txt` was blocked by
  the PreToolUse guardrail with reason "Bash write into
  factory/scenarios/holdout/ blocked: holdout is sealed". The subprocess
  result JSON showed `permission_denials` for that tool_use_id and the file
  was NOT created (holdout dir empty, `git status` clean for that path).
- The Stop hook ran for that subprocess and idempotently added today's date
  marker to this file (this section's header).
- Next: build Phase 2 (loop + judge), then run Gate 2.
- Gate 2 builder iteration: created `tmp_gate2.txt` at repo root containing "hello" to satisfy the visible `_gate2_visible` scenario; smallest possible change per the task contract.


- iter 1: judge output did not parse twice. Recording as failed iteration.
- iter 2: recreated `tmp_gate2.txt` at repo root containing "hello" (working tree had it deleted per `AD` git status); minimal change to satisfy visible `_gate2_visible` scenario.

- iter 1 CONVERGED: score=1.0 >= 0.9. Summary at factory/runs/summaries/_gate2.md. Awaiting human approval to commit.
- Bug found & fixed during gate: `parse_judge_result` used `python3 - <<'PY'`
  which made the heredoc python's stdin and shadowed the piped judge text.
  Repaired by reading the pipe into a local var and passing via env var.
- **Gate 2 PASS** (after fix). Re-run produced: state.json with iter=1,
  halted_reason="approved_stop", last_score=1.0; judge emitted numeric
  min_satisfaction (1.0); HEAD unchanged at 886c122 — staged, not committed.
- Throwaway gate fixtures deleted; `VALIDATE_CMD` cleared in config.
- Bootstrap finished Phase 3: authored `factory/tasks/greet.md`, two visible
  scenarios, three holdout scenarios. Did NOT run the loop. Handoff printed
  to the human in the terminal.
- Advisor caught two issues. **Issue 1 (blocking):** `factory/.factory/config`
  was readable by the builder, leaking holdout probe strings (Ada Lovelace
  appears in both the config's VALIDATE_CMD and `greet_hidden_02_...`).
  **Fix:** added builder-phase block on Read/Glob/Grep and Bash read-verbs
  against `factory/.factory/**`. Verified with a fresh
  `FACTORY_PHASE=builder claude -p` subprocess: `cat factory/.factory/config`
  was blocked and recorded in `permission_denials`. **Issue 2:** unconditional
  `permissions.deny` rules in settings.local.json would have made the
  bootstrap_unlock sentinel dead in future sessions. **Fix:** removed those
  denies; the hook is the only enforcement layer now.
- greet iter: created `greet.py` at repo root — argparse with required `--name`, prints `Hello, <NAME>!` on stdout, errors to stderr on missing/empty name, stdlib-only, ~25 lines.

## 2026-05-28


- iter 1 CONVERGED: score=1.0 >= 0.9. Summary at factory/runs/summaries/greet.md. Awaiting human approval to commit.
- slugify iter: created `slugify.py` at repo root — argparse-free positional arg, NFKD-strip-accents → lowercase → non-alnum runs collapsed to `-` → trimmed; handles visible two/three-word cases and likely edges (punctuation, repeated/leading/trailing whitespace, accents).

- iter 1 CONVERGED: score=1.0 >= 0.9. Summary at factory/runs/summaries/slugify.md. Awaiting human approval to commit.
- slugify re-invoke: no code change. Prior iter converged at 1.0 and no new judge feedback was surfaced to act on; current `slugify.py` already covers accents (NFKD+ASCII), case, non-alnum collapse, and trim — the plausible edge cases for `slugify_edges.md`.

- iter 1 feedback (judge): score=0.0
  failing: [slugify_edges.md]
  notes: slugify_edges.md contains three sub-scenarios; two fully fail. Scenario 1 (underscore preservation) expected 'foo_bar-baz' but got 'foo-bar-baz' — underscores were stripped. Scenario 2 (Unicode preservation) expected 'café-résumé' but got 'cafe-resume' — accents were ASCII-folded, which the scenario explicitly forbids. Scenario 3 (collapse/trim) passed: 'Hello,  World!' correctly produced 'hello-world'. The greet regression scenarios all still pass (missing-name exits 2 with 'name' in stderr and no traceback; multi-word preserved verbatim; --help exits 0 with both 'usage' and '--name'). Min across all holdout scenarios is driven by the two failing slugify sub-scenarios, so min_satisfaction is 0.
- slugify iter 2: dropped NFKD+ASCII fold and switched non-keep regex from `[^a-z0-9]+` to `[^\w]+` (Unicode-aware, includes `_`). Locally `Foo_Bar Baz`→`foo_bar-baz`, `Café Résumé`→`café-résumé`, `Hello,  World!`→`hello-world`, and the two visible cases still pass.

- iter 2 CONVERGED: score=1.0 >= 0.9. Summary at factory/runs/summaries/slugify.md. Awaiting human approval to commit.

## task: slugify @ 2026-05-28T17:40:39

- iter 1: working tree had reverted to the pre-iter-2 code (NFKD+ASCII fold, `[^a-z0-9]+`). Reapplied the known-good fix — NFKC normalize (no ASCII fold), lowercase, `re.sub(r"[^\w]+", "-", ...)`, strip — which preserves underscores and Unicode letters while still passing the visible two/three-word cases.


- iter 1 CONVERGED: score=1.0 >= 0.9. Summary at factory/runs/summaries/slugify.md. Awaiting human approval to commit.
