# Decisions / ADRs

ADR-style notes on assumptions and choices that shape the factory or its tasks.
Newest at the bottom.

## 2026-05-27 — bootstrap session

- **Repo state at bootstrap:** essentially empty — just `README.md` with the
  text `# test`. No source, no language constraints. The factory itself is the
  bulk of the new files.
- **Seed task chosen:** a small Python CLI (`greet.py`). Reasoning: (a) Python
  is available on this machine and needs no install, (b) a CLI has crisp
  observable behavior (stdout, stderr, exit codes) which the judge can score
  without ambiguity, (c) it is small enough to plausibly converge inside
  `MAX_ITERS=8`. This is scaffolding for the first real loop, not a
  long-term feature commitment.
- **VALIDATE_CMD design:** left unset in `factory/.factory/config` for now.
  The seed task's VALIDATE_CMD is suggested in the launch instructions; the
  human is expected to set it before running the loop (or leave it empty and
  let the judge use `Bash` itself to invoke the script). Reason: keeping the
  config minimal makes the human's first edit explicit.
- **Bootstrap unlock sentinel:** `factory/.bootstrap_unlock` (when present)
  lets the guardrail permit writes to `factory/scenarios/holdout/`. Used by
  the bootstrap to seed holdout. To add more holdout scenarios later: create
  the empty sentinel, write, delete the sentinel. `.factory/` writes are
  NEVER unlocked, and destructive commands are NEVER unlocked.
- **`.factory/` is also sealed from builder _reads_.** The config holds
  `VALIDATE_CMD`, which encodes the probe inputs the judge uses. If the
  builder could read it, it would learn what the holdout tests and could
  overfit. Builder-phase `Read/Glob/Grep` and Bash read-verbs (`cat`,
  `head`, `grep`, etc.) targeting `factory/.factory/**` are blocked.
  Verified by a fresh `FACTORY_PHASE=builder claude -p` subprocess that
  attempted `cat factory/.factory/config` and was blocked by the hook.
- **No unconditional `permissions.deny` in the settings file.** Earlier
  draft listed `Write(factory/scenarios/holdout/**)` and similar in
  `permissions.deny`, but those fire *before* the hook and would have made
  the bootstrap_unlock sentinel dead in any future re-seeding session.
  Removed them; the hook is the single source of truth, and the sentinel
  is genuinely load-bearing for re-seeding.
- **Hooks live in `.claude/settings.json`, not `.claude/settings.local.json`.**
  The factory's safety guarantees depend on the guardrail hook firing, so
  the registration must travel with the repo. `settings.local.json` is in
  the user's global gitignore (per-machine convention). Hooks moved to the
  tracked `settings.json` at commit time. If a user has their own
  `settings.local.json`, Claude Code will merge the two — the hooks from
  the repo still fire.
- **Hook live-load caveat:** hooks defined in `.claude/settings.local.json`
  are loaded at Claude Code session start. Mid-session edits to the settings
  file do not retroactively activate hooks in the running session. Gate 1
  proved the hook *does* fire in a fresh `claude -p` subprocess (which is
  where the loop runs anyway, so the boundary is correct).
- **No spent_usd telemetry.** State tracks `iter`, `last_model`,
  `halted_reason`, `last_score`. The Max plan / Agent SDK credit is the
  spend ceiling; the loop pauses cleanly on `rate_limit` / `billing_error`
  rather than estimating dollars.
- **VALIDATE_CMD ships set, not empty.** Supersedes the "left unset" entry
  above. The seed task's VALIDATE_CMD now lives in
  `factory/.factory/config` and probes all four observable behaviors
  (basic, multi-word, missing arg, --help). Because the builder cannot
  read `.factory/`, shipping it set does not leak probe inputs.
