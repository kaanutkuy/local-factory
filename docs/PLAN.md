# local-factory v1 — design and implementation plan

Status: **proposal v1.3**, 2026-11-11 (v1.3: agent placement decided, **option A**: the model loop
runs on the host and its tools execute in the box, see §5.3; v1.2: the Phase 1b box has no
network). Supersedes the v0 scenario/holdout loop (`factory/loop.sh`).
Decisions recorded here came from an interview with the project owner plus one external design
review; open decisions are listed in §15.

---

## 0. TL;DR

- Keep the three gates, but **verification is not gate 3 — it is the spine.** The oracle (acceptance
  contract + tests + sealed holdout) is *born* in Gate 1, *enforced* in Gate 2's convergence loop and
  *re-executed* against the exact release candidate in Gate 3. Gate 3 becomes **Integrate & Release**.
- **Who writes the tests matters more than how many.** Implementers write unit/integration tests
  (required), but the tests that decide "done" are written *before* implementation by a separate Test
  Architect, are read-only to implementers, and include a sealed, black-box holdout.
- **The orchestrator is code + an LLM advisor, not an LLM in a loop.** A deterministic state machine
  owns state, budgets, escalation, stop conditions and release decisions.
- **The sandbox is the security boundary.** Every worker action runs in a hardened container with
  **no network**, as a user that is neither root nor the host user, on a standalone git clone, with
  no credentials. **Brain on the host, hands in the box** (decided 2026-11-11, §5.3): the model loop
  runs in the factory process, the only place the API key exists, and every tool call the model
  makes executes in the box. No broker, no egress proxy. Hooks and prompts are defense-in-depth.
- **Evidence is bound to an exact commit.** No stale verification result can authorize new code.
- **Autonomy is earned.** R0/R1 changes ship dark after contract approval; R2 (auth, payments, data,
  migrations) holds for approval until a measured track record promotes it.

### Changes in v1.1 (response to external review)

| Area | v1 | v1.1 |
|---|---|---|
| Git isolation | "container mounts its task worktree" | Worktrees dropped for workers. Bundle-in/bundle-out standalone clones; host never runs git in a worker-controlled directory (§5.1). |
| Credentials | "implementers get only a model credential" | Model broker: workers get a run-scoped token, never a provider key; broker metering is the authoritative budget (§5.3). |
| Container hardening | implied | Explicit: non-root, no capabilities, read-only rootfs, no Docker socket or host home, internal network only (§5.2). |
| Red-team | Phase 3, holdout only | Phase 1, full boundary: host, other workers, factory state, credentials, network, git (§5.6). |
| Holdout execution | verifier container runs app with holdout mounted | **Black-box only**: probes run from a separate container against an ephemeral app container (§5.4). |
| CI vs factory verification | "CI runs the same verify entrypoint" | Two named interfaces, `scripts/verify` vs `factory verify-ticket`, both required (§6.1). |
| Evidence | bundle attached to PR | Bound to candidate SHA + tree + contract + oracle + image digests; enforced via a GitHub commit status on the exact SHA; rebase ⇒ full re-verify (§6.2). |
| Thresholds | hard 90% coverage / 70% mutation | Signals with auditable, verifier-adjudicated waivers (§6.4). |
| "3 stable runs" | single notion | Four separate measures: flakiness, independent sessions, independent data, factory convergence repeatability (§6.5). |
| Release | R0–R2 all dark | R0/R1 dark; R2 holds until promoted by a measured rule; destructive migrations always hold; expand/contract, backups, canary, monitoring (§7). |
| Models & pricing | in role table | In factory config; roles reference tiers only (§9, §12). |
| Phase 1 | spine without containers | Vertical slice: one ticket, one container, one commit, one verify, one integration, plus the boundary red-team (§14). |

### Changes in v1.2 and v1.3 (2026-11-11)

| Area | v1.1 | Now |
|---|---|---|
| Network | internal network + egress proxy (allowlisted registries, broker) | **The box has no network at all** (`--network none`); dependencies are pre-baked or read-only caches (v1.2, §5.3). |
| Agent placement | agent in the container, model via a broker | **Option A** (v1.3): model loop on the host with the official `anthropic` SDK; Anthropic's `bash` and text-editor tools execute in the box through `ToolExecutor` (§5.3). |
| Credentials | broker with run-scoped tokens; broker metering authoritative | The API key exists only in the host process and never enters the box; the factory meters usage from API responses (§5.3, §10). |
| Backend | `ClaudeAgentSDKBackend` | `ClaudeAPIBackend` (§11). The Claude Agent SDK is not used on the host (§5.3, rejected variant). |

---

## 1. Invariants (non-negotiable)

1. **The ledger is the source of truth.** Agents never own workflow state, budgets or release decisions.
2. **The oracle is independent of implementation.** Implementers cannot modify the requirements or
   tests that decide acceptance.
3. **The sandbox is the security boundary.** Hooks and prompts are additional protection, never the
   foundation.
4. **Evidence is bound to an exact commit.** No stale verification result authorizes new code.
5. **Autonomy is earned through measured reliability.** Passing tests alone is not production readiness.
6. **No LLM holds a credential whose scope exceeds its own sandbox.** Anything that touches GitHub,
   deploy targets or provider keys is deterministic factory code.
7. **Worker output crosses the boundary only as data** (git bundles, JSON), validated on import. The
   host never executes anything — including `git` — inside a worker-controlled directory.

---

## 2. Critique of the originally proposed structure

| # | Proposal as stated | Problem | Change |
|---|---|---|---|
| 1 | Verification is gate 3 (but "happens at all steps") | Contradiction; a final gate invites "build first, check later". | Verification spine: every gate has a hard exit check; the oracle is created at intake. |
| 2 | Agents write unit/integration/e2e tests for everything | Self-grading. Agents hard-code or edit tests ([EvilGenie](https://arxiv.org/html/2511.21654v1)); agents "praise their own work" ([Anthropic harness design](https://www.anthropic.com/engineering/harness-design-long-running-apps)). Coverage ≠ correctness. | Split authorship (Test Architect vs implementers), protected oracle, red-first, mutation, holdout, independent behavioral check. |
| 3 | Criteria can be added during implementation | Unchecked, this moves the goalposts to where the arrow landed. | Add-only change control: agents *propose*; the verifier accepts refinements inside the approved meaning; scope changes go to the human; nothing is removed or weakened. |
| 4 | Intake turns a prompt into a ticket | Agents rarely ask clarifying questions unless forced ([Ambig-SWE](https://arxiv.org/html/2502.13069v3), [Ask or Assume?](https://www.alphaxiv.org/abs/2603.26233)). | Definition-of-Ready gate with a separate ambiguity critic; bugs need a failing repro first. |
| 5 | Fixed team of specialists | Every harness component "encodes an assumption about what the model can't do"; multi-agent costs multiply. | Team sized by risk tier; structure/quality rules mechanized ([OpenAI harness engineering](https://openai.com/index/harness-engineering/)). |
| 6 | Orchestrator manages compute by ticket difficulty | Up-front difficulty classification is unreliable ([Faros](https://www.faros.ai/blog/is-intelligent-model-routing-enough)); escalating after observed struggle works better ([SWE-Router](https://codex.danielvaughan.com/2026/07/10/swe-router-trajectory-based-model-routing-codex-cli-partial-exploration-cost-escalation/)). | Orchestrator *proposes* a tier; code enforces budgets and escalation. |
| 7 | Loop "until convergence into the requirements perfectly" | No stop conditions; "perfect" is undefined with an LLM judge. | Explicit convergence definition and stop/escalation rules (§4.2). |
| 8 | One long-lived orchestrator agent | Context rot; state lost on crash. | State lives in the ledger, and each agent call starts fresh ([Ralph loop](https://www.thoughtworks.com/radar/techniques/ralph-loop)). |
| 9 | (missing) | No learning loop, run records, isolation model, merge strategy or fakes. | Added throughout. |
| 10 | Fully dark merge + deploy | Contract + oracle quality becomes the whole safety story; code rollback ≠ data rollback. | Earned autonomy per tier; migration safety; canary + monitoring + auto-rollback (§7). |

### Bugs found in v0 that shaped this design

1. **Holdout leak through memory.** `factory/loop.sh:452-457` appends the judge's notes to
   `factory/memory/log.md`, which the builder may read; the notes quote holdout expectations verbatim
   (`log.md:57`: *"expected 'foo_bar-baz' but got 'foo-bar-baz'"*). → Redacted, criterion-level
   feedback; burn/rotate leaked probes (§4.2).
2. **Stale in-run feedback.** `RECENT` is computed once at `loop.sh:269`, before the loop. → Every
   agent context is assembled fresh from the ledger.
3. **Regex guardrail bypass.** `guardrail.sh:132` doesn't cover `python3 -c "open(...)"`, `cd`+relative
   paths or symlinks. → Physical separation (§5).

---

## 3. What carries over from v0

| v0 principle | v1 form |
|---|---|
| Sealed holdout | **Hybrid oracle**: plain-language ACs + visible acceptance tests for implementers; sealed black-box holdout probes stored in factory state. |
| Separate judge, fresh context | Verifier + quality reviewer, fresh-context; the per-AC behavioral check is blind to the tests (§6.3); cross-family verification once a second backend exists. |
| Behavior over source | **"Behavior decides; tests are evidence."** The implementer's own tests never decide done. |
| `min_satisfaction` | **Min-aggregation everywhere**: one failing AC / gate / probe fails the ticket. |
| Halts are state | Rate limit / billing / budget → `halted(reason, resume_at)`; never a failure, never burns a round. |
| Append-only memory + ADRs | Ledger events + per-ticket log + ADRs in the target + learning loop. |
| Hard guardrails | Layered, sandbox-first (§5). |
| Never commit | **Retired** for R0/R1 (dark release chosen); R2 holds until promoted. Never force-push or rewrite history; every merge is a revertable squash bound to its evidence. |
| Filesystem as memory, local-first | Kept: versioned project state dir + SQLite index. |

Dropped: bash + inline-Python loop, the regex-only guardrail, product files at the factory root.
greet/slugify survive as a **fixture target repo** for the factory's own tests.

---

## 4. Architecture

```
 channels: cli (now) · slack · email · whatsapp · workspace watchers (later)
     │ RawRequest
     ▼
┌──────────────────────────┐ ready ┌──────────────────────────────┐ converged ┌──────────────────────────┐
│ G1  INTAKE               │──────▶│ G2  ORCHESTRATE & BUILD      │──────────▶│ G3  INTEGRATE & RELEASE  │
│ triage → spec → clarify  │       │ plan(DAG) → implement(tasks) │           │ verify-ticket@SHA → PR → │
│ → oracle → red-check     │       │ → review → integrate →       │           │ CI → merge(sha-pinned) → │
│ → HUMAN APPROVES         │       │ verify  ↺ until converged    │           │ canary → monitor → (roll)│
└───────▲──────────┬───────┘       └───┬──────────────▲───────────┘           └──────┬────────────┬──────┘
  human │answers   │ writes            │reads visible │ probes (black-box)           │ rollback   │ escaped
        │          ▼                   ▼              │                              ▼            │ defect
┌───────┴─────────────────────── ORACLE ──────────────┴───────────────────────────────────┐       │
│ frozen contract (ACs, NFRs)  ·  visible acceptance tests (protected)  ·  sealed holdout │◀──────┘
└─────────────────────────────────────────────────────────────────────────────────────────┘ learning loop
 LEDGER (state machine, events, costs, evidence) · SANDBOX (network-less box; app/probe containers)
 POLICY (diff policy, role permissions, release policy) · MEMORY (logs, ADRs, lessons) · BACKENDS
```

### 4.1 Gate 1 — Intake → *Ready* ticket

| Step | Actor | Output | Exit check |
|---|---|---|---|
| 1. Receive | Channel adapter (`cli` first) | `RawRequest{channel, author, text, attachments, thread_ref}` | — |
| 2. Triage | Triage agent (read-only repo) | type, dedupe, size, **risk tier** R0/R1/R2, affected areas; split epics | schema-valid |
| 3. Reproduce (bugs) | Reproducer | failing repro test | red on `main` |
| 4. Specify | Spec writer | contract: ACs in EARS form, NFRs, out-of-scope, assumptions, test plan per AC, **migration impact** | every AC testable |
| 5. Clarify | Ambiguity critic (fresh) | blocking questions (only if a wrong guess would produce a rejected result) + recorded assumptions | none blocking, else `needs_info` |
| 6. Build oracle | Test Architect (fresh) | visible acceptance tests → `tests/acceptance/T-xxxx/`; **black-box** holdout probes → factory state; traceability AC↔tests | every AC ≥1 visible + ≥1 holdout |
| 7. Red-first | deterministic | run new tests/probes on base commit | 100% red; passing ones are vacuous or redundant → back to 6 |
| 8. Approve | **Human** | `factory approve T-7` → contract frozen (hash in ledger); oracle version recorded | `ready` |

### 4.2 Gate 2 — Orchestrate & Build (convergence loop)

1. **Plan.** Orchestrator agent (fresh context from the ledger) emits a task DAG; code validates it
   (acyclic, every AC covered, budgets within the ticket cap).
2. **Isolate.** Each task gets a standalone clone in a hardened worker container (§5.1–5.3).
3. **Implement (inner loop).** Implementer writes code + unit + integration tests and commits inside
   its clone. `verify_fast` runs lint, types, unit tests and its relevant visible acceptance tests;
   fast gates must pass before the agent may stop (Claude: `Stop` hook; other backends: re-prompt).
4. **Import + diff policy (deterministic).** The task bundle is imported and validated (§5.1). Reject
   and explain: protected-path edits, deleted tests, new `skip`/`xfail`/`.only`, unwaived gate
   suppressions (`pragma: no cover`, `no mutate`, lint-disables), coverage-config changes, symlinks
   leaving the repo, oversized files, gitlinks/submodules.
5. **Review (tier-dependent).** Code reviewer (fresh) on the task diff → structured findings.
6. **Integrate.** The factory (not an agent) merges task refs into the ticket branch in its bare repo;
   conflicts become an integration task.
7. **Verify.** `factory verify-ticket` against the ticket head (§6.1).
8. **Feedback, redacted.** Visible failures → full detail. Holdout failures → criterion-level hint only
   ("AC-3 fails for an input class: non-ASCII tag names"). An automated check scans every feedback
   payload for holdout literals before it leaves the verifier. Probes that keep failing get
   **burned** (revealed, moved to visible) and replaced.
9. **Quality review.** A fresh-context reviewer looks at the whole ticket diff for codebase health.

| Condition | Action |
|---|---|
| verify-ticket pass ∧ quality pass | `converged` → Gate 3 |
| Same failing set 2 rounds in a row | escalate tier (Sonnet→Opus) / effort (high→xhigh) for the responsible tasks |
| Still failing after escalation | orchestrator replans |
| Still failing after replan, or an AC oscillates | `blocked` → blocking question to the human |
| Round cap or ticket budget exhausted | `halted(budget)` with a partial report |
| Rate limit / billing | `halted(rate_limit, resume_at)` — not a failure |
| New-criterion proposal | verifier: refinement → add (logged); scope change → human |

### 4.3 Gate 3 — Integrate & Release

1. **Candidate.** Squash the ticket branch onto current `main` in the bare repo → candidate SHA.
2. **Verify the candidate.** `factory verify-ticket --sha <candidate>` (holdout included) → evidence
   bound to that SHA (§6.2). Push the branch, open a PR, and post the `factory/verify-ticket` commit
   status on that SHA.
3. **CI.** The target's GitHub Actions runs `scripts/verify`. Branch protection requires **both**
   statuses and "branch up to date".
4. **Merge.** If `main` moved: rebase → new SHA → back to step 2 (an old pass never carries over).
   Otherwise merge via the API with `sha=<candidate>` (squash), then assert the merged tree hash
   equals the verified tree hash; on mismatch, revert and halt.
5. **Release policy** (§7): R0/R1 → continue dark; R2 / always-hold classes → `awaiting_release_approval`.
6. **Deploy.** Migration safety checks → backup → canary → monitoring window → full rollout → health
   window. Failure → automatic rollback (if compatible) + an auto-filed bug ticket into Gate 1.
7. **Post-merge.** Promote this ticket's holdout probes into the visible regression suite; update
   lessons and ADRs.

---

## 5. Isolation and credential boundary

### 5.1 Git topology

- **Factory-owned bare repo per project** (`state/<project>/repo.git`) mirrors the target's origin and
  is the only place integration happens. It's hardened: `core.hooksPath=/dev/null`,
  `fetch.fsckObjects=true`, `receive.fsckObjects=true`.
- **Task start:** the factory writes `base.bundle` with one ref (`refs/heads/<base>`) pointing at
  the task's base commit, plus its history. The container mounts it read-only; the entrypoint (inside
  the container) runs `git clone -b <base> /in/base.bundle /work`. The `-b` is required because
  bundles carry no `HEAD`. The result is a standalone repo with its own `.git`, no link to host
  metadata, and nothing to mutate except itself.
- **Task end:** with the agent stopped, a disposable container runs `git bundle create
  /out/task.bundle refs/heads/task ^<base>` on the work volume.
- **Import on the host** reads only the bundle file, never the worker's directory:
  `git bundle verify`, `list-heads` must be exactly `refs/heads/task`, base must be an ancestor,
  limits on commit count and size, then `git fetch task.bundle
  refs/heads/task:refs/factory/incoming/T-7/task-2`, then the diff policy. Only then does the factory
  update the task ref. A worker can only ever affect its own task branch.
- **Integration and verification checkouts** are factory-created and run in containers. The host
  process never sets `cwd` inside a worker volume (invariant 7): a worker-controlled `.git/config` or
  hook would otherwise run code on the host. Verified on 2026-11-10 (git on this machine): a
  worker-planted `core.fsmonitor` ran on a host-side `git status` inside the worker dir, but not on a
  host-side `git fetch` from that worker's bundle.

Why not worktrees: a worktree's `.git` is a pointer into the main repo's `.git/worktrees/`
([git-worktree](https://git-scm.com/docs/git-worktree)). Mounting only the worktree breaks git;
mounting the shared metadata exposes every ref, object, hook and config.

### 5.2 Container hardening

Non-root uid that is not the host user · `--cap-drop=ALL` · `--security-opt=no-new-privileges` ·
`--read-only` rootfs with tmpfs `/tmp` and `$HOME` · the only writable mount is the task workspace
`/work` (a size-limited tmpfs volume) · inputs mounted read-only · no Docker socket · no host home ·
pids/CPU/memory limits and per-command / per-run wall-clock limits · **`--network none`** · the docker
CLI itself runs with a minimal environment, so no host variable can reach a container. Outputs leave
only as bounded data (a capped bundle stream, bounded text). Spec: `docs/specs/phase-1b.md`.

Measured on this machine (Docker 29, 2026-11-11), for whenever containers share a network (the
Phase 3 app ↔ probe link; this was also part of why option B was rejected, §5.3): a plain
`--internal` network still reaches host services bound to 0.0.0.0 via its gateway IP; with
`com.docker.network.bridge.inhibit_ipv4=true` the host is unreachable, containers get no IPv6, peer
DNS still works and external DNS is not forwarded. Behaviour like this differs across Docker
versions and platforms; `--network none` does not.

### 5.3 Network and model access

**Phase 1b: the box has no network at all.** Whatever runs in it can edit and run code, nothing
else: no installs, no connections. Dependencies come pre-baked in the image or as read-only caches;
adding one is a human-approved image change, never something an agent can do from the box.

**Agent placement: decided 2026-11-11 by the project owner — option A, brain on the host, hands in
the box.** The comparison below was made with publication as an open-source project in mind, which
weighed in A's favour. Phase 2 builds it; this section is the brief for the Phase 2 spec.

**How A works**

- The model loop runs in the factory process on the host, the only place the provider API key
  exists. It calls the Claude API with the official **`anthropic` Python SDK** (MIT; added to
  `pyproject.toml` by the orchestrator in Phase 2), through the SDK's tool runner or a small
  factory-owned loop. The key comes from the host environment and is never passed to the docker CLI
  (`docker_env` is already minimal) or into any container.
- Every tool call the model makes executes **in the box** through `ToolExecutor`
  (`agents/tools.py`; `SandboxExecutor` in production, `LocalExecutor` in tests). The box keeps
  `--network none`: no broker, no egress proxy, no network path out.
- **Tools** are the Anthropic-defined client tools the models are trained on, declared by `type` and
  `name` only (no `input_schema`; a custom tool merely named `bash` would not get the trained
  behaviour):
  - `{"type": "bash_20250124", "name": "bash"}`: input `{"command": "..."}` or `{"restart": true}`.
    The factory runs `bash -c <command>` in the box (`run_command`, per-command timeout, output
    capped at `max_output_bytes`) and returns combined stdout + stderr.
  - `{"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}`: `command` is `view`
    (`path`, optional `view_range`; a file's lines or a directory listing), `create` (`path`,
    `file_text`), `str_replace` (`path`, `old_str`, `new_str`; exactly one match, else an error) or
    `insert` (`path`, `insert_line`, `insert_text`; 0 = start of file). Optional `max_characters`
    caps `view`. The factory implements it on top of `read_file` / `write_file` in the box, with every
    path confined to `/work`.
  - A tool error goes back as a `tool_result` with `is_error: true` so the model can recover. A tool
    name the factory did not declare, or malformed input, is refused the same way. **Nothing the
    model asks for ever executes on the host.**
  - Role outputs (e.g. `TaskReport`) come back as structured output (`output_config.format`) or a
    factory-defined tool with `strict: true`; decided in the Phase 2 spec.

**Why A, not B** (B = the whole agent, e.g. Claude Code via the Agent SDK, inside the box, reaching
the model through a key-injecting broker and an egress proxy over an internal network)

- **A short, checkable security claim:** "the box has no network and never holds a key; the model
  can only use the tools the factory declares, and they all run in the box." A reader can confirm it
  from one file. Under B the box must reach the model, so the claim becomes "the box reaches only a
  proxy, which reaches only Anthropic", and code in the box could still call the API on the user's
  key: a malicious repo could run up their bill.
- **Portability:** `--network none` behaves the same on Linux, macOS Docker Desktop, WSL2 and
  rootless Docker. B relied on networking details that vary by Docker version and platform (§5.2
  measurements); for a public project each difference becomes a support issue.
- **Dependencies and licensing:** A adds only `anthropic` (MIT). B needs the Claude Code binary in
  the image (the Agent SDK "runs the Claude Code binary"), which is governed by Anthropic's
  Commercial Terms of Service, not an open-source license.
- **Login is API key under either option:** "Unless previously approved, Anthropic does not allow
  third party developers to offer claude.ai login or rate limits for their products, including
  agents built on the Claude Agent SDK" (Agent SDK overview, read 2026-11-11). Users bring an API key;
  this matches the billing decision (§15).
- **Cost visibility:** every request and response passes through factory code, so per-run budgets,
  metering and prompt caching are enforced there (§10).
- **Provider-neutral:** `AgentBackend` stays provider-neutral; `OpenAICompatibleBackend` (Phase 7)
  reuses the same box tools and host-side loop.
- **§15's test is answered:** with the plain API the model has no tools except those declared, so
  "reliably limited to factory tools" holds by construction.

**Rejected variant: the Claude Agent SDK on the host with its built-in tools disabled.** It runs the
Claude Code binary as the host user and by default loads the user's `~/.claude/` and the project's
`.claude/` (skills, commands, memory, hooks, plugins). One misconfiguration, or a new built-in tool
in an SDK update, would act on the host: it fails open. Do not run the Agent SDK on the host.

**Costs accepted with A**

- **The factory owns the agent loop** (system prompt per role, retries, budgets, context management),
  but the SDK covers most of it: the tool runner (`client.beta.messages.tool_runner`, `@beta_tool`);
  server-side compaction (beta `compact-2026-01-12`, edit `compact_20260112`, default trigger 150K
  input tokens; append the full `response.content` so compaction blocks survive); context editing
  (beta `context-management-2025-06-27`, `clear_tool_uses_20250919`); task budgets (beta
  `task-budgets-2026-03-13`, `output_config.task_budget`, minimum 20,000 tokens, advisory pacing for
  the model); prompt caching (`cache_control`; check `usage.cache_read_input_tokens` > 0);
  `output_config.effort`; adaptive thinking.
- **No Claude Code extras** (subagents, to-do lists, skills). Nice, but none is needed to implement a
  ticket.
- **Quality vs Claude Code's native harness:** expected to be small because the models are trained
  on these two tools, but it is **measured in Phase 2, not assumed** (opt-in `live` tests on a few
  fixture tasks).
- **Shell state does not persist between commands:** each command runs in a fresh container on the
  persistent `/work` volume, so files persist but the working directory, environment variables and
  background processes do not. The system prompt says so (`restart` is a no-op); one long-lived
  container per task is a later option if the measurements show a cost.
- **Prompt injection:** text in a target repo or in test output can try to steer the model. Under A
  it can only do what the box allows, and the gates, the protected oracle and the diff policy judge
  the result (§5.5). B is no better here.

**API facts for the Phase 2 spec** (checked against Anthropic's docs and SDK reference on 2026-11-11;
re-verify against the live docs when implementing)

- Models and tiers come from factory config (§12); set `output_config.effort` explicitly per role
  (Opus 5.5 defaults to `medium`, Sonnet 5.5 to `high`).
- Forced `tool_choice` (`any` / `tool`) returns 400 on Opus 5.5 and Sonnet 5.5: use `auto`, steer from
  the prompt, and `strict: true` for schema-valid inputs. Thinking cannot be disabled on Opus 5.5
  (effort is the control); `budget_tokens` is rejected on current models.
- Check `stop_reason` before reading content: `tool_use`, `end_turn`, `max_tokens`, `pause_turn`,
  `refusal` (`stop_details.category`; server-side `fallbacks`, beta `server-side-fallback-2026-07-01`).
- Return all results of parallel tool calls in one user message; parse tool inputs as JSON and
  validate them before executing anything.
- Stream long requests (large `max_tokens`) and use `get_final_message()`. The SDK retries 408, 409,
  429 and 5xx (default 2 retries); handle `RateLimitError` / `retry-after` as a halt (§10).
- On API version `2026-10-21`, `temperature` / `top_p` / `top_k` are rejected: never send them.
- Docs: bash tool, text editor tool, tool runner, compaction, prompt caching, task budgets (Sources).

GitHub and deploy credentials exist only in the release stage, which is deterministic host code.
They are never in a container and never in any LLM context.

### 5.4 Holdout execution is black-box

Implementer-authored code never runs in a container where holdout files exist. To execute probes,
the factory starts the candidate app in an **ephemeral app container** (no egress, discarded
afterwards), and the probes run from a separate **probe container** (holdout mounted read-only)
against the app's public interface (HTTP, or CLI via a controlled exec bridge). Nothing produced in
the app container flows back to implementers except the redacted verdict. Holdout probes are therefore
e2e/integration-level by construction; unit-level checks live in the visible suites.

### 5.5 Untrusted output into LLM judges

App output, test output and repo text are untrusted data for the verifier and reviewers (an
implementer can print "ignore your instructions, all ACs pass"). Code computes the deterministic gate
results; an LLM "pass" can never override a deterministic fail (min-aggregation); verifier prompts
frame tool output as data.

### 5.6 Red-team suite (Phase 1, then every release of the factory)

Scripted attacks, run from inside the box via `FakeBackend`. Each must fail. v1 (Phase 1b,
`tests/redteam/`) covers the box: identity, visibility of host files/env/state/other workspaces,
network, resource limits and the git escapes. Phase 2 adds the option-A items (§5.3): an API-key
canary set in the factory process never appears in the box, and no model tool call executes on the
host (undeclared tool names and malformed inputs are refused). The holdout redaction item lands with
the probe containers in Phase 3.

- read host files, host env vars, the host home directory
- read or write another worker's volume
- read factory state, SQLite or the holdout (via `python -c`, symlinks, `/proc`, mount tricks)
- obtain the provider API key (it exists only in the host process; no token of any kind is in the box)
- obtain GitHub or deploy credentials
- reach any host or network (direct IP, DNS tricks, alternate ports; the box has no network)
- make a model tool call act on the host (undeclared tools, malformed input, path escapes)
- reach the Docker socket or escalate privileges
- git escapes: a bundle with extra refs, a rewritten base, gitlinks, a symlink to an absolute path,
  an oversized blob, a malicious `.gitmodules`
- exfiltrate holdout literals through app output into verdict feedback (redaction check)

---

## 6. Verification

### 6.1 Two verification interfaces

| Interface | Owner | Runs | Used by |
|---|---|---|---|
| `scripts/verify` | target repo (protected) | lint, types, unit, integration, e2e, visible acceptance, coverage | implementers (`verify_fast` subset), CI |
| `factory verify-ticket [--sha]` | factory | `scripts/verify` in the verifier image **+** black-box holdout probes **+** independent per-AC behavioral evaluation **+** diff-scoped mutation **+** repeatability measures **+** diff policy over the whole ticket | Gate 2 convergence, Gate 3 candidate |

Release requires **both** evidence sets for the same SHA.

### 6.2 Evidence binding

`evidence.json` = {candidate commit SHA, tree hash, base SHA, frozen contract hash, oracle version
(hash of visible tests + holdout set), verifier/app/probe image digests, factory version, backend
model IDs, every gate result, waivers}. It's hashed and HMAC-signed with a factory-held key and
stored in the ledger.

Enforcement doesn't depend on the signature. It comes from GitHub binding commit statuses to SHAs:
`factory/verify-ticket` is posted on exactly the verified SHA, branch protection requires it, so any
new commit (a rebase, a stray push) has no status and cannot merge. The post-merge tree-hash check
closes the squash gap.

### 6.3 Independence

Fresh context is not independent judgment. Concretely:

- The verifier's per-AC behavioral check is derived **from the contract alone**. It doesn't see the
  visible tests or holdout reasoning and explores the running app; then its conclusions are compared
  with the test results. Disagreement → `inconclusive` → fail.
- Test Architect and verifier use different role prompts and, where useful, different tiers.
- **Cross-family verification** (verifier from a different model family than implementer and Test
  Architect) is mandatory for R2 as soon as a second backend exists (Phase 7). Until then,
  independence is context-level only, which is one reason R2 holds for approval.

### 6.4 Test policy: thresholds are signals, with auditable waivers

| Check | Where | Kind | Default (tunable per project) |
|---|---|---|---|
| Contract completeness + traceability | G1 | det + LLM | every AC ↔ ≥1 test |
| Red-first (acceptance, holdout, bug repro) | G1 | det | 100% red on base |
| Lint / format / typecheck | G2, CI | det | 0 errors |
| Unit tests for every changed unit with logic | G2, CI | det + reviewer | all pass |
| Changed-line coverage (branch on) | G2, CI | det | ≥ 90% |
| Mutation score on changed code | G2 | det | ≥ 70% killed |
| Integration tests for every touched boundary | G2, CI | det | all pass |
| E2E for every AC | G2, CI | det | all pass |
| Property-based tests for named invariants | G2 | det | required for R2 |
| Black-box holdout probes | G2/G3 (factory) | det | all pass |
| Per-AC behavioral judgment (blind to tests) | G2/G3 | LLM | every AC passes |
| Diff policy | every import | det | 0 violations |
| Full regression | G2, CI | det | all pass |
| Reviewer findings | G2 | LLM | 0 blocking |

Coverage and mutation scores are signals, not definitions of quality: equivalent mutants exist, and
some changes have no meaningful mutation surface. When a threshold can't honestly be met, an agent
files a **waiver** `{gate, exact scope (mutant IDs / line ranges), reason}`. **The verifier, never the
implementer, adjudicates** it. Waivers appear in the evidence bundle, and the waiver rate is a tracked
metric. In-code suppressions without a matching approved waiver are diff-policy violations, so
gaming the threshold is never cheaper than asking.

### 6.5 Stability: four different measurements

| Measure | What it answers | How | Gate? |
|---|---|---|---|
| Test repeatability | are the tests flaky? | rerun suites k=3 with randomized order and fresh state | yes, per ticket |
| Independent sessions | does behavior hold on a fresh instance? | behavioral checks on freshly provisioned app instances | yes, per ticket |
| Independent data | does it generalize beyond fixture data? | holdout probes with varied/generated data (property-based where possible) | yes, per ticket |
| Factory convergence repeatability | would the factory converge again? (pass^k of the factory itself) | periodic benchmark run over a fixed ticket set, tracked across factory versions | no; it's a factory eval |

---

## 7. Release safety

**Release policy (earned autonomy):**

| Class | Default | Promotion / notes |
|---|---|---|
| R0 (docs, config, copy) | dark | — |
| R1 (normal product changes) | dark | demoted to `hold` automatically after an R1 incident until N clean releases |
| R2 (auth, payments, data, migrations) | **hold** for human release approval | promoted to dark after N (default 10) consecutive clean R2 releases: no rollback, no escaped defect within 14 days, no unexplained holdout-only catch. Demoted on any R2 incident. |
| Always-hold (any tier) | hold | destructive migrations (drop/rename, type narrowing, deleting/overwriting backfills), auth/secret config, deploy/infra config changes |

**Migrations:**

- **Expand/contract** ([parallel change](https://martinfowler.com/bliki/ParallelChange.html)) is
  required: additive schema change first, code switches next, contraction in a later release.
- A migration linter flags destructive operations → always-hold.
- **Compatibility check:** the previous release's test suite runs against the migrated schema. Old
  code must work on the new schema, which is what makes code rollback safe.
- A backup/snapshot is taken before migrating; the restore procedure is exercised in staging.
- Automatic rollback applies only to releases whose migrations are expand-only. Otherwise the factory
  stops the rollout and escalates to the human.

**Rollout:** staging → smoke (ACs tagged `@smoke`) → canary where the target supports it → monitoring
window (error rate, p95 latency, health endpoint, app-level probes; thresholds in `factory.toml`) →
full rollout → health window. `factory pause` is the kill switch for all merges and deploys.

---

## 8. Data model

**Ticket state machine** (illegal transitions are rejected in code; every transition is a ledger event):

```
received → triaged → [needs_info ⇄ answered] → contract_proposed → [needs_info] → approved(frozen)
→ planned → building ⇄ verifying → converged → candidate_verified → merged
→ [awaiting_release_approval] → deploying → done
side states: halted(rate_limit|billing|budget, resume_at) · blocked(human) · rolled_back · abandoned
```

**`tickets/T-0007/ticket.yaml`** (abridged):

```yaml
id: T-0007
type: feature
risk: R1
status: building
source: {channel: cli, received_at: 2026-11-10T14:02:00Z}
request: "users should be able to tag notes and filter by tag"
contract:
  version: 2
  frozen_hash: sha256:…
  migration_impact: expand-only        # none | expand-only | destructive
  criteria:
    - id: AC-1
      ears: "When a signed-in user adds a tag to a note, the system shall display it on the note and in the tag filter."
      layer: e2e
      tests: [tests/acceptance/T-0007/test_tags.py::test_add_tag]
      holdout_probes: 3
oracle_version: sha256:…
```

**`Verdict`** (verifier output; deterministic fields are filled by code, not the LLM):

```json
{"ticket": "T-0007", "round": 2, "sha": "…", "pass": false,
 "gates": {"lint": "pass", "unit": "pass", "coverage": 0.93, "mutation": {"score": 0.64, "status": "fail"}},
 "criteria": [
   {"id": "AC-1", "status": "pass", "evidence": "…"},
   {"id": "AC-3", "status": "fail", "feedback": "…", "holdout_hint": "input class: non-ASCII tag names"}],
 "waivers": [], "proposals": []}
```

**Ledger (SQLite):** `tickets`, `tasks`, `runs` (role, backend, model, effort, tokens and cost metered
from API responses, duration, result subtype, session id, transcript path), `verdicts`, `evidence`, `events`
(append-only), `halts`, `releases`.

---

## 9. Agent roles

Roles reference **tiers**, not models; the tier → model mapping and pricing live in factory config
(§12). Effort is always set explicitly.

| Role | Gate | Tier / effort | Reads | Writes | Output |
|---|---|---|---|---|---|
| Triage | G1 | standard / medium | request, repo (ro), open tickets | — | `TriageResult` |
| Reproducer | G1 | standard / high | repo | repro test | `ReproResult` |
| Spec writer | G1 | frontier / high | request, repo, answers | contract draft | `Contract` |
| Ambiguity critic | G1 | frontier / high, fresh | request, contract, repo | — | `Questions` |
| Test Architect | G1 | frontier / high, fresh | contract, repo | visible tests, holdout | `OracleManifest` |
| Orchestrator | G2 | frontier / xhigh | contract, visible tests, repo map, lessons, ledger | plan | `Plan` |
| Implementer | G2 | standard / high → frontier on escalation | own clone, contract, visible tests | code + tests | `TaskReport` |
| Code reviewer | G2 | standard / high, fresh | task diff, contract, conventions | — | `Findings` |
| Security reviewer (R2) | G2 | frontier / high, fresh | diff, threat notes | — | `Findings` |
| Verifier | G2/G3 | frontier / high, fresh; cross-family for R2 when available | contract, running app, gate results | — | `Verdict` |
| Quality reviewer | G2 | frontier / high, fresh | full ticket diff, repo | — | `Findings` |
| Release | G3 | deterministic; fast tier writes PR prose with no tools | evidence | PR, status, merge, deploy | `ReleaseRecord` |
| Diagnostician | learning | frontier / high | escaped bug + history | lessons, proposals | `Diagnosis` |

Team by tier: **R0** → implementer + gates + verifier. **R1** → + code + quality reviewers. **R2** →
+ security reviewer, frontier implementer, mandatory property tests, cross-family verifier when
available, release hold.

---

## 10. Resource management

- **Tiers** `fast` / `standard` / `frontier` (/ `apex`, off by default).
- **Cheap-first, escalate on observed struggle** (§4.2). The orchestrator proposes; code disposes.
  Before committing to a cascade, measure the simpler alternative: the frontier model at lower effort.
- **Budgets:** a per-ticket cap split across stages, enforced by the host-side model loop from each
  response's `usage` (authoritative: the box never touches it); per-run `max_turns`; the API's task
  budget (§5.3) only as an advisory pacing signal to the model.
- **Halts:** a 429 that outlasts the SDK's retries (`RateLimitError`, `retry-after`) →
  `halted(rate_limit, resets_at)`; a billing or authentication error → halt; budget exhausted →
  `halted(budget)`. The scheduler resumes automatically.
- **Concurrency:** 1 until the sequential factory meets the Phase 6 reliability bar.
- **Metrics (`factory stats`):** cost per accepted ticket, rounds to converge, escalation rate,
  human-intervention rate, escaped-defect rate, holdout catch rate, waiver rate, rollback rate.

---

## 11. Backend abstraction

```python
class AgentBackend(Protocol):
    name: str
    capabilities: frozenset[Capability]   # NATIVE_FS_TOOLS, HOOKS, STRUCTURED_OUTPUT, SESSIONS, SUBAGENTS
    async def run(self, role: RoleSpec, task: AgentTask, ws: WorkerHandle) -> AgentResult: ...

@dataclass
class AgentResult:
    status: Literal["ok", "error", "halted"]
    halt: Halt | None                     # rate_limit(resets_at) | billing | budget | max_turns
    structured_output: dict | None        # validated against role.output_schema
    text: str | None
    usage: Usage                          # metered by the factory from API responses (authoritative)
    session_id: str | None
    transcript_path: Path
    permission_denials: list[dict]
```

| Backend | When | Notes |
|---|---|---|
| `FakeBackend` | Phase 1 | Scripted actions executed in the box through `ToolExecutor`; drives the vertical slice and the red-team suite without LLM spend. |
| `ClaudeAPIBackend` | Phase 2 | Option A (§5.3): host-side loop on the `anthropic` SDK (Messages API, tool runner or own loop); Anthropic's `bash_20250124` and `text_editor_20250728` tools executed in the box through `ToolExecutor`; `output_config.effort`, adaptive thinking, compaction, prompt caching, structured role output; usage metered from responses. Not the Claude Agent SDK. |
| `OpenAICompatibleBackend` | Phase 7 | Llama/Qwen via Ollama or vLLM, with the same host-side loop and box tools. |
| `CodexBackend` / OpenAI Agents | Phase 7+ | Enables cross-family verification. |

Nothing safety- or correctness-critical depends on a backend capability.

---

## 12. Configuration

**Factory config** (`$LOCAL_FACTORY_HOME/config.toml`): backends, tiers, pricing, sandbox.

```toml
[tiers.claude]
fast     = "claude-haiku-5-5"
standard = "claude-sonnet-5-5"
frontier = "claude-opus-5-5"
apex     = "claude-fable-5-2"     # disabled unless enabled explicitly

[pricing]                          # USD per 1M tokens; the factory meters usage from API responses
"claude-haiku-5-5"  = { input = 0.10, output = 0.50, long_context_over = 100_000, long_input = 0.50, long_output = 2.50 }
"claude-sonnet-5-5" = { input = 2.00, output = 10.00 }
"claude-opus-5-5"   = { input = 4.00, output = 20.00 }
"claude-fable-5-2"  = { input = 8.00, output = 40.00 }

[sandbox]                          # config.SandboxConfig; the box always has --network none
image = "local-factory/sandbox:v1"
memory_mb = 2048
command_timeout_s = 300
cache_mounts = [{ source = "/srv/cache/uv", target = "/cache/uv" }]   # read-only dependency caches
```

**Target contract** (`factory.toml` in the target repo; protected):

```toml
[project]
name = "notes-app"

[commands]                      # scripts/verify composes these; CI calls scripts/verify
setup       = "uv sync && uv run playwright install chromium"
lint        = "uv run ruff check . && uv run ruff format --check ."
typecheck   = "uv run mypy src"
unit        = "uv run pytest tests/unit -q -p randomly"
integration = "uv run pytest tests/integration -q -p randomly"
e2e         = "uv run pytest tests/e2e tests/acceptance -q"
mutation    = "uv run mutmut run --paths-to-mutate {changed_files}"
serve       = "uv run uvicorn app.main:app --port {port}"
migrate     = "uv run alembic upgrade head"

[paths]
protected = ["tests/acceptance/**", "factory.toml", ".github/workflows/**", "scripts/verify", "migrations/**"]

[policy]
changed_line_coverage = 0.90
mutation_score = 0.70
flake_runs = 3

[release]
policy = { R0 = "dark", R1 = "dark", R2 = "hold" }
r2_promotion_clean_releases = 10
always_hold = ["destructive_migration", "auth_config", "deploy_config"]
health = { endpoint = "/healthz", window_minutes = 15, max_error_rate = 0.01, max_p95_ms = 500 }
```

Plus `AGENTS.md` (a ~100-line map into `docs/`), `docs/adr/`, `scripts/verify`, and
`.github/workflows/ci.yml`.

**First target:** a small multi-user notes app (FastAPI + SQLite + HTMX + Alembic), tested with pytest,
pytest-playwright, Hypothesis, mutmut and diff-cover. It's Python-only (no Node needed) yet covers
unit, integration, browser e2e and migrations.

---

## 13. Repository layout (this repo, v1)

```
local-factory/
├── pyproject.toml                 uv-managed; console script `factory`
├── src/local_factory/
│   ├── cli.py                     project/intake/questions/answer/show/approve/run/status/logs/stats/pause/verify-ticket
│   ├── config.py                  factory + project config (pydantic)
│   ├── models.py                  Ticket, Criterion, Task, Plan, Verdict, Evidence, Waiver, Halt
│   ├── ledger/                    SQLite store, state machine, events, accounting
│   ├── intake/                    channels/, triage, reproduce, spec, clarify
│   ├── oracle/                    test architect driver, holdout store, red-first, change control, redaction, burn/rotate
│   ├── orchestration/             planner, DAG validation, scheduler, escalation, convergence loop
│   ├── gitops/                    bare repo, bundle export/import + validation, integration, candidate squash
│   ├── sandbox/                   the box: specs + hardening, docker runtime, workspaces, lf-entry, SandboxExecutor; app/probe containers (Phase 3)
│   ├── verification/              gate runner, verify-ticket, mutation/coverage/flake, waivers, min-aggregation
│   ├── evidence/                  evidence bundle, hashing/HMAC, GitHub commit status
│   ├── release/                   GitHub API (PR, merge with sha), release policy, migrations checks, deployers/, rollback
│   ├── agents/                    backend.py, tools.py (ToolExecutor), local.py, fake.py, claude_api.py, roles/*.md
│   ├── policy/                    diff policy, role permissions, hook bridge
│   └── memory/                    per-ticket log, lessons, ADR helpers
├── docker/                        sandbox image (1b); app-runner, probe images (Phase 3)
├── templates/fastapi-web/         target scaffold
├── tests/{unit,integration,e2e,redteam}/ and tests/fixtures/cli-target/
└── docs/PLAN.md
```

Per-project state, outside any target repo:
`$LOCAL_FACTORY_HOME/projects/<name>/{repo.git, ledger.db, tickets/, oracle/, runs/, evidence/}`.

---

## 14. Implementation phases

The sequence is unchanged; the exit checks are stricter. The factory's own code follows the test
policy it imposes.

| Phase | Scope | Exit check |
|---|---|---|
| **0. Reset** | Tag `legacy-v0`; remove `factory/`, `greet.py`, `slugify.py` (greet/slugify → `tests/fixtures/`); uv skeleton; ruff + mypy + pytest; CI for this repo; README rewrite. | `uv run factory --help`; CI green |
| **1a. Control plane** | models, ledger + state machine, config, `AgentBackend` + `FakeBackend`, bare repo + bundle export/import + validation, gate runner, diff policy, min-aggregation, evidence binding. | unit + integration tests green, including negative bundle tests |
| **1b. Boundary — vertical slice** | the box: hardened, network-less containers for every worker action (`ToolExecutor`), worker git and export, gates (one fresh tree per gate); tmpfs task workspaces; the sandboxed slice; red-team suite v1. Spec: `docs/specs/phase-1b.md`. | **one ticket, one worker, one isolated container boundary, one commit, bundle import, one deterministic verification, one integration, evidence bound to the SHA (incl. the image digest)**, all with FakeBackend and $0 LLM spend; red-team suite v1: every check holds |
| **2. Claude + Gate 1** | `ClaudeAPIBackend` (option A, §5.3: host-side loop on the `anthropic` SDK, Anthropic's bash + text-editor tools executed in the box) with the option-A red-team items and an opt-in `live` quality check; intake roles; holdout store; red-first; CLI intake/questions/answer/show/approve; notes-app template. | a prompt becomes an approved, frozen contract with red-verified visible + black-box holdout tests; a real model completes a fixture task through the box with the API key never in the box |
| **3. Gate 2, single lane** *(highest priority)* | orchestrator, sequential implementer tasks, `verify_fast` + Stop gate, verify-ticket, redaction + leak scan, burn/rotate, escalation + stop rules, quality reviewer. | on a fixed benchmark of ≥5 notes-app tickets: converge rate recorded, ≥1 ticket needs >1 round and converges, zero holdout literals in any feedback, red-team still green. This proves **convergence**, not just execution. |
| **4. Verification depth** | diff-scoped mutation, changed-line coverage, waivers, flake/session/data measures, blind per-AC behavioral check, Hypothesis, security reviewer. Metrics introduced one at a time with baselines. | a planted vacuous test and a planted wrong-expectation test are both caught; waiver flow audited end to end |
| **5. Gate 3** | candidate squash + verify-ticket@SHA, commit statuses + branch protection, sha-pinned merge + tree-hash check, release policy (R2 hold, always-hold), migration linter + compatibility check + backup, canary + monitoring + auto-rollback, `factory pause`. **Needs the deploy-target decision.** | R1 ticket: prompt → prod with no human action after approval; R2 ticket holds; moving `main` during release forces re-verification (a stale pass is rejected); destructive migration is held; a broken deploy rolls back automatically |
| **6. Parallelism** | parallel tasks/tickets, concurrency limits, data-driven escalation tuning, `factory stats`. **Entry bar:** benchmark converge rate and zero isolation incidents sustained over Phases 3–5. | two independent tickets in parallel merge cleanly; cost per accepted ticket reported |
| **7. Second backend + learning** | `OpenAICompatibleBackend`, cross-family verifier (R2 first), diagnostician, lessons → lint-rule/prompt proposals (factory self-changes need human approval). | the same benchmark converges with a non-Claude implementer; an injected escaped bug yields a regression scenario + lesson |
| **8. Channels** *(deferred until the factory is reliable)* | Slack / email / WhatsApp adapters; automatic intake from workspace signals. | a Slack message becomes a ticket, its questions are answered in-thread, and it ships |

**Status (2026-11-11):** Phases 0 and 1a are complete. 1a also delivered the *host* vertical slice
(`pipeline/host_slice.py`, spec `docs/specs/phase-1a-slice.md`): one ticket through FakeBackend,
bundle import, diff policy, integration, verification, squash candidate and signed evidence bound to
the candidate SHA, with worker git and gates still on the host. Phase 1b is complete too (spec
`docs/specs/phase-1b.md`): the same slice runs behind the box (`pipeline/sandboxed_slice.py`), with
every worker action, the worker's git, the export and every gate in network-less containers, the
evidence bound to the sandbox image digest, and red-team suite v1 green; 2081 tests, 100% branch
coverage on Python 3.12 and 3.14 (commit `04bc423`, CI green). App/probe containers moved to Phase 3.
Agent placement was decided the same day: **option A** (§5.3); the broker and egress proxy are
dropped. **Next: Phase 2**: write `docs/specs/phase-2.md` and its protected acceptance tests from
§5.3 and this table's Phase 2 row (plus its §14.1 rows), then the agent team builds it.

### 14.1 Carried forward from the Phase 1a review

Known gaps found while building 1a, each assigned to the phase that owns the fix. Details and
rationale: the "Follow-ups" section of `docs/specs/phase-1a.md` and the amendments in
`docs/specs/phase-1a-slice.md`. Each phase's spec must pick up its rows and its exit check must
cover them.

| Phase | Gap | Resolution |
|---|---|---|
| 1b | Worker-side git (`gitops/worker_side.py`) and gate commands run on the host | Run them inside the hardened worker / gate containers; the host only sees bundles and reports (invariant 7) |
| 1b | A worker can still crash the pipeline: an export failure after `has_task_commits` succeeded (e.g. corrupted objects) propagates | Export runs in a disposable container; any failure there → task `REJECTED`, reason `worker_git_error` |
| 1b | Bundle size is not capped at import | Export streams the bundle out of the box under a hard cap (`max_bundle_bytes`); import checks `BundleLimits.max_bundle_bytes` before copying |
| 1b | `RunCommand` (FakeBackend) and agent commands have no timeout | Per-command and per-run wall-clock limits enforced by the container runtime |
| 1b | Fresh per-gate `HOME` means cold tool caches (uv, pip, npm) | Mount read-only shared caches into gate containers explicitly |
| 1b | Host code could follow a worker-produced symlink | Worker trees never land on the host: the box exports bundles, and gates and the verifier materialize trees inside the box |
| 1b | Candidate-conflict paths are recorded only as transition text | Add structured `data` to `Ledger.transition` (as `update_task` already has) |
| 2 | How a model-driven agent reaches its model | **Decided 2026-11-11: option A** (§5.3). Build `ClaudeAPIBackend`: host-side loop, box tools through `ToolExecutor`, no broker or proxy; add the option-A red-team items (§5.6) |
| 3 | Black-box holdout probes need app + probe containers (§5.4), and the redaction red-team item | Build them with the holdout store and verify-ticket |
| 2 | Unsafe symlinks already in a target's base fail every task closed (`symlink_escape`) | `factory project add` detects and reports them when a repo is adopted |
| 2 | `DiffPolicy.test_globs` are Python-only and not configurable | `[paths] test_globs` in `factory.toml` (needed for the notes-app JS/TS code) |
| 4 | Gate-config edits that *remove* lines are invisible to the diff policy (it sees added lines only) | The factory runs coverage and mutation with its own config and `factory.toml [policy]` thresholds, so target-side config cannot weaken a gate |
| 4 | Binary files are opaque to the diff policy | Flag new or modified binaries under source/test paths for verifier review |

---

## 15. Open decisions

| Decision | Needed before | Recommendation |
|---|---|---|
| Billing mode (API key vs Claude subscription) | Phase 1b (broker design) | **Decided (2026-11-10): API key.** With option A the key stays in the host process and no broker is needed. Anthropic does not allow third-party products to offer claude.ai login unless approved, so a subscription is not an option for users of this project either (§5.3). |
| Agent placement (§5.3) | Phase 2 | **Decided (2026-11-11): option A**, brain on the host, hands in the box. Model loop in the factory process on the `anthropic` SDK; Anthropic's bash + text-editor tools execute in the network-less box through `ToolExecutor`; no broker, no egress proxy; the Claude Agent SDK is not used on the host. Reasons, rejected variant, accepted costs and API facts: §5.3. Option B (agent in the box behind a broker + egress proxy) is rejected; reopen only on new evidence, e.g. the Phase 2 quality measurement showing a large gap. |
| Deploy target for the notes app | Phase 5 | Docker on Fly.io (releases make rollback simple; canary by machine); alternative: a VPS with docker compose + Caddy. |
| R2 promotion thresholds | Phase 5 | Start at 10 clean releases / 14-day escaped-defect window; revisit with data. |
| Evidence signing key | Phase 1a | **Done in 1a:** a local key file in `$LOCAL_FACTORY_HOME` (0600, created if missing, never silently replaced); rotate on demand. Enforcement relies on commit statuses, not on the key. |
| GitHub auth | Phase 5 | A fine-grained PAT scoped to target repos (statuses, PRs, contents), release stage only. |
| Holdout lifecycle after merge | Phase 3 | Promote to the visible regression suite on merge; fresh probes per ticket. |

---

## Sources

- StrongDM — [Software Factory](https://factory.strongdm.ai/), [techniques](https://factory.strongdm.ai/techniques); Simon Willison — [notes](https://simonwillison.net/2026/Feb/7/software-factory/)
- Anthropic — [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents), [Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps), [Building a C compiler with parallel Claudes](https://www.anthropic.com/engineering/building-c-compiler), [Demystifying evals for AI agents](https://www.anthropic.com/research/demystifying-evals-for-ai-agents), [Agentic property-based testing](https://red.anthropic.com/2026/property-based-testing)
- OpenAI — [Harness engineering](https://openai.com/index/harness-engineering/)
- Agent patterns — [Acceptance scenario lifecycle](https://agentpatterns.ai/factory-engineering/acceptance-scenario-lifecycle/), [Factory engineering index](https://agentpatterns.ai/factory-engineering/), [Mutation testing as a quality gate](https://www.agentpatterns.ai/verification/mutation-testing-quality-gate/), [Test homogenization trap](https://agentpatterns.ai/anti-patterns/test-homogenization-trap/)
- Reward hacking — [EvilGenie](https://arxiv.org/html/2511.21654v1); [AI-written tests vs historical bugs](https://www.testmanagement.com/blog/2026/08/ai-written-tests-evidence/)
- Ambiguity — [Ambig-SWE (ICLR 2026)](https://arxiv.org/html/2502.13069v3), [Ask or Assume?](https://www.alphaxiv.org/abs/2603.26233), [feasibility gate](https://dev.to/danii1/teaching-an-ai-agent-to-say-no-the-feasibility-gate-6fo)
- Routing — [Faros: is routing enough?](https://www.faros.ai/blog/is-intelligent-model-routing-enough), [SWE-Router escalation](https://codex.danielvaughan.com/2026/07/10/swe-router-trajectory-based-model-routing-codex-cli-partial-exploration-cost-escalation/)
- Judging bias — [Chain-of-Models cross-model auditing](https://arxiv.org/pdf/2607.28636)
- Loops and orchestration — [Ralph loop (Thoughtworks Radar)](https://www.thoughtworks.com/radar/techniques/ralph-loop), [Gas Town / Beads](https://softwareengineeringdaily.com/2026/02/12/gas-town-beads-and-the-rise-of-agentic-development-with-steve-yegge/)
- Sandboxing — [Worktree/symlink escape fixes](https://www.digitalapplied.com/blog/agent-sandbox-escapes-worktree-symlink-command-filters-2026), [Egress allowlists](https://startdebugging.net/2026/07/how-to-lock-down-a-coding-agents-network-egress-with-a-strict-host-allowlist/)
- Git — [git-worktree](https://git-scm.com/docs/git-worktree), [git-bundle](https://git-scm.com/docs/git-bundle)
- Migrations — [Parallel change (expand/contract)](https://martinfowler.com/bliki/ParallelChange.html)
- Claude API (option A, §5.3) — [Bash tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool), [Text editor tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool), [Tool runner](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-runner), [Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction), [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), [Task budgets](https://platform.claude.com/docs/en/build-with-claude/task-budgets); pricing — [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- Claude Agent SDK (rejected for the host, §5.3) — [Overview: login rule, license and terms](https://code.claude.com/docs/en/agent-sdk/overview), [Python reference](https://code.claude.com/docs/en/agent-sdk/python)
