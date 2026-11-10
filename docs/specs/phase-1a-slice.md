# Phase 1a — host vertical slice (integration)

Wires the Phase 1a modules into one deterministic pass: **one ticket, one task, one worker, one
commit, one bundle import, policy check, integration, verification, candidate, evidence bound to the
candidate SHA.** The worker runs on the host in a temp dir here; Phase 1b moves it into a hardened
container without changing this module's interface.

Owner: the **integrator**. Owns `src/local_factory/pipeline/**` and `tests/unit/pipeline/**`,
`tests/integration/pipeline/**`. Acceptance: `tests/acceptance/test_host_slice.py` (protected).

## API — `local_factory/pipeline/host_slice.py`

```python
class VerifierReport(BaseModel):            # the verifier role's structured output
    criteria: list[CriterionResult]

IMPLEMENTER_ROLE: RoleSpec                  # name "implementer", tier STANDARD, no output schema
VERIFIER_ROLE: RoleSpec                     # name "verifier", tier FRONTIER, output_schema=VerifierReport

SliceStatus = Literal["candidate_verified", "rejected", "failed_verification", "halted", "agent_error"]

@dataclass(frozen=True)
class SliceOutcome:
    status: SliceStatus
    ticket: Ticket                           # fresh snapshot at the end
    task: Task
    rejection: str | None = None             # bundle rejection code | "policy" | "conflict" | "candidate_conflict"
    violations: tuple[Violation, ...] = ()
    verdict: Verdict | None = None           # the last verdict recorded
    candidate_sha: str | None = None
    evidence: Evidence | None = None
    signature: str | None = None

async def run_host_slice(*, ledger: Ledger, repo: BareRepo, project: ProjectConfig,
                         backend: AgentBackend, ticket_id: str, work_root: Path, key: bytes,
                         main_ref: str = "refs/heads/main",
                         clock: Callable[[], datetime] = utc_now) -> SliceOutcome
```

## Flow (the ticket must be APPROVED, i.e. its contract is frozen)

1. `PLANNED`; create one task (goal = contract summary, role implementer, tier STANDARD,
   `criteria_served` = all AC ids); `BUILDING`.
2. Ticket ref `refs/heads/factory/<ticket>` is created at `main` if missing; the task's base is its head.
3. Base bundle → `init_from_bundle` into `work_root/<ticket>/task-<n>/work`; task `RUNNING` + `base_sha`.
4. Run the implementer via the backend; record a `RunRecord` (backend name, model `"fake"` if the
   backend is fake, else the configured model, tier, usage, status).
   - `halted` → `ledger.halt(...)`, task `HALTED`, outcome `halted`.
   - `error` → task `REJECTED`, outcome `agent_error` (ticket stays `BUILDING`).
5. Harness commit: `commit_all(work, "<task id>: <goal>")` (the harness always commits whatever the
   agent left in the worktree).
6. Export the task bundle; task `SUBMITTED`. If the worker has no commits beyond base, skip export and
   reject with `no_changes`.
7. `import_task_bundle(...)` into `refs/factory/tasks/<ticket>/task-<n>`. `BundleRejected` → task
   `REJECTED`, outcome `rejected` with `rejection = code`.
8. `check_changes(result.changes, DiffPolicy.from_project(project))`. Violations → task `REJECTED`,
   outcome `rejected`, `rejection="policy"`, violations attached. The ticket ref is not moved.
9. `merge_into(ticket ref, task head)`; conflict → `rejected`/`"conflict"`. Else task `ACCEPTED` + `head_sha`.
10. `VERIFYING`. Verify the ticket head (§ Verification below) → verdict round 1. Not passed →
    `BUILDING`, outcome `failed_verification`.
11. `CONVERGED`. `squash_onto(base=main, head=ticket head, message="<ticket>: <title>")` → candidate;
    conflict → `BUILDING`, `rejected`/`"candidate_conflict"`.
12. Verify the **candidate** → verdict round 2 (`sha` = candidate). Not passed → `BUILDING`,
    `failed_verification`.
13. `build_evidence(...)` from the candidate verdict with `tree_hash = repo.tree_hash(candidate)`, the
    ticket's frozen `contract_hash` and `oracle_version`, `factory_version = local_factory.__version__`,
    `models = {role: model}`; `check_binding(...)` must return `[]` (else raise — it's a factory bug);
    `sign`, `ledger.record_evidence(...)`; `CANDIDATE_VERIFIED`. Outcome `candidate_verified`.

The slice never moves `main` and never pushes anywhere.

## Verification (used by steps 10 and 12)

`export_tree(sha, <fresh dir>)` → `run_gates(gates_from_project(project), cwd=that dir)` → run the
verifier role via the backend in that dir (task_id `None`, so FakeBackend scripts key on
`"verifier"`) and record its run. Its `VerifierReport.criteria` become the criterion results; if the
verifier errors, the criteria list is empty (so min-aggregation fails the verdict with
`criteria:none`). `make_verdict(...)` → `ledger.record_verdict(...)`.

## Amendments after integration review (round 2)

- **Persist every rejection in the ledger** (invariant 1). Each task transition to `REJECTED` or
  `HALTED` goes through `update_task(..., reason=<code>, data=...)`: `reason` is the bundle code,
  `"policy"`, `"conflict"`, `"candidate_conflict"`, `"agent_error"`, `"worker_git_error"` or the halt
  reason; `data` carries `violations` (rule/path/line/message), `conflicts` (paths) or `error` text.
  `RunRecord.error` carries the backend's error text for `status="error"` runs.
- **`no_changes` is decided explicitly** with `worker_side.has_task_commits(work, base_sha=...)`
  before exporting; an export failure after that is a factory bug and propagates.
- **A worker that breaks its own repository** (any `GitError` from a `worker_side` call) → task
  `REJECTED` with reason `"worker_git_error"`, outcome `rejected`; the ticket stays `BUILDING`.
- Accepted resolutions from the integrator's report: BUILDING tickets get a new task based on the
  ticket head; `factory_config` parameter for model lookup; verifier halts are halts (no verdict);
  verifier criteria are reconciled against the contract fail-closed (missing → INCONCLUSIVE,
  duplicates → first non-pass, unknown ids dropped); rounds = max recorded + 1; verify dirs are kept.
- Accepted (round 2): `worker_git_error` covers worker-side calls made after the agent had the repo
  (`commit_all`, `has_task_commits`); `init_from_bundle` failures are factory errors. A candidate
  conflict does not reject the (cleanly merged) task; it is recorded on the ticket's transition back
  to `BUILDING`.
- **Follow-up (Phase 1b):** an export failure after `has_task_commits` succeeded is worker-caused
  (e.g. corrupted objects) and should be `worker_git_error`, not a propagated exception — a worker
  must never be able to crash the pipeline. In 1b, export runs in a disposable container, and any
  failure there maps to `worker_git_error`. `Ledger.transition` may also need a `data` field so
  candidate-conflict paths are structured rather than text.
