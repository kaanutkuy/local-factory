# Phase 1a spec — control plane

Goal: the deterministic core of the factory, fully tested, no LLM and no containers. Phase 1b moves
worker-side execution into hardened containers without changing these interfaces.

Shared contract (orchestrator-owned, already written): `local_factory/models.py`,
`local_factory/hashing.py`, `local_factory/config.py`. Acceptance tests (orchestrator-owned,
protected): `tests/acceptance/`. They are the definition of done for each module, together with
`scripts/verify` (ruff, format, mypy --strict, ≥90% branch coverage).

## Ownership

| Agent | Owns (create/edit only these) | Acceptance tests |
|---|---|---|
| **ledger** | `src/local_factory/ledger/**`, `tests/unit/ledger/**`, `tests/unit/test_models.py`, `tests/unit/test_config.py`, `tests/unit/test_hashing.py` | `test_ledger.py` |
| **gitops** | `src/local_factory/gitops/**`, `tests/unit/gitops/**`, `tests/integration/gitops/**` | `test_gitops.py` |
| **verification** | `src/local_factory/verification/**`, `src/local_factory/policy/**`, `tests/unit/verification/**`, `tests/unit/policy/**` | `test_verification.py`, `test_diff_policy.py` |
| **agents** | `src/local_factory/agents/**`, `src/local_factory/evidence/**`, `tests/unit/agents/**`, `tests/unit/evidence/**` | `test_agents.py`, `test_evidence.py` |

Test subdirectories need an `__init__.py` (mypy resolves tests as packages).

---

## ledger — `local_factory.ledger`

`ledger/state_machine.py`

```python
TERMINAL: frozenset[TicketStatus]                      # DONE, ROLLED_BACK, ABANDONED
TRANSITIONS: Mapping[TicketStatus, frozenset[TicketStatus]]
class IllegalTransition(Exception): ...
def allowed_targets(status: TicketStatus) -> frozenset[TicketStatus]
def check_transition(src: TicketStatus, dst: TicketStatus) -> None   # raises IllegalTransition
```

Transition table (HALTED is *not* in it — it is entered via `halt()` and left via `resume()`):

| From | To |
|---|---|
| received | triaged, needs_info, abandoned |
| triaged | contract_proposed, needs_info, abandoned |
| needs_info | triaged, contract_proposed, abandoned |
| contract_proposed | approved, needs_info, abandoned |
| approved | planned, abandoned |
| planned | building, abandoned |
| building | verifying, blocked, abandoned |
| verifying | building, converged, blocked, abandoned |
| converged | candidate_verified, building, abandoned |
| candidate_verified | merged, converged, abandoned |
| merged | awaiting_release_approval, deploying, done |
| awaiting_release_approval | deploying |
| deploying | done, rolled_back |
| blocked | building, contract_proposed, abandoned |
| done, rolled_back, abandoned | — (terminal) |

`ledger/errors.py` — `LedgerError` (base), `IllegalTransition`, `TicketNotFound`, `TaskNotFound`,
`ContractFrozen`, all subclasses of `LedgerError`. `local_factory.ledger` re-exports `Ledger` and all
five errors.

`ledger/store.py` — `class Ledger` (SQLite, WAL, one connection; context manager; `close()`).
Constructor `Ledger(path: Path, *, clock: Callable[[], datetime] = utc_now)` creates the schema if
missing and reopens existing ledgers. Every mutating method is one transaction, calls the clock for
its timestamp, and appends ≥1 event. Invalid operations raise `LedgerError` (or the specific
subclass): halting a halted or terminal ticket, resuming a ticket that is not halted, freezing
without the preconditions below.

```python
create_ticket(*, title, type: TicketType, risk: RiskTier, request: str, actor: str) -> Ticket
    # ids T-0001, T-0002, … monotonic across reopen; status RECEIVED; event "created"
get_ticket(ticket_id) -> Ticket
list_tickets(status: TicketStatus | None = None) -> list[Ticket]          # ordered by id
transition(ticket_id, to: TicketStatus, *, actor, reason="") -> Ticket
    # validates via check_transition; to=HALTED is rejected (IllegalTransition); also rejected
    # when the ticket is HALTED (resume first); event "transition" with from/to status
propose_contract(ticket_id, contract: Contract, *, actor) -> Ticket       # event "contract_proposed"
    # raises ContractFrozen once frozen; stores the contract, does not change status
set_oracle_version(ticket_id, version: str, *, actor) -> Ticket           # ContractFrozen once frozen
freeze_contract(ticket_id, *, actor) -> Ticket
    # requires status CONTRACT_PROPOSED, a contract and an oracle version (else LedgerError);
    # sets contract_hash = hashing.content_hash(contract); transitions to APPROVED;
    # events "contract_frozen" then "transition"
halt(ticket_id, halt: Halt, *, actor) -> Ticket
    # any non-terminal, non-halted status -> HALTED; remembers the prior status; event "halted"
resume(ticket_id, *, actor) -> Ticket       # HALTED -> exactly the prior status; event "resumed"
current_halt(ticket_id) -> Halt | None
events(ticket_id: str | None = None) -> list[Event]                       # ordered by seq
create_task(ticket_id, *, goal, role, tier: Tier, criteria_served=(), depends_on=(), actor) -> Task
    # ids "<ticket>/task-1", "-2", … per ticket; status PENDING; event "task_created"
get_task(task_id) -> Task ; list_tasks(ticket_id) -> list[Task]
update_task(task_id, *, actor, status=None, base_sha=None, head_sha=None,
            reason: str = "", data: Mapping[str, JsonValue] | None = None) -> Task  # "task_updated"
    # at least one of status/base_sha/head_sha must change (else LedgerError); `reason` and
    # `data` are stored in the event's data as {"reason": ..., "detail": {...}} so rejections,
    # violations and conflicts are reconstructable from the ledger alone (invariant 1)
record_run(run: RunRecord) -> None ; runs(ticket_id) -> list[RunRecord]   # event "run_recorded"
ticket_cost_usd(ticket_id) -> float                                       # sum of run usage cost
record_verdict(verdict: Verdict) -> None ; verdicts(ticket_id) -> list[Verdict]  # "verdict_recorded"
record_evidence(evidence: Evidence, *, evidence_hash: str, signature: str) -> None  # "evidence_recorded"
evidence_for(ticket_id, sha) -> Evidence | None                           # latest for that sha
evidence_record(ticket_id, sha) -> EvidenceRecord | None                 # latest, with its proofs
    # EvidenceRecord: frozen dataclass (evidence: Evidence, evidence_hash: str, signature: str),
    # defined in ledger/store.py and re-exported; release-time checks re-verify the signature
```

The `events` table is append-only **in the database**: SQLite triggers abort any `UPDATE` or
`DELETE` on it. Models are stored as JSON (`model_dump_json`) and validated on read.

Also: unit tests for `models.py`, `config.py`, `hashing.py` (orchestrator wrote them; you test them).

---

## gitops — `local_factory.gitops`

`gitops/_git.py`

```python
FACTORY_NAME = "local-factory"; FACTORY_EMAIL = "local-factory@localhost"
class GitError(Exception): args: tuple[str, ...]; returncode: int; stderr: str
def git_env(extra: Mapping[str, str] | None = None) -> dict[str, str]
    # minimal env: PATH, LANG=C, HOME=<a fresh empty temp dir or os.devnull-safe value>,
    # GIT_CONFIG_NOSYSTEM=1, GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT=0,
    # GIT_AUTHOR_/COMMITTER_ NAME/EMAIL = factory identity; NO inherited GIT_* variables
def run_git(args: Sequence[str], *, git_dir: Path | None = None, cwd: Path | None = None,
            input: bytes | None = None, check: bool = True) -> subprocess.CompletedProcess[bytes]
    # always prefixes: -c core.hooksPath=/dev/null -c core.fsmonitor=false
    #                  -c protocol.allow=never -c protocol.file.allow=always
```

`gitops/bare.py` — `class BareRepo` (`path: Path`). Every host-side operation on factory-owned repos.

```python
BareRepo.init(path) -> BareRepo      # bare; hardened config: core.hooksPath=/dev/null,
                                     # core.fsmonitor=false, fetch/receive/transfer.fsckObjects=true,
                                     # gc.auto=0
BareRepo.open(path) -> BareRepo      # GitError if not a bare repository
fetch_from(source: Path | str, refspecs: Sequence[str]) -> None   # trusted sources only (seeding)
rev_parse(rev) -> str                # full sha; GitError if unknown
ref_exists(ref) -> bool ; update_ref(ref, sha, *, old_sha=None) -> None ; delete_ref(ref) -> None
is_ancestor(ancestor, descendant) -> bool
tree_hash(rev) -> str ; read_file(rev, path) -> bytes
list_commits(base, head) -> list[str]                 # base..head, oldest first
diff(base, head) -> list[ChangedFile]
    # rename detection on; mode/old_mode; size of new blob; is_binary; symlink_target for
    # MODE_SYMLINK; added_lines (1-based new-file line numbers, text without the trailing
    # newline) for non-binary, non-symlink A/M/R files. Deletions: path = old path, mode None.
    # Sorted by path.
export_tree(rev, dest: Path) -> None                  # materialize files (no .git) into empty dest
```

`gitops/worker_side.py` — the only code that runs git inside a worker directory. In production it
executes inside the sandbox; in 1a tests it runs in `tmp_path`.

```python
TASK_BRANCH = "task"
init_from_bundle(bundle: Path, ref: str, dest: Path) -> None
    # git init dest; fetch <bundle> <ref>:refs/heads/task; checkout task
commit_all(work: Path, message: str) -> str | None     # add -A + commit; None if nothing to commit
has_task_commits(work: Path, *, base_sha: str) -> bool  # task branch has commits not in base_sha
export_task_bundle(work: Path, out: Path, *, base_sha: str) -> None   # refs/heads/task ^base_sha
```

`gitops/bundles.py` — host side.

```python
@dataclass(frozen=True)
class BundleLimits: max_commits: int = 50; max_file_bytes: int = 5_000_000; max_total_bytes: int = 50_000_000
class BundleRejected(Exception): code: str; detail: str
@dataclass(frozen=True)
class ImportResult: head_sha: str; commits: tuple[str, ...]; changes: tuple[ChangedFile, ...]
export_base_bundle(repo: BareRepo, *, sha: str, out: Path) -> str
    # bundle with exactly one temporary ref pointing at sha; returns that ref name (for
    # init_from_bundle); the temporary ref is removed from the repo afterwards
import_task_bundle(repo, bundle: Path, *, base_sha: str, dest_ref: str,
                   limits: BundleLimits = BundleLimits()) -> ImportResult
```

`import_task_bundle` reads only the bundle file. Rejection codes (checked in this order; on any
rejection no ref is left behind):

| code | condition |
|---|---|
| `invalid_bundle` | not a readable bundle (`list-heads` fails) |
| `unexpected_refs` | bundle heads ≠ exactly `refs/heads/task` |
| `invalid_bundle` | `git bundle verify` fails (corrupt, or prerequisites missing) |
| `fetch_failed` | fetch into a quarantine ref fails (incl. fsck errors) |
| `base_not_ancestor` | `base_sha` is not an ancestor of the head |
| `no_changes` | head == base |
| `too_many_commits` | more than `max_commits` in base..head |
| `gitlink` | any entry with mode 160000 |
| `symlink_escape` | symlink target absolute, or resolving outside the repo root |
| `unsafe_path` | a path with a `..` component, a leading `/`, or a `.git` component (any case) |
| `file_too_large` / `too_large` | a blob over `max_file_bytes` / total new bytes over `max_total_bytes` |

On success: fetched into `refs/factory/quarantine/<uuid>`, validated, then `dest_ref` is set to the
head and the quarantine ref deleted.

`gitops/integrate.py`

```python
@dataclass(frozen=True)
class MergeOutcome: sha: str | None; conflicts: tuple[str, ...]   # sha None iff conflicts
merge_into(repo, *, target_ref: str, source: str, message: str) -> MergeOutcome
    # fast-forward if target is an ancestor of source; else merge-tree --write-tree, commit-tree
    # with parents (target, source); update target_ref with old-value CAS. Conflicts: no ref change.
squash_onto(repo, *, base: str, head: str, message: str) -> MergeOutcome
    # candidate commit: tree of merge(base, head), single parent base; updates no refs
```

---

## verification + policy — `local_factory.verification`, `local_factory.policy`

`verification/gates.py`

```python
DEFAULT_GATE_ORDER = ("lint", "typecheck", "unit", "integration", "e2e")
@dataclass(frozen=True)
class GateSpec: name: str; command: str; timeout_s: float = 900.0
gates_from_project(config: ProjectConfig, names: Sequence[str] = DEFAULT_GATE_ORDER) -> list[GateSpec]
    # in `names` order; names without a configured command are omitted
run_gate(spec, *, cwd: Path, env: Mapping[str, str] | None = None, tail_bytes: int = 16_000) -> GateResult
    # bash -c <command> in its own process group; exit 0 -> PASS, non-zero -> FAIL (exit_code set);
    # timeout -> ERROR (whole group killed); output_tail = last tail_bytes of stdout+stderr.
    # Environment is MINIMAL, never os.environ: PATH, LANG=C.UTF-8, HOME and TMPDIR = a fresh
    # per-gate temp dir (removed afterwards), then `env` overrides on top. Gates execute
    # implementer-written code, so host secrets must not be visible (invariant 6).
run_gates(specs, *, cwd, env=None, fail_fast=False) -> list[GateResult]
    # fail_fast: after the first non-PASS, remaining gates are SKIPPED (not executed)
```

`verification/aggregate.py` — min-aggregation.

```python
aggregate(gates, criteria, waivers=(), *, require_criteria: bool = True) -> AggregateResult
make_verdict(*, ticket_id, round, sha, gates, criteria, waivers=()) -> Verdict
```

Rules: no gates → `"gates:none"`; `require_criteria` and no criteria → `"criteria:none"`; gate
FAIL/ERROR → `"gate:<name>"`; SKIPPED → `"gate:<name>:skipped"`; WAIVED without an effective waiver
for that gate (`waiver.gate == name and waiver.is_effective`) → `"gate:<name>:unwaived"`; criterion
FAIL/INCONCLUSIVE → `"criterion:<id>"`; duplicate gate names or criterion ids → `ValueError`.
`failing` order: none-markers, gates in input order, criteria in input order. `passed = not failing`.

`policy/diff_policy.py`

```python
@dataclass(frozen=True)
class DiffPolicy:
    protected: tuple[str, ...]
    test_globs: tuple[str, ...] = ("tests/**", "**/test_*.py", "**/*_test.py")
    max_file_bytes: int = 1_000_000
    @classmethod
    def from_project(cls, config: ProjectConfig) -> DiffPolicy
glob_match(path: str, pattern: str) -> bool
    # "/"-separated; "**" matches zero or more whole segments; "*", "?" and "[...]" within a segment
check_changes(changes: Sequence[ChangedFile], policy: DiffPolicy, waivers: Sequence[Waiver] = ()) -> list[Violation]
```

| rule | condition |
|---|---|
| `protected_path` | any change whose `path` or `old_path` matches a protected glob |
| `test_deleted` | a deleted file matching `test_globs`, or a rename from a test path to a non-test path |
| `skip_marker` | in a **test file** (matches `test_globs`, or is named `conftest.py`): added line containing `pytest.mark.skip`, `pytest.mark.xfail`, `pytest.skip(`, `importorskip(`, `unittest.skip`, `skipTest(`, `SkipTest`, `it.skip(`, `describe.skip(`, `test.skip(`, `.only(`, `xit(`, `xdescribe(`, `fit(`, `fdescribe(` (identifier-like markers only where a name starts) |
| `suppression` | added line containing `pragma: no cover`, `pragma: no branch`, `pragma: no mutate`, `noqa`, `type: ignore`, `pyright: ignore`, `mypy: ignore-errors`, `nosec`, `pylint: disable`, `eslint-disable`, `@ts-ignore`, `@ts-expect-error`, `@ts-nocheck`, `istanbul ignore`, `c8 ignore` (case-insensitive, flexible spacing) — unless an effective waiver has `scope == f"{path}:{line_no}"` |
| `gate_config` | any change to `.coveragerc`, `pytest.ini`, `.pytest.ini`, `pytest.toml`, `.pytest.toml`, `tox.ini`, `setup.cfg`, `.mutmut-config`; or an added line in `pyproject.toml` that opens a `[tool.coverage`, `[tool.pytest`, `[tool.mutmut`, `[tool.mypy` or `[tool.ruff` table, or sets one of the **keys** `fail_under`, `addopts`, `omit`, `exclude_also`, `exclude_lines`, `testpaths`, `norecursedirs`, `python_files`, `python_classes`, `python_functions` |
| `gitlink` | mode 160000 |
| `symlink_escape` | symlink target absolute, resolving outside the repo, or traversing another symlink known in the change set (targets are resolved component by component, not lexically: `a -> .` plus `b -> a/..` escapes) |
| `file_too_large` | `size > max_file_bytes` |

One violation per (rule, path, line). Violations sorted by (path, line or 0, rule). Messages are
instructive: say what was blocked, why, and what to do instead (e.g. "report it in your task
report"). Matching is case-sensitive.

---

## agents + evidence — `local_factory.agents`, `local_factory.evidence`

`agents/backend.py`

```python
class Capability(StrEnum): NATIVE_FS_TOOLS, HOOKS, STRUCTURED_OUTPUT, SESSIONS, SUBAGENTS
Effort = Literal["low", "medium", "high", "xhigh", "max"]
@dataclass(frozen=True)
class RoleSpec: name: str; prompt: str; tier: Tier; effort: Effort = "high";
                tools: tuple[str, ...] = (); max_turns: int = 50;
                output_schema: type[BaseModel] | None = None
class AgentTask(FrozenModel): ticket_id: str; task_id: str | None = None; instructions: str;
                              context: dict[str, JsonValue] = {}
@dataclass(frozen=True)
class WorkerHandle: work_dir: Path
class AgentResult(FrozenModel):
    status: RunStatus; halt: Halt | None = None; structured_output: dict[str, JsonValue] | None = None
    text: str | None = None; usage: Usage = Usage(); session_id: str | None = None
    transcript_path: Path | None = None; permission_denials: tuple[dict[str, JsonValue], ...] = ()
    error: str | None = None
    # validator: status == "halted" iff halt is not None; status == "error" requires error
class AgentBackend(Protocol):
    name: str; capabilities: frozenset[Capability]
    async def run(self, role: RoleSpec, task: AgentTask, worker: WorkerHandle) -> AgentResult: ...
```

`agents/fake.py` — `FakeBackend`, a scripted backend that executes actions in the worker directory.

```python
@dataclass(frozen=True) class WriteFile: path: str; content: str
@dataclass(frozen=True) class DeleteFile: path: str
@dataclass(frozen=True) class RunCommand: argv: tuple[str, ...]          # cwd=work_dir, no shell
@dataclass(frozen=True) class EmitOutput: data: dict[str, JsonValue]
@dataclass(frozen=True) class RaiseHalt: reason: HaltReason; resume_at: datetime | None = None
@dataclass(frozen=True) class Fail: message: str
Action = WriteFile | DeleteFile | RunCommand | EmitOutput | RaiseHalt | Fail
class FakeBackend:
    name = "fake"; capabilities = frozenset({Capability.STRUCTURED_OUTPUT})
    def __init__(self, scripts: Mapping[str, Sequence[Action]], *,
                 usage: Usage = Usage(input_tokens=100, output_tokens=50)) -> None
    calls: list[tuple[RoleSpec, AgentTask]]
    async def run(self, role, task, worker) -> AgentResult
```

Script lookup: `task.task_id` first, then `role.name`; none → `error="no script for …"`. Actions run in
order. Paths are confined to `work_dir`: absolute paths, `..` escapes and symlink escapes produce
`status="error"` (no exception) and nothing is written. `RunCommand` non-zero exit → `error`.
`RaiseHalt` stops and returns `halted`. `Fail` stops and returns `error`. With `role.output_schema`
set, the (last) `EmitOutput` must validate against it, else `error`; a missing `EmitOutput` is an
`error` too. Without a schema, `EmitOutput` data is passed through. Every result carries the
configured usage.

`evidence/binding.py`

```python
build_evidence(*, ticket_id, verdict: Verdict, tree_hash, base_sha, contract_hash, oracle_version,
               factory_version, image_digests=None, models=None, clock=utc_now) -> Evidence
    # candidate_sha = verdict.sha; gates/criteria/waivers/passed copied from the verdict;
    # ValueError (message names the ticket) if ticket_id != verdict.ticket_id
evidence_hash(evidence) -> str                         # hashing.content_hash
load_or_create_key(path: Path) -> bytes
    # creates 32 random bytes, file mode 0600 (parent dirs created); an existing key file with
    # group/other permissions raises EvidenceKeyError
sign(evidence, key: bytes) -> str                      # hex HMAC-SHA256 over canonical_json(evidence)
verify_signature(evidence, signature: str, key: bytes) -> bool   # constant-time
check_binding(evidence, *, sha, tree_hash, contract_hash, oracle_version) -> list[str]
    # human-readable mismatches; [] means bound. Also reports evidence.passed == False.
class EvidenceKeyError(Exception)
```

---

## Follow-ups recorded during Phase 1a review

- **Gate-config edits that remove lines are invisible to the diff policy** (`ChangedFile` carries
  added lines only), e.g. deleting `fail_under` or widening `omit` inside an existing table.
  Resolution (Phase 4): the factory measures coverage and mutation with its **own** config file and
  the thresholds in `factory.toml [policy]`, passed explicitly to the tools, so target-side config
  cannot weaken a gate.
- **Symlink chains need full-tree knowledge.** The diff policy resolves only through symlinks in the
  change set; `import_task_bundle` performs the authoritative check against every symlink in the
  head tree. In Phase 1b nothing on the host follows a worker-produced symlink at all.
- **`RunCommand` has no timeout** (FakeBackend). Revisit when commands run inside containers (1b).
- **Unsafe symlinks already present in a target's base are fail-closed**: every task on that repo is
  rejected (`symlink_escape`) until the link is fixed, because a worker can turn a dormant unsafe link
  into a live escape by retargeting another link. `factory project add` (Phase 2) must detect and
  report such links when a repo is adopted, rather than letting tasks fail one by one.
- **Binary files are opaque to the diff policy** (no `added_lines`). Phase 4: flag new or modified
  binary files under source/test paths for verifier review.
- **Bundle size is not capped at import.** Phase 1b: cap via the sandbox's `/out` quota plus a
  `BundleLimits.max_bundle_bytes` check before copying.
- **Fresh per-gate `HOME` means cold tool caches** (uv, pip, npm). Phase 1b: mount read-only shared
  caches into gate containers explicitly.
- **`DiffPolicy.test_globs` are Python-only by default** and not configurable per project yet.
  Phase 2: add `[paths] test_globs` to `factory.toml` for JS/other targets.
