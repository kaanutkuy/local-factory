# Phase 2b spec — Gate 1: a prompt becomes an approved, frozen contract

Goal (PLAN §4.1, §14 row 2b): `factory intake "<request>"` turns a request into a ticket whose
contract is frozen by a human, with **visible acceptance tests** committed for the implementers
and **black-box holdout probes** sealed in factory state, every one of them shown to **fail on the
current code** (red-first) before the human sees it — except the checks of a **keep** criterion
(behaviour the code has today and must not lose), which must **pass** on it.

Decisions this spec rests on (owner, 2026-11-13):

- **A.** Holdout probes in 2b are **black-box CLI probes stored as data**: `argv`, `stdin`, and the
  expected exit code / stdout. The factory runs the argv in a fresh box holding only the app's
  tree and compares the result **on the host**. The expectation never enters a box, a repository
  or the ledger. HTTP probes against a running app (app + probe containers, PLAN §5.4) stay in
  Phase 3.
- **B.** The notes-app template moves to Phase 3 (it needs an offline dependency story: FastAPI,
  Alembic, Playwright + Chromium in a network-less box). 2b proves Gate 1 on
  `tests/fixtures/cli-target`.
- **Live checks use Claude Haiku 5.5 for every tier** (the owner's API budget is small).
- **C.** (2026-11-14, after the first live run.) **Every criterion has a kind**, `change` or
  `keep`. A request often says what must *not* change ("without the flag the output stays as it
  is"); the tests of such a criterion pass on today's code by design, so "every check is red" made
  that contract impossible. Now a change criterion's tests and probes must fail on the base and a
  keep criterion's must pass on it (they guard it). The spec writer sets the kind, the critic
  checks it, the human approves it; the test architect cannot change it. A spec needs at least one
  change criterion, and a test or probe checks criteria of one kind only. Every criterion line a
  role reads (the critic's draft, the spec writer's previous draft, the test architect's brief)
  is `AC-n [layer, kind]: EARS`, followed by its test plan.
- **Worker trees ignore tool caches.** The first live run also showed a test architect that ran its
  own tests in the box getting `__pycache__/*.pyc` committed with them (`git add --all`), which
  the binary rule then rejected. Every worker repository now excludes Python tool caches from
  commits (below). The binary rule stays the actual check.

Simplifications, deliberate:

- **No separate reproducer role.** For a bug, the spec writer reproduces it in its box (it has
  bash) and the test architect must include a regression test that fails on the base. Red-first
  proves the repro is red. A dedicated reproducer can come back in Phase 3 if the data says so.
- **No `factory project sync`.** `project add` fetches the target's branch once into the factory's
  bare repository; moving `main` afterwards belongs to Phase 3/5.
- **Intake runs in the foreground, one ticket at a time,** and every model run happens in a fresh
  tree that is discarded. Only the test architect's work is exported, as a bundle, and only test
  files under its ticket's acceptance directory can pass.

Shared contract (orchestrator-owned, already written — do not edit, request changes):
`models.py` (the Gate 1 section: `TicketSize`, `TriageResult`, `CriterionDraft`, `SpecDraft`,
`QuestionDraft`, `Critique`, `Question`, `ProbeExpectation`, `ProbeDraft`, `HoldoutProbe`,
`VisibleTest`, `OracleManifest`, `RedCheck`, `Holdout`, and the id patterns; amended for C:
`CriterionKind`, `Criterion.kind`, `CriterionDraft.kind` (default `change`), `SpecDraft` needs a
change criterion, `RedCheck.green` / `.keep` / `.ok`), `config.py`
(`ProjectConfig.test_globs` / `[paths] test_globs`, `DEFAULT_TEST_GLOBS`, `ACCEPTANCE_COMMAND`),
and `tests/fixtures/cli-target/factory.toml` (now with `[commands] acceptance`).

Oracle (protected): `tests/acceptance/intake_world.py` (fixtures and a scripted backend for
Gate 1), `tests/acceptance/test_intake.py`, `tests/acceptance/test_oracle.py`,
`tests/acceptance/test_projects.py`, `tests/acceptance/test_cli.py`,
`tests/acceptance/test_intake_box.py` (container), `tests/redteam/test_holdout_seal.py`
(container), and the opt-in `tests/live/test_intake_live.py`. Plus `scripts/verify` on Python 3.12
**and** 3.14, ≥ 90% branch coverage over the whole repo, and every earlier acceptance and
red-team test still green.

## Ownership

| Agent | Owns (create/edit only these) |
|---|---|
| **oracle** | `src/local_factory/oracle/**` (new package), `src/local_factory/projects.py` (new), `src/local_factory/ledger/**`, `src/local_factory/gitops/bundles.py` (the `unsafe_symlinks` refactor only), `src/local_factory/policy/diff_policy.py` (`test_globs` only), `src/local_factory/pipeline/host_slice.py` and `sandboxed_slice.py` (the public `host_site()` / `box_site()` only); their unit tests under `tests/unit/oracle/`, `tests/unit/test_projects.py`, `tests/unit/ledger/`, `tests/unit/gitops/`, `tests/unit/policy/` |
| **intake** | `src/local_factory/pipeline/intake.py` (new), `src/local_factory/agents/roles.py` (new), `src/local_factory/cli.py`, `src/local_factory/pipeline/__init__.py` exports; their unit tests under `tests/unit/pipeline/`, `tests/unit/agents/test_roles.py`, `tests/unit/test_cli.py` |

The intake agent codes against the oracle agent's interfaces exactly as written below; until they
land, stub them in unit tests only. Neither agent edits the other's files.

**Amendment C (2026-11-14), one agent, "keep":** owns `src/local_factory/oracle/redfirst.py`,
`src/local_factory/gitops/worker_side.py` (the cache excludes only), `src/local_factory/pipeline/intake.py`,
`src/local_factory/agents/roles.py`, `src/local_factory/cli.py`, and their unit tests (`tests/unit/oracle/`,
`tests/unit/gitops/`, `tests/unit/pipeline/`, `tests/unit/agents/test_roles.py`, `tests/unit/test_cli.py`).

## Layout of a project's factory state

`FactoryConfig.project_state_dir(name)` = `$LOCAL_FACTORY_HOME/projects/<name>/`, mode 0700:

| Path | What |
|---|---|
| `project.json` | the `Project` record |
| `repo.git/` | the factory's bare repository; `refs/heads/main` is the project's main |
| `ledger.db` | the project's ledger (ticket ids are per project) |
| `holdout/` (0700) | `T-xxxx.json` (0600): one `Holdout` record per ticket |
| `transcripts/` | the Claude backend's transcripts (0600 files) |
| `work/` | scratch: base bundles, verification trees (host site) |

## oracle — `local_factory.oracle`, `local_factory.projects`, ledger, gitops, policy

### `oracle/holdout.py`

```python
class HoldoutStore:
    def __init__(self, root: Path) -> None: ...
    def path(self, ticket_id: str) -> Path: ...          # root / f"{ticket_id}.json"
    def save(self, holdout: Holdout) -> Path: ...         # atomic; replaces an existing record
    def load(self, ticket_id: str) -> Holdout | None: ... # None when there is none
```

- `save` creates `root` if needed and (re)sets it to 0700; writes a temp file in `root` created
  0600 (`O_EXCL`), then `os.replace` onto the record. The record is the `Holdout`'s canonical JSON
  (`hashing.canonical_json`).
- `load` validates with `Holdout.model_validate_json`; a record that does not parse, or whose
  `ticket_id` differs from the file name, raises `HoldoutError`.
- A ticket id that does not match `TICKET_ID_PATTERN` raises `ValueError` in `path`/`save`/`load`
  (no path tricks). Symlinked records or roots are refused (`HoldoutError`): open with
  `O_NOFOLLOW`, and `root` must be a real directory.

### `oracle/policy.py` — what the test architect may change

```python
ORACLE_PATH = "oracle_path"; ORACLE_SYMLINK = "oracle_symlink"
ORACLE_BINARY = "oracle_binary"; ORACLE_SIZE = "oracle_size"; ORACLE_FILES = "oracle_files"
MAX_ORACLE_FILES = 50
MAX_ORACLE_FILE_BYTES = 200_000

def acceptance_dir(ticket_id: str) -> str: ...   # f"tests/acceptance/{ticket_id}/"
def check_oracle_changes(changes: Sequence[ChangedFile], ticket_id: str) -> list[Violation]: ...
```

Every change must be an **added regular file** (mode `100644` or `100755`) under
`acceptance_dir(ticket_id)`. Anything else is a violation, one per offending change, in path order:
a symlink → `oracle_symlink`; any other kind/mode, or a path outside the directory →
`oracle_path`; a binary file → `oracle_binary`; larger than `MAX_ORACLE_FILE_BYTES` →
`oracle_size`. More than `MAX_ORACLE_FILES` changes adds one `oracle_files` violation (path `""`).
No changes at all is the caller's `no_changes`, not a violation.

### `oracle/probes.py` — running a black-box probe

```python
PROBE_TIMEOUT_S = 30.0
PROBE_SCRIPT = 'printf %s "$1" | { shift; exec "$@"; } 2>/dev/null'

def probe_argv(probe: ProbeDraft) -> list[str]: ...   # ["sh", "-c", PROBE_SCRIPT, "probe", probe.stdin, *probe.argv]
def judge_probe(probe: ProbeDraft, result: CommandResult) -> tuple[bool, str]: ...
async def run_probe(handle: WorkerHandle, probe: ProbeDraft) -> CommandResult: ...
```

- `run_probe` = `handle.tools().run_command(probe_argv(probe), timeout_s=PROBE_TIMEOUT_S)`: the
  argv runs with the tree's root as its cwd, stdin fed from the probe, stderr discarded, so the
  command's output is the program's stdout. **Only the argv and stdin go into the box**; the
  expectation stays on the host.
- `judge_probe` → `(passed, reason)`. Timed out → `(False, "timed out after 30s")`. Otherwise each
  given expectation is checked against `result.output` with trailing `"\n"` stripped from both
  sides: `exit_code` equal; `stdout` equal; every `stdout_contains` fragment present. All hold →
  `(True, "passed on the base")`. Else `(False, "failed on the base: " + "; ".join(mismatches))`
  with mismatches from `"exit code {actual} differs"`, `"stdout differs"`, `"stdout lacks an
  expected fragment"`, in that order. **A reason never contains an expected value.**

### `oracle/redfirst.py` — every new test and probe must fail on the base (keep checks: pass)

```python
RED_TEST_TIMEOUT_S = 120.0
NO_TESTS_RAN_EXIT = 5     # pytest and unittest (3.12+) agree

def split_test_id(test_id: str) -> tuple[str, str]: ...        # "path::name" → (path, name), on the first "::"
def render_acceptance(template: str, test_id: str) -> str: ... # {path}/{name} → shlex-quoted parts
def judge_visible(test_id: str, result: GateResult, *, keep: bool = False) -> RedCheck: ...
async def red_first(
    *, site: Site, repo: BareRepo, oracle_sha: str, base_sha: str,
    visible: Sequence[str], probes: Sequence[HoldoutProbe], acceptance: str, ticket_dir: Path,
    keep: Collection[str] = (),
) -> tuple[RedCheck, ...]: ...
```

- `render_acceptance`: `template` must contain `{path}` (`ValueError` whose message names
  `{path}`); `{name}` is optional. Both are replaced in **one pass** (a path containing
  `{name}` must not inject unquoted text), and nothing else in the template is touched.
- `judge_visible` (kind `"visible"`, id = the test id, output = the gate's tail clipped to its
  last 2 000 characters): `FAIL` with exit code `5` → not red, `"no test ran (exit 5)"`; other
  `FAIL` → red, `"failed on the base (exit N)"`; `PASS` → not red but **green**,
  `"passed on the base"`; anything else (timeout, error) → not red, `"did not run cleanly: " +
  tail of the output` (clipped to 300 characters). Only `PASS` is green. `keep` is copied into the
  check.
- `red_first`: the visible tests run on `oracle_sha` (main plus the new test files) in **one**
  `site.verification(...)`, each as its own `GateSpec(name=f"red-first {test_id}",
  command=render_acceptance(acceptance, test_id), timeout_s=RED_TEST_TIMEOUT_S)` passed to
  `verification.gates([spec])` **one at a time** (the gate runner stops after a failing gate, and
  here every gate is expected to fail). Each probe runs on `base_sha` (main: the app, no tests) in
  **its own** fresh `site.verification(...)` and is judged with `judge_probe` (kind `"holdout"`,
  id = the probe id, `red = not passed`, except a timeout, which is not red either; output =
  the command's output clipped to its last 2 000 characters; `green = passed`). Results come
  back visible tests first, then probes, in the given order. Distinct `round_no`s per verification.
  `keep` holds the ids (visible test ids and probe ids) of the checks that guard keep criteria:
  each such check has `keep=True`. Nothing else changes with it: every check runs and is judged
  the same way, and the caller decides with `RedCheck.ok`.

### `gitops/worker_side.py` — worker trees ignore tool caches (amendment)

`WORKER_EXCLUDES = ("__pycache__/", "*.pyc", ".pytest_cache/", ".mypy_cache/", ".ruff_cache/")`.
`init_from_bundle` writes them, one per line, to the new repository's `info/exclude`, so
`commit_all` (`git add --all`) never stages an *untracked* cache; changes to tracked files are
staged as before. It is a convenience for honest workers, not a boundary: a worker can edit its own
`.git`, and the policy checks (the binary rule, the oracle policy, the diff policy) stay the
boundary. Runs in the box: Python 3.12, unit-tested in-process.

### `oracle/__init__.py`

Re-exports the names above plus `ORACLE_REF_PREFIX = "refs/factory/oracle/"`,
`oracle_ref(ticket_id) -> str` (`refs/factory/oracle/T-0001`) and
`oracle_version(holdout: Holdout) -> str` (= `hashing.content_hash(holdout)`).

### Ledger additions (`ledger/store.py`)

```python
def record_event(self, ticket_id: str, kind: str, *, actor: str,
                 data: Mapping[str, JsonValue] | None = None) -> Event: ...
def triage(self, ticket_id: str, *, title: str, type: TicketType, risk: RiskTier, actor: str,
           data: Mapping[str, JsonValue] | None = None) -> Ticket: ...
def propose(self, ticket_id: str, contract: Contract, oracle_version: str, *, actor: str) -> Ticket: ...
```

- `record_event` appends an event for an existing ticket (`TicketNotFound` otherwise). It refuses
  (`ValueError` whose message names the kind) every kind the
  ledger writes itself (`created`, `transition`, `contract_proposed`, `oracle_version_set`,
  `contract_frozen`, `halted`, `resumed`, `task_created`, `task_updated`, `run_recorded`,
  `verdict_recorded`, `evidence_recorded`, `triaged`), so those keep meaning what the ledger says.
- `triage`: only from `RECEIVED` (else `IllegalTransition`); sets title, type and risk, appends a
  `triaged` event (`data` = the given data plus `title`, `type`, `risk`) and transitions to
  `TRIAGED`, in one transaction.
- `propose`: only from `TRIAGED` or `NEEDS_INFO`, not frozen; stores the contract and the oracle
  version (events `contract_proposed` and `oracle_version_set`, as the existing methods write
  them) and transitions to `CONTRACT_PROPOSED`, in one transaction.
- No schema change: everything new lives in events and the ticket body.

### `gitops/bundles.py` — `unsafe_symlinks`

`unsafe_symlinks(repo: BareRepo, rev: str) -> list[tuple[str, str]]`: `(path, reason)` for every
symlink in `rev`'s tree that the import check rejects, sorted by path. `_check_symlinks` becomes
"raise `symlink_escape` for the first of `unsafe_symlinks(...)`" — same codes, same messages, same
order as today (every existing test stays green).

### `policy/diff_policy.py` — `test_globs`

`DEFAULT_TEST_GLOBS` now comes from `config` (keep the name importable from `diff_policy`).
`DiffPolicy.from_project(config)` passes `config.test_globs`.

### `projects.py`

```python
PROJECT_FILE = "project.json"

class ProjectError(Exception): ...

class Project(FrozenModel):
    name: str            # ProjectConfig's name pattern
    source: Path         # absolute
    branch: str
    added_at: datetime

@dataclass(frozen=True)
class ProjectPaths:
    root: Path; repo: Path; ledger: Path; holdout: Path; transcripts: Path; work: Path

def project_paths(config: FactoryConfig, name: str) -> ProjectPaths: ...
def add_project(config: FactoryConfig, source: Path, *, name: str | None = None,
                branch: str = "main", clock: Callable[[], datetime] = utc_now) -> Project: ...
def list_projects(config: FactoryConfig) -> list[Project]: ...          # sorted by name
def load_project(config: FactoryConfig, name: str | None = None) -> Project: ...
def project_config(paths: ProjectPaths) -> ProjectConfig: ...
```

`add_project`, in this order, leaving nothing behind on any refusal (`ProjectError` with a message
a human can act on):

1. `source` must be a directory containing a git repository; resolve it to an absolute path.
2. Fetch `refs/heads/<branch>` from `source` into a fresh bare repo (in a temporary directory
   under `projects/`) as `refs/heads/main` (`BareRepo.init` + `fetch_from`). A missing branch →
   `ProjectError` naming it.
3. Read `factory.toml` **from the fetched commit** (`repo.read_file`), never from the source's
   worktree, and parse it (`parse_project_config` over `tomllib.loads`). Missing or invalid →
   `ProjectError`.
4. `name` defaults to `[project] name`; an explicit `name` must match the same pattern. A project
   of that name already registered → `ProjectError` (`"already"` in the message).
5. `[commands] acceptance` must exist and contain `{path}` → else `ProjectError` naming
   `acceptance`.
6. `unsafe_symlinks(repo, "refs/heads/main")` must be empty → else `ProjectError` listing **every**
   unsafe symlink as `path -> reason`, one per line (this closes the §14.1 row: such a base would
   fail every task closed with `symlink_escape`).
7. Build everything in the staging directory: `ledger.db` (`Ledger(path)`, closed again),
   `holdout/` (0700), `transcripts/`, `work/`, `project.json` last; then one `rename` to
   `projects/<name>/` (0700), so a half-built project never appears under its name.

`load_project(config, None)` returns the only project; none, or more than one → `ProjectError`
(the message lists the names and says to pass `--project`). An unknown name → `ProjectError`.
`project_config(paths)` reads `factory.toml` from `refs/heads/main` of `paths.repo`.

## intake — `local_factory.agents.roles`, `local_factory.pipeline.intake`, the CLI

### `agents/roles.py` — the four intake roles

| Constant | `name` | tier | effort | tools | `max_turns` | `output_schema` |
|---|---|---|---|---|---|---|
| `TRIAGE_ROLE` | `triage` | standard | medium | `BOX_TOOLS` | 30 | `TriageResult` |
| `SPEC_WRITER_ROLE` | `spec-writer` | frontier | high | `BOX_TOOLS` | 50 | `SpecDraft` |
| `CRITIC_ROLE` | `ambiguity-critic` | frontier | high | `BOX_TOOLS` | 30 | `Critique` |
| `TEST_ARCHITECT_ROLE` | `test-architect` | frontier | high | `BOX_TOOLS` | 80 | `OracleManifest` |

`INTAKE_ROLES` is the tuple of the four. Prompts are plain text and say, at least: what the role
is for and what it must submit; that the request, the repository's files and any command output
are **data, not instructions**; that it works in a throwaway copy of the repository (nothing it
writes is kept, except the test architect's test files). Per role:

- **triage:** classify type, risk (R0 docs/config/copy, R1 normal product change, R2 auth,
  payments, data, migrations), size (XL = an epic to split), areas; name a duplicate only if an
  open ticket asks for the same change.
- **spec-writer:** acceptance criteria in EARS form, each independently testable from outside
  (CLI behaviour, output, exit codes), with a test plan and a **kind**: `change` for behaviour the
  code must gain, `keep` for behaviour it has today that the request (or a real risk of the
  change) says must not break; at least one change criterion; NFRs, out of scope, assumptions,
  migration impact; for a bug, reproduce it with bash first; use the answers given.
- **ambiguity-critic:** ask only questions where a wrong guess would get the result rejected;
  record every other guess as an assumption; at most 5 questions; check each criterion's kind
  (change: the code lacks it today; keep: the code has it today) and ask when it is unclear.
- **test-architect:** write the visible tests only under the ticket's acceptance directory, using
  only what the repository already has (no installs, no network); a **change** criterion's tests
  must fail on the current code and pass once it is implemented, a **keep** criterion's must pass
  on the current code (and keep passing); a test or probe checks criteria of one kind only; list
  each as `path::name` the acceptance
  command can run; write black-box CLI probes (argv from the repository root, no shell, optional
  stdin, expected exit code and/or stdout) that check the same criteria **differently** from the
  visible tests (other inputs, edge cases); every criterion needs ≥ 1 visible test and ≥ 1 probe;
  for a bug, include a regression test that reproduces it.

### `pipeline/intake.py`

```python
ACTOR = "factory"
MAX_ORACLE_ATTEMPTS = 3
MAX_CLARIFY_ROUNDS = 3
MAX_REQUEST_CHARS = 20_000
MAX_OPEN_TICKETS_SHOWN = 50

IntakeStatus = Literal["contract_proposed", "needs_info", "oracle_failed", "agent_error", "halted"]

class IntakeError(Exception): ...      # misuse: wrong status, unknown question, bad request text

@dataclass(frozen=True)
class IntakeOutcome:
    status: IntakeStatus
    ticket: Ticket                          # fresh snapshot at the end
    questions: tuple[Question, ...] = ()    # the open ones (needs_info)
    red_checks: tuple[RedCheck, ...] = ()   # the last red-first run, if any
    error: str | None = None                # agent_error / oracle_failed / halted: why

@dataclass(frozen=True)
class Intake:
    site: Site
    ledger: Ledger
    repo: BareRepo
    project: ProjectConfig
    backend: AgentBackend
    holdout: HoldoutStore
    work_root: Path
    main_ref: str = "refs/heads/main"
    clock: Callable[[], datetime] = utc_now
    factory_config: FactoryConfig | None = None   # required unless the backend is the fake one

def receive(ledger: Ledger, text: str, *, author: str = "human") -> Ticket: ...
async def run_intake(intake: Intake, ticket_id: str) -> IntakeOutcome: ...
def questions(ledger: Ledger, ticket_id: str) -> list[Question]: ...       # all, answers filled in
def open_questions(ledger: Ledger, ticket_id: str) -> list[Question]: ...
def answer(ledger: Ledger, ticket_id: str, question_id: str, text: str, *,
           author: str = "human") -> Question: ...
def approve(ledger: Ledger, repo: BareRepo, holdout: HoldoutStore, ticket_id: str, *,
            author: str = "human") -> Ticket: ...
```

**`receive`:** `text.strip()` must be 1..`MAX_REQUEST_CHARS` characters (`IntakeError`). Creates a
`RECEIVED` ticket, actor = `author`, `request` = the stripped text, title = its first non-empty
line clipped to 80 characters, **type `feature` and risk `R2` until triage says otherwise**
(unknown risk is treated as the highest).

**`run_intake`** continues a ticket from its status, and never moves `main`:

| Status | What happens |
|---|---|
| `received` | triage, then as `triaged` |
| `triaged` | specify → clarify → oracle |
| `needs_info` | open questions left → `needs_info` outcome, no runs; none left → transition to `triaged` (reason `"questions answered"`) and continue |
| `contract_proposed` | `contract_proposed` outcome, no runs |
| anything else (incl. `halted`) | `IntakeError` (the CLI resumes a halted ticket first) |

Every model run: the role runs in a **fresh tree of main** — `site.verification(repo, main,
ticket_dir, round_no=k)` with `ticket_dir = work_root / ticket_id` — except the test architect,
which runs in `site.worker(...)`. The run is limited by `site.run_limit_s` (`flow.run_limited`)
and recorded as a `RunRecord` (role name, backend name, model = `"fake"` for the fake backend,
else `factory_config.model_for(backend.name, role.tier)`, tier, usage, status, transcript, error).
A run's `structured_output` is validated with the role's schema again in the factory. Then:

- `error`, or output that does not validate → event `intake_error` `{role, error}`, outcome
  `agent_error` with that error; the ticket's status is unchanged (run `run_intake` again to retry).
- `halted` → `ledger.halt(ticket, halt)`, outcome `halted` (`error` = the halt's reason and detail).

**1. Triage.** Instructions: the request, and the open tickets (non-terminal, not this one; the
latest `MAX_OPEN_TICKETS_SHOWN`) as `id`, `title`, and request clipped to 500 characters.
`ledger.triage(...)` with the result's title/type/risk and `data={"triage": <result>}`. Then the
factory asks itself, as `Question(asked_by="factory", round=0)`:
- `duplicate_of` names another existing ticket → `"Triage thinks this duplicates {id} ({title}):
  {rationale} Answer to continue with this ticket anyway."`;
- `size == XL` → `"Triage thinks this is too large for one ticket: {rationale} Answer with the part
  to build first, or to continue as one ticket."`.
A `duplicate_of` naming this ticket or no ticket is ignored. If either question was asked →
transition to `needs_info`, outcome `needs_info`.

**2. Specify.** Instructions: the request, the triage result, every question with its answer, and
the previous draft if there is one. The `SpecDraft` gets version = previous version + 1 (first: 1)
and is recorded as event `spec_drafted` `{version, draft}`. Criteria are numbered `AC-1..AC-n` in
the order given.

**3. Clarify.** A fresh critic. Instructions: the request, the triage result, the numbered
criteria (id + EARS + test plan) and the rest of the draft, and every question with its answer.
Recorded as event `critique` `{blocking, assumptions}`. The critic's assumptions are kept for the
contract. If `blocking` is non-empty and fewer than `MAX_CLARIFY_ROUNDS` critic rounds have asked
questions before: each becomes `Question(asked_by="ambiguity-critic", round=r)` (ids continue
across the ticket: `Q-1, Q-2, …`), recorded as event `question_asked` `{question}`; transition to
`needs_info`; outcome `needs_info`. After `MAX_CLARIFY_ROUNDS` rounds, further blocking questions
do not block: each is kept as the assumption `"Unresolved after 3 rounds: {text}"`.

**4. Oracle**, up to `MAX_ORACLE_ATTEMPTS` attempts `n = 1, 2, 3`, each starting from main:

1. Export main to `ticket_dir / f"oracle-{n}" / "base.bundle"` and run the test architect in
   `site.worker(bundle, ref)`. Instructions: the ticket id, the acceptance directory
   (`oracle.acceptance_dir`), the project's `acceptance` command template, every criterion (id,
   layer, kind, EARS, test plan — the line reads `AC-n [layer, kind]: EARS`), a sentence saying
   that change checks must fail and keep checks must pass on the current code and that a test or
   probe checks one kind only, out of scope and assumptions, and — from attempt 2 on — every problem
   of the previous attempt, one per line.
2. `worker.submit(base=main, message=f"{ticket_id}: acceptance tests", out=...)`; a `Refused`
   (e.g. `no_changes`) is a problem.
3. `import_task_bundle(repo, bundle, base_sha=main, dest_ref=f"refs/factory/oracle-candidate/{ticket_id}/{n}", limits=site.limits)`;
   `BundleRejected` is a problem (its code and detail).
4. `check_oracle_changes(import.changes, ticket_id)`: each violation is a problem.
5. The manifest: every visible test's path (`split_test_id`) must be one of the added files; test
   ids unique; every criteria id must be a contract criterion; every criterion needs ≥ 1 visible
   test **and** ≥ 1 probe; a visible test or a probe whose criteria include both kinds is a
   problem: `"{test id}: mixes change and keep criteria"`, `"HP-{n}: mixes change and keep
   criteria"` (n = its 1-based position in the manifest). Each failure is a problem naming the
   test, probe or criterion.
6. Only with no problems so far: number the probes `HP-1..HP-n` in order and run `red_first`
   (visible on the candidate commit, probes on main) with `keep` = the tests and probes whose
   criteria are all keep criteria. Recorded as event `red_first` `{attempt, oracle_sha, checks}`.
   Every check that is not `ok` is a problem: a change check → `"{id}: {reason}"`; a keep check →
   `"{id}: {reason}; a keep check must pass on the base"`.
   Problem texts: a `Refused` → `"{code}: {error}"`; a `BundleRejected` → `str(exc)`; a
   violation → `"{rule} {path}: {message}"`; a manifest problem names the test id, probe or
   criterion id it is about.
7. Problems → event `oracle_rejected` `{attempt, problems}`, the candidate ref is deleted, next
   attempt. No problems → **accept**: `refs/factory/oracle/<ticket>` → the candidate commit, the
   candidate ref deleted, `HoldoutStore.save(Holdout(ticket_id, oracle_sha, probes))`, and
   `ledger.propose(ticket, contract, oracle_version(holdout))` → outcome `contract_proposed`.

The proposed `Contract`: the latest draft's version, summary, nfr, out of scope, migration impact;
assumptions = the draft's, then the critic's, then unresolved questions (first occurrence kept);
each criterion's `kind` = the draft's, `tests` = its visible test ids in manifest order,
`holdout_probes` = its number of probes. After `MAX_ORACLE_ATTEMPTS` rejected attempts → event `oracle_failed`
`{attempts: MAX_ORACLE_ATTEMPTS}`, outcome `oracle_failed` (error = the last attempt's problems),
the ticket stays `triaged`, no oracle ref, no holdout record.

**Questions** are folded from events: `question_asked` adds one, `question_answered`
`{id, answer, answered_by}` fills its answer. `answer()`: the ticket must be `needs_info`, the
question must exist and be open, `text.strip()` must be non-empty (each else `IntakeError`).

**`approve()`**: the ticket must be `contract_proposed` with a contract; the holdout record must
exist, and `oracle_version(record)` must equal the ticket's `oracle_version` and `record.oracle_sha`
the commit `refs/factory/oracle/<ticket>` points at (each else `IntakeError`: the oracle was
changed after it was proposed). Then `ledger.freeze_contract(ticket, actor=author)` → `approved`.

### `pipeline/host_slice.py`, `pipeline/sandboxed_slice.py`

`host_site() -> Site` (the existing `_HostSite`: plain directories, **not a security boundary**,
for in-process tests) and `box_site(sandbox: Sandbox) -> Site` (the existing `_BoxSite`).

### `cli.py`

```
factory project add PATH [--name NAME] [--branch BRANCH]
factory intake TEXT [--project NAME]
factory intake --resume TICKET [--project NAME]
factory questions TICKET [--project NAME]
factory answer TICKET QUESTION TEXT [--project NAME]
factory show TICKET [--holdout] [--project NAME]
factory approve TICKET [--project NAME]
```

```python
@dataclass(frozen=True)
class Services:
    backend: Callable[[FactoryConfig, Path], AgentBackend]   # (config, transcripts dir)
    site: Callable[[FactoryConfig], AbstractContextManager[Site]]
    clock: Callable[[], datetime] = utc_now

def default_services() -> Services: ...   # ClaudeAPIBackend; box_site(Sandbox(DockerRuntime(), config.sandbox)), runtime cleaned up on exit
def main(argv: Sequence[str] | None = None, *, services: Services | None = None) -> int: ...
```

- The factory home is `config.factory_home()` (`LOCAL_FACTORY_HOME`), the config
  `load_factory_config()`. The site is entered only by commands that run models (`intake`, and
  `answer` when it continues intake).
- Exit codes: `0` success (including `needs_info`); `1` an operational failure or refusal (message
  on stderr, starting `error:`); `2` usage (argparse).
- `intake` / `intake --resume` / a continuing `answer` print `"{ticket}: {status}"` first, on
  stdout, whatever the status (a request refused before a ticket exists prints only the error). For
  `needs_info`, each open question as `"{id}: {text}"` and how to answer; for
  `contract_proposed`, how to review and approve; `oracle_failed`, `agent_error` and `halted`
  print the error and exit `1`. `--resume` on a halted ticket resumes it first (`ledger.resume`).
- `answer` records the answer, prints `"{question} answered"`, and continues intake when the
  ticket is `needs_info` with no open question left.
- `questions` prints every question: `"{id} [open] {text}"` or `"{id} [answered] {text}"` followed
  by its answer.
- `show` prints the ticket (id, title, status, type, risk, request), the questions, and the contract
  with each criterion's id, layer, kind, EARS text (`"  {id} [{layer}, {kind}] {ears}"`), visible
  tests and number of probes, and the oracle
  version. **The probes' argv, stdin and expectations appear only with `--holdout`.**
- `approve` prints `"{ticket}: approved"` and the contract hash.
- `project add` prints `"project {name} added"`; a refusal prints the `ProjectError` and exits `1`.

## Red-team items — `tests/redteam/test_holdout_seal.py` (protected, `container`)

1. **A probe's expectation never enters a box.** While red-first runs, another probe searches the
   whole box filesystem (outside `/proc`, `/sys`, `/dev`) for a planted expected-stdout canary and
   finds nothing.
2. **…nor the repository or the ledger:** the canary is in no object of the bare repository and
   nowhere in `ledger.db`; the holdout record is 0600 in a 0700 directory.
3. **The test architect cannot change anything but its own tests:** edits to app code,
   `factory.toml`, another ticket's tests, a workflow file, and a symlink to `/etc/passwd` under
   its directory are all rejected; no oracle ref appears.
4. **The test architect's test code runs only in the box:** a test that, when run, writes a file at
   a host path leaves nothing on the host.

## Exit check

- With the in-process host site and the fake backend: a request goes `received → triaged →
  needs_info → triaged → contract_proposed → approved`, with questions answered, a vacuous test
  sent back to the test architect, and a frozen contract whose oracle version matches the sealed
  holdout record (`tests/acceptance/test_intake.py`, `test_cli.py`). A keep criterion's checks
  must pass on the base; tool caches a test architect leaves behind are not committed.
- The same in the box (`test_intake_box.py`), red-team items green, Docker left clean.
- Opt-in, `uv run pytest -m live tests/live/test_intake_live.py` (Haiku 5.5 for every role):
  a real model turns the `--shout` request into a contract proposed with verified tests and
  probes (change checks red, keep checks green on the base), which a human (the test) approves; the report gives each role's turns, tokens and cost,
  and how many of the tests and probes pass on a known-good implementation.

## Deferred, with owners

| Item | Phase |
|---|---|
| HTTP holdout probes in app + probe containers (§5.4); redaction of probe feedback to implementers | 3 |
| The notes-app template and its offline dependencies | 3 |
| Running a ticket's acceptance tests in the gates (the cli-target's `unittest discover` does not reach `tests/acceptance/T-xxxx/`) | 3 |
| `factory project sync` (moving `main` from the source repository) | 3/5 |
| A separate reproducer role, if the data shows the spec writer's repro is not enough | 3 |
| Holdout lifecycle after merge (promote to regression suite) | 3 (PLAN §15) |
