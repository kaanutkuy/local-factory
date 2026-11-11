# Phase 1b spec — the box

Goal: move every worker-side action of the Phase 1a slice behind a container boundary. **Brain on
the host, hands in the box**: a backend's decision-making (FakeBackend's script now, a model's tool
loop later) runs in the factory process, and everything it *does* to a worker — write, read and
delete files, run commands — runs inside a hardened container with **no network**, as a user that
is **neither root nor the host user**, with only the task workspace writable. The worker's git,
the task export and every gate run in the box too. Results leave the box only as data: git
bundles and bounded text, validated on the host (invariant 7).

How a *real* agent reaches its model is **not** decided here (PLAN §15, "agent placement"): option A
keeps the model loop on the host and uses the box only through `ToolExecutor`; option B would run
the agent inside the box behind a broker. 1b builds what both need. Nothing in 1b opens a network
path out of the box.

Shared contract (orchestrator-owned, already written — do not edit, request changes):
`config.py` (`SandboxConfig`, `CacheMount`, `FactoryConfig.sandbox`), `agents/tools.py`
(`ToolExecutor`, `CommandResult`, `ToolError`), `docker/sandbox/` (the image, including the
`/usr/local/bin/lf-entry` launcher), `scripts/build-images`, the `container` pytest marker.

Oracle (protected): `tests/acceptance/test_box_*.py`, `tests/acceptance/test_core_1b.py`,
`tests/acceptance/test_sandboxed_slice.py`, and the red-team suite `tests/redteam/**`. Plus
`scripts/verify` on Python 3.12 **and** 3.14, ≥ 90% branch coverage over the whole repo.

## Ownership

| Agent | Owns (create/edit only these) | Acceptance |
|---|---|---|
| **runtime** | `src/local_factory/sandbox/{spec,runtime,workspace}.py`, `tests/unit/sandbox_runtime/**`, `tests/integration/sandbox_runtime/**` | `test_box_spec.py`, `test_box_runtime.py` |
| **box** | `src/local_factory/sandbox/{__init__,entry,executor,box}.py` (+ any further new `sandbox/` module), `src/local_factory/gitops/worker_side.py`, `tests/unit/sandbox_box/**`, `tests/integration/sandbox_box/**`, `tests/integration/gitops/test_worker_side.py` | `test_box_tools.py`, `test_box_gates.py` |
| **core** | `src/local_factory/agents/**` except `tools.py`, `src/local_factory/ledger/**`, `src/local_factory/gitops/{bundles,__init__}.py`, `tests/unit/agents/**`, `tests/unit/ledger/**`, `tests/integration/gitops/test_bundles.py` | `test_core_1b.py` (+ 1a `test_agents.py`, `test_ledger.py`, `test_gitops.py` stay green) |
| **integrator** (wave 2) | `src/local_factory/pipeline/**`, `tests/unit/pipeline/**`, `tests/integration/pipeline/**` | `test_sandboxed_slice.py`, `tests/redteam/**` (+ 1a `test_host_slice.py` stays green) |

Container tests carry `@pytest.mark.container` (plus `integration` where they also use real git).
They need Docker and the image (`scripts/build-images`); without them they **fail** with a message
saying so — never skip. Every container, volume and holder a test creates must be gone when the
test ends (use a per-test `DockerRuntime` and call `cleanup()` in the fixture finalizer). The box
runs Python 3.12 regardless of the host's Python; code that runs inside it must be 3.12-clean.

Code that runs **inside** the box (`sandbox/entry.py`, `gitops/worker_side.py`, the sync helpers in
`agents/local.py`) is invisible to host coverage when executed in containers: unit-test it
in-process on `tmp_path` as well (call `entry.main([...])` directly).

---

## runtime — `local_factory.sandbox.{spec,runtime,workspace}`

### `sandbox/spec.py`

```python
BOX_WORK = "/work"; BOX_HOME = "/home/sandbox"; BOX_FACTORY = "/opt/local-factory"; BOX_IN = "/in"
LABEL = "local-factory"                     # label value "1" on every object the factory creates
INSTANCE_LABEL = "local-factory.instance"   # value: the owning DockerRuntime's instance id
class SpecError(ValueError): ...

@dataclass(frozen=True)
class BindMount:   source: Path; target: str; read_only: bool = True
@dataclass(frozen=True)
class VolumeMount: volume: str; target: str; read_only: bool = False
@dataclass(frozen=True)
class TmpfsMount:  target: str; size_mb: int; executable: bool = False

@dataclass(frozen=True)
class ContainerSpec:
    image: str
    argv: tuple[str, ...]
    uid: int
    gid: int
    memory_mb: int
    cpus: float
    pids_limit: int
    tmp_size_mb: int
    env: Mapping[str, str] = {}             # field(default_factory=dict)
    binds: tuple[BindMount, ...] = ()
    volumes: tuple[VolumeMount, ...] = ()
    tmpfs: tuple[TmpfsMount, ...] = ()
    workdir: str = "/"
    interactive: bool = False               # attach stdin

    @classmethod
    def from_config(cls, config: SandboxConfig, *, argv: Sequence[str], env=None, binds=(),
                    volumes=(), tmpfs=(), workdir="/", interactive=False) -> ContainerSpec
        # image, uid, gid, memory_mb, cpus, pids_limit, tmp_size_mb taken from config

def create_args(spec: ContainerSpec, *, name: str, labels: Mapping[str, str]) -> list[str]
```

`ContainerSpec.__post_init__` raises `SpecError` for: uid or gid < 1; empty `argv` or any NUL;
an empty image or one containing whitespace; `memory_mb` < 16, `cpus` ≤ 0, `pids_limit` < 1,
`tmp_size_mb` < 1, a `TmpfsMount.size_mb` < 1; env keys that are not identifiers, values with NUL,
or a key `HOME`; any mount target that is not a normalized absolute path, is `/`, contains `,` or
`=`, equals `/tmp` or `BOX_HOME`, or duplicates another target; a `workdir` that is not a normalized
absolute path; a volume name not matching `^[A-Za-z0-9][A-Za-z0-9_.-]*$`; a bind source that is
relative, contains `,` or `=`, or — after `os.path.realpath` — is `/`, is the host user's home
directory or one of its ancestors, is or lies inside `~/.docker`, `/proc`, `/sys`, `/dev`,
`/var/lib/docker`, `/run/containerd`, or is any of `/var/run/docker.sock`, `/run/docker.sock`.

`create_args` returns the argv after the docker binary. Hardening is **unconditional**: no field
or argument can remove or weaken any of these, and nothing else is emitted.

```
create --name <name> --label local-factory=1 [--label k=v ...]          # labels sorted by key
  --user <uid>:<gid> --cap-drop ALL --security-opt no-new-privileges --read-only
  --network none --ipc private --pids-limit <n> --memory <m>m --memory-swap <m>m --cpus <c>
  --init --log-driver none --pull never --hostname sandbox
  --tmpfs /tmp:rw,nosuid,nodev,exec,size=<t>m
  --tmpfs /home/sandbox:rw,nosuid,nodev,exec,size=<t>m,uid=<uid>,gid=<gid>,mode=0700
  --env HOME=/home/sandbox --env TMPDIR=/tmp --env LANG=C.UTF-8 [--env K=V ...]   # spec env sorted
  [--mount type=bind,source=<s>,target=<t>[,readonly] ...]
  [--mount type=volume,source=<v>,target=<t>,volume-nocopy[,readonly] ...]
  [--tmpfs <t>:rw,nosuid,nodev,(exec|noexec),size=<n>m,uid=<uid>,gid=<gid>,mode=0700 ...]
  --workdir <w> [--interactive] <image> <argv...>
```

Every `--env` carries `=`: a bare `--env KEY` would copy the variable from the host. Every
volume mount carries `volume-nocopy`: without it Docker copies the image's directory (owner and
mode) onto the live workspace volume, making `/work` 0755 — or unwritable for a non-default uid. Never `-v`,
`--privileged`, `--cap-add`, `--device`, `--pid`, `--userns`, `--network` other than `none`.

### `sandbox/runtime.py`

```python
class SandboxError(Exception): ...   # the factory's side failed: docker missing, daemon down,
                                     # image missing, create/inspect failed. Never worker-caused.
OutputMode = Literal["head", "tail"]

@dataclass(frozen=True)
class ContainerResult:
    exit_code: int | None    # State.ExitCode; None when timed_out or output_exceeded
    timed_out: bool
    oom_killed: bool         # State.OOMKilled
    output_exceeded: bool    # "head"/sink mode only: stdout passed max_stdout_bytes
    stdout: bytes            # head: first max bytes; tail: last max bytes; sink: b""
    stderr: str              # last stderr_tail_bytes, decoded with replacement
    duration_s: float

def docker_env(environ: Mapping[str, str] | None = None) -> dict[str, str]
    # the docker CLI's own environment: PATH and HOME, plus DOCKER_HOST, DOCKER_CONTEXT,
    # DOCKER_CONFIG, DOCKER_CERT_PATH, DOCKER_TLS_VERIFY when set. Nothing else, ever.

class DockerRuntime:
    def __init__(self, *, docker: str = "docker", instance: str | None = None) -> None
        # instance defaults to "lf" + 12 random hex chars; must match [a-z0-9][a-z0-9-]*
    docker: str
    instance: str
    def check(self, image: str) -> str        # daemon reachable and image present -> image id
    def image_id(self, image: str) -> str     # "sha256:..."; SandboxError if missing
    async def run(self, spec: ContainerSpec, *, timeout_s: float, stdin: bytes | None = None,
                  max_stdout_bytes: int = 1_000_000, output: OutputMode = "head",
                  stdout_sink: BinaryIO | None = None,
                  stderr_tail_bytes: int = 16_000) -> ContainerResult
    def start_detached(self, spec: ContainerSpec) -> str   # create + start; the container name
    def remove_container(self, name: str) -> None          # force-remove; missing is fine
    def create_volume(self, *, size_mb: int, uid: int, gid: int) -> str
    def remove_volume(self, name: str) -> None             # missing is fine
    def owned(self) -> tuple[list[str], list[str]]         # (containers, volumes) of this instance
    def cleanup(self) -> None                              # remove all of them: containers first
```

- Every docker CLI process runs with `docker_env()`, list argv, no shell.
- Names: `lf-<instance>-<12 hex>`. Every container and volume is labelled `local-factory=1` and
  `local-factory.instance=<instance>`.
- `run`: `docker create` (failure → `SandboxError` with docker's stderr) → `docker start --attach`
  (`--interactive` iff `spec.interactive`; `stdin` is written then closed) → read stdout per
  `output` (`head`: keep the first `max_stdout_bytes`, kill the container when exceeded and set
  `output_exceeded`; `tail`: keep the last `max_stdout_bytes`, never kill for size; with
  `stdout_sink`: write every chunk to the sink, kill when the total exceeds `max_stdout_bytes`) →
  `docker inspect` for `State.ExitCode` and `State.OOMKilled`. On `timeout_s` (wall clock from
  start) the container is killed and `timed_out` set. The container is force-removed in **every**
  case: success, timeout, exception, and cancellation of the awaiting task (removal must not be
  skipped by `CancelledError`). `check(image)` is not called by `run`.
- `stdin` without `spec.interactive`, or `stdout_sink` with `output="tail"`: `ValueError`.
- `check`: `SandboxError` whose message says what is wrong; for a missing image it names the image
  and says to run `scripts/build-images`.
- `create_volume`: local driver, tmpfs-backed: `--opt type=tmpfs --opt device=tmpfs
  --opt o=size=<n>m,uid=<uid>,gid=<gid>,mode=0700`, labelled as above.

### `sandbox/workspace.py`

```python
class TaskWorkspace:
    """A memory-backed, size-limited /work for one task. A tmpfs volume loses its contents when
    no container has it mounted, so a holder container keeps it mounted from open() to close()."""
    def __init__(self, runtime: DockerRuntime, config: SandboxConfig) -> None
    volume: str                                   # set by open(); AttributeError/RuntimeError before
    def open(self) -> TaskWorkspace               # idempotent
    def close(self) -> None                       # remove holder, then volume; idempotent
    def __enter__(self) -> TaskWorkspace ; def __exit__(...) -> None   # open / close
    def mount(self, *, read_only: bool = False) -> VolumeMount         # the volume at BOX_WORK
```

`open()`: `create_volume(size_mb=config.work_size_mb, uid=config.uid, gid=config.gid)`, then a
holder from `ContainerSpec.from_config(config, argv=("sleep", "infinity"),
volumes=(self.mount(read_only=True),))` via `start_detached`. If starting the holder fails, the
volume is removed before the error propagates. After `close()` the data is gone.

---

## box — `local_factory.sandbox` (entry, executor, box), `gitops.worker_side`

### Inside the box: `sandbox/entry.py`

`main(argv: Sequence[str]) -> int`, run by `/usr/local/bin/lf-entry` (`python3 -I`, with
`/opt/local-factory` on `sys.path`). Subcommands:

| command | does | exit |
|---|---|---|
| `materialize --bundle B --ref R --dest D [-- CMD...]` | `worker_side.init_from_bundle(B, R, D)`; with `CMD`: `chdir(D)`, stderr → stdout (`dup2`), `execvp(CMD)` | failure before exec: 70, reason on stderr |
| `write --root R --path P` | write stdin (UTF-8 text) to P via `agents.local.write_file` | 0; refused/failed: 2 |
| `read --root R --path P` | the file's bytes to stdout via `agents.local.read_file` (`max_bytes` from `--max-bytes`, default 1 000 000) | 0; refused/failed: 2 |
| `delete --root R --path P` | `agents.local.delete_file` | 0; refused/failed: 2 |
| `exec -- CMD...` | stderr → stdout, `execvp(CMD)` in the current directory | the command's; not found: 127 |
| `export --work W --base SHA --message M` | `commit_all(W, M)`; `has_task_commits`; bundle `refs/heads/task ^SHA` written to **stdout** | 0 bundle written; 3 no commits; 4 any git/OS failure, reason on stderr |

Exit 2 and 4 print `error: <reason>` as the last stderr line (≤ 2 000 chars). Unknown subcommand
or bad arguments: exit 64. `worker_side` gains `export_task_bundle_to_stdout(work, *, base_sha)`
(or an equivalent `out="-"` form) for `export`.

### Host side: `sandbox/executor.py`

```python
class SandboxExecutor:            # implements agents.tools.ToolExecutor
    def __init__(self, runtime: DockerRuntime, config: SandboxConfig, workspace: TaskWorkspace,
                 *, factory_src: Path) -> None
```

Every call is **one** container from `ContainerSpec.from_config(config, ...)` with the workspace
at `/work` (rw), `factory_src` bound read-only at `/opt/local-factory/local_factory`, workdir
`/work`:

- `write_file` → `lf-entry write`, content on stdin (`interactive=True`); `read_file` → `lf-entry
  read --max-bytes <config.max_output_bytes>`, head mode with the same cap (exceeded →
  `ToolError`); `delete_file` → `lf-entry delete`. Timeout `config.command_timeout_s`. Exit ≠ 0 or
  timeout → `ToolError` carrying the `error:` reason (bounded).
- `run_command(argv, timeout_s)` → `lf-entry exec -- argv`, tail mode with
  `config.max_output_bytes` → `CommandResult(exit_code, output, timed_out)`; an OOM kill appends
  `[local-factory] killed: out of memory` to the output. Empty `argv` → `ToolError`.
- `SandboxError` from the runtime propagates (a factory failure is never blamed on the worker).

### Host side: `sandbox/box.py` — the facade (re-exported from `local_factory.sandbox`)

```python
@dataclass(frozen=True)
class ExportOutcome:
    status: Literal["exported", "no_changes", "worker_git_error", "bundle_too_large"]
    detail: str = ""          # bounded (<= 2 000 chars)

class Sandbox:
    def __init__(self, runtime: DockerRuntime, config: SandboxConfig, *,
                 factory_src: Path | None = None) -> None
        # factory_src defaults to the directory of the imported local_factory package.
        # SandboxError if config.uid == os.getuid() or config.gid == os.getgid(): the box never
        # runs as the host user.
    runtime: DockerRuntime; config: SandboxConfig; factory_src: Path
    def check(self) -> str                       # runtime.check(config.image)
    def image_id(self) -> str
    def workspace(self) -> TaskWorkspace         # unopened; use `with sandbox.workspace() as ws`
    def executor(self, ws: TaskWorkspace) -> SandboxExecutor
    async def materialize(self, ws: TaskWorkspace, *, bundle: Path, ref: str) -> None
    async def export(self, ws: TaskWorkspace, *, base_sha: str, message: str, out: Path) -> ExportOutcome
    async def run_gates(self, specs: Sequence[GateSpec], *, bundle: Path, ref: str,
                        fail_fast: bool = False, tail_bytes: int = 16_000) -> list[GateResult]
```

- The box runs as another uid, so it can only read what is world-readable. Bundles handed to
  `materialize` / `run_gates` are the factory's own files and are `chmod 0644`-ed before being
  mounted. `Sandbox.__init__` raises `SandboxError` if any file under `factory_src` is not
  world-readable or any directory in it is not world-readable and searchable (it is mounted as is).
  `__pycache__` directories are exempt: the box can always compile from source, and the host
  interpreter writes them with the user's umask.
- `materialize`: `bundle` bound read-only at `/in/base.bundle`; `lf-entry materialize`; failure
  → `SandboxError` (the bundle is the factory's own, so this is a factory error).
- `export`: `out` must not exist (`FileExistsError`); it is created `0600` with `O_EXCL` and is the
  `stdout_sink`, capped at `config.max_bundle_bytes`; timeout `config.command_timeout_s`. Exit 0
  with ≥ 1 byte → `exported`; exit 3 → `no_changes`; `output_exceeded` → `bundle_too_large`;
  anything else (other exit codes, empty output, timeout, OOM) → `worker_git_error`, `detail` =
  the `error:` reason or a description of the timeout/OOM. `out` is deleted unless `exported`.
  A worker can make export fail or lie; it can never make it raise.
- `run_gates`: **one fresh container per gate**, so no gate can alter what a later gate sees:
  `bundle` read-only at `/in/tree.bundle`, a `TmpfsMount("/work", config.work_size_mb,
  executable=True)`, each `config.cache_mounts` entry bound read-only, `config.gate_env` as env,
  argv `lf-entry materialize --bundle /in/tree.bundle --ref <ref> --dest /work -- bash -c
  <command>`, tail mode with `tail_bytes`, timeout `spec.timeout_s`. Exit 0 → PASS; non-zero →
  FAIL with `exit_code`; timed out → ERROR (`exit_code` None); OOM appends a note. `fail_fast`,
  SKIPPED results and the `[local-factory] ` note prefix behave exactly as in
  `verification.run_gates`. `output_tail` never exceeds `tail_bytes` characters.

---

## core — `agents`, `ledger`, `gitops.bundles`

### `agents/local.py` and `agents/backend.py`

```python
# agents/local.py — usable inside the box (entry.py imports the sync helpers) and on the host
def write_file(root: Path, path: str, content: str) -> None
def read_file(root: Path, path: str, *, max_bytes: int) -> bytes
def delete_file(root: Path, path: str) -> None
    # ToolError on: NUL, absolute path, a path not naming a file, parent or target resolving
    # outside root (.. or symlinks), symlink loops (one message on every Python version),
    # missing file (read/delete), directory (read/delete), larger than max_bytes (read)
class LocalExecutor:     # ToolExecutor on a plain directory; NOT a security boundary
    def __init__(self, root: Path, *, max_output_bytes: int = 1_000_000) -> None
    # run_command: asyncio subprocess, cwd=root, stdin=DEVNULL, own process group, stdout and
    # stderr interleaved, keep the last max_output_bytes; on timeout kill the whole group and
    # return CommandResult(None, output, timed_out=True). Minimal env as gates have (PATH, LANG,
    # HOME=TMPDIR=fresh temp dir).

# agents/backend.py
@dataclass(frozen=True)
class WorkerHandle:
    work_dir: Path
    executor: ToolExecutor | None = None
    def tools(self) -> ToolExecutor          # executor, or LocalExecutor(work_dir)
```

### `agents/fake.py`

- `WriteFile`, `DeleteFile`, `RunCommand` are executed through `worker.tools()`, never directly.
  With an executor set, FakeBackend does not touch `work_dir` on the host at all.
- `RunCommand(argv, timeout_s: float | None = None)`; `FakeBackend(..., command_timeout_s: float =
  300.0)` is the default when `timeout_s` is None. A timeout → `status="error"`, error text
  containing `timed out`. Non-zero exit → error with the exit status and output tail (as in 1a).
  `ToolError` → error `action <n> (<description>): <message>`. All 1a semantics otherwise
  unchanged (`tests/acceptance/test_agents.py` stays green as written).

### `ledger` — structured transition data (PLAN §14.1)

`Ledger.transition(ticket_id, to, *, actor, reason="", data: Mapping[str, JsonValue] | None = None)`.
Without `data` the event's data is exactly `{"reason": reason}` (unchanged); with `data` it is
`{"reason": reason, "detail": {...data}}`.

### `gitops.bundles` — bundle size cap (PLAN §14.1)

`BundleLimits.max_bundle_bytes: int = 100_000_000`. `import_task_bundle` checks the opened file's
size (`fstat`, no copying yet) right after opening it: larger → `BundleRejected("bundle_too_large")`.
It is checked before every other code (an unopenable file is still `invalid_bundle`).

---

## integrator — `pipeline/sandboxed_slice.py`

```python
async def run_sandboxed_slice(*, ledger: Ledger, repo: BareRepo, project: ProjectConfig,
                              backend: AgentBackend, sandbox: Sandbox, ticket_id: str,
                              work_root: Path, key: bytes, main_ref: str = "refs/heads/main",
                              clock: Callable[[], datetime] = utc_now,
                              factory_config: FactoryConfig | None = None) -> SliceOutcome
```

Same preconditions, flow, ledger records and `SliceOutcome` as `run_host_slice`
(`docs/specs/phase-1a-slice.md`), with every worker-side step in the box:

1. Preconditions as 1a, plus `sandbox.check()` — all before the first ledger change.
2. Base bundle → `work_root/<ticket>/task-<n>/base.bundle`; open a workspace; `materialize`.
3. Implementer: `backend.run(role, task, WorkerHandle(work_dir=Path("/work"),
   executor=sandbox.executor(ws)))` under `asyncio.wait_for(..., sandbox.config.run_timeout_s)`.
   Timeout → treated as a backend error `timeout: the run exceeded <n>s` (task `REJECTED`,
   `agent_error`, recorded in the run and the task event).
4. `sandbox.export(ws, base_sha=..., message="<task id>: <goal>", out=<task dir>/task.bundle)`:
   `no_changes` / `worker_git_error` / `bundle_too_large` → task `REJECTED` with that reason and
   `{"error": detail}`; `exported` → `SUBMITTED`. The workspace is closed right after export (and
   on every other path, including exceptions).
5. Import with `BundleLimits(max_bundle_bytes=sandbox.config.max_bundle_bytes)`, diff policy,
   merge — as 1a.
6. Verification of a sha: `export_base_bundle` → `work_root/<ticket>/verify-<sha[:12]>-<round>.bundle`;
   `sandbox.run_gates(...)` on it; the verifier runs through the backend in a **fresh** workspace
   materialized from the same bundle (discarded afterwards). Rest as 1a.
7. Evidence: as 1a, plus `image_digests={"sandbox": sandbox.image_id()}`.
8. A candidate conflict transitions back to `BUILDING` with `reason="candidate_conflict"` and
   `data={"conflicts": [...paths]}` (change `run_host_slice` the same way).

Host-side code never runs git in, and never writes, a worker tree: after a run `work_root` holds
only directories and the factory's own bundle files. `run_host_slice` stays (in-process, through
`LocalExecutor`) for fast tests; its docstring must say it is not a security boundary. Share code
between the two slices rather than duplicating the flow.

---

## Red-team suite v1 — `tests/redteam/` (protected, `container`)

Every check must hold. Written by the orchestrator before implementation:

- **identity**: in the box the uid is neither 0 nor the host uid; no effective capabilities;
  `NoNewPrivs` set; the root filesystem is read-only; no Docker socket; pids and memory limits are
  in force.
- **visibility**: canaries planted on the host — a file outside the box, environment variables
  (including names a provider/GitHub credential would use), factory state (ledger, evidence key,
  bare repo), another task's workspace — never appear anywhere in the box, nor in anything that
  comes back out of it.
- **network**: only loopback exists; nothing on the host (loopback or the Docker bridge) and no
  name lookup is reachable.
- **resources**: a command that outlives its timeout is killed and reported, a fork bomb is held
  by the pids limit, a memory hog by the memory limit; the pipeline carries on and nothing is left
  running or mounted afterwards.
- **pipeline**: a worker that deletes or corrupts its repository, overflows the bundle cap,
  rewrites its base, plants an absolute symlink, or edits a protected file is rejected with the
  right reason, and the host never materializes the worker's tree.

---

## Deferred, with owners

| To | What |
|---|---|
| Phase 2 | **Agent placement** (PLAN §15): **decided 2026-11-11, option A** = host loop + `ToolExecutor` (no network in the box, no broker); B was rejected. Phase 2 builds `ClaudeAPIBackend` (PLAN §5.3). |
| Phase 2 | Read-only dependency caches (`cache_mounts`, `gate_env`) exercised by the notes-app template; adding a dependency to a target is a human-approved image/cache change, never something an agent can do from the box. |
| Phase 3 | App + probe containers for black-box holdout probes (PLAN §5.4), the holdout-literal redaction check, and their red-team items. |
| ~~Option B only~~ | Broker, egress proxy and their red-team items: dropped with option B (2026-11-11). Phase 2 adds the option-A red-team items instead (PLAN §5.6). |
