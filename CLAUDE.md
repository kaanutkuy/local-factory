# Working in local-factory

Read `docs/PLAN.md` (design, invariants in §1) and the current phase spec in `docs/specs/` before
changing code. This file is the map; those are the territory.

## Toolchain

- `uv` only (never pip). `uv run <tool>`; `scripts/verify` is the one gate that must pass:
  ruff check, ruff format --check, mypy --strict, pytest with ≥90% branch coverage.
- `pyproject.toml`, `uv.lock`, `src/local_factory/{models,config,hashing}.py`, `CLAUDE.md` and
  `docs/` are **orchestrator-owned**. Do not edit them; request changes in your report.
- Do not add dependencies. If one seems necessary, say so in your report.

## Ownership and protected tests

- Each implementer owns specific paths (listed in the phase spec). Touch nothing else.
- `tests/acceptance/**` is the oracle. It was written before the implementation and is
  **protected**: never edit, skip, or delete it. If you believe an acceptance test is wrong, stop
  and report why — the orchestrator decides.
- Never weaken a gate: no `skip`/`xfail`, no `# pragma: no cover`, no lowered thresholds. A
  `# type: ignore[code]` or `# noqa: CODE` needs an inline justification and a mention in your report.

## Code conventions

- Python ≥ 3.12, fully typed, `from __future__ import annotations`, pathlib, small functions.
- Domain types come from `local_factory.models`; config from `local_factory.config`.
- Subprocesses: list argv only (no `shell=True` except the gate runner, which executes configured
  shell commands by design). Git always goes through `local_factory.gitops._git` with its sanitized
  environment — never the user's global git config, never an inherited `GIT_*` variable.
- Security invariant: host-side code never runs git (or anything) with `cwd`/`GIT_DIR` inside a
  worker-controlled directory. Only `local_factory/gitops/worker_side.py` operates inside worker
  directories, and in production it runs inside the sandbox.
- No network in tests. No sleeps. Tests are deterministic and isolated (`tmp_path`).
- Git identity in tests and factory commits comes from env (`local-factory <local-factory@localhost>`);
  never rely on, or modify, the machine's git config.

## Testing

- Unit tests for every unit with logic (`tests/unit/`); integration tests with real git in
  `tmp_path` (`tests/integration/`, marked `@pytest.mark.integration`).
- Run only your own tests while iterating (`uv run pytest tests/unit/test_x.py`); run `ruff` and
  `mypy` on your own paths. Do not reformat files you don't own.
