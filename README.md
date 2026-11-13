# local-factory

A local software factory. Requests become **contract-first tickets**; an orchestrated team of
agents builds them inside hardened sandboxes; an **independent oracle** (protected acceptance tests
plus a sealed, black-box holdout) decides when work is done; releases are **bound to evidence** for
the exact commit that ships.

```
 G1 INTAKE ──ready──▶ G2 ORCHESTRATE & BUILD ──converged──▶ G3 INTEGRATE & RELEASE
 triage → spec →      plan → implement → review →         verify-ticket@SHA → PR → CI →
 clarify → oracle →   integrate → verify ↺                 sha-pinned merge → canary → monitor
 human approves       (until converged)                    (auto-rollback)
            └──────────── verification spine: every gate has a hard exit check ────────────┘
```

**Status:** v1 rewrite — Phases 0, 1a (control plane + host vertical slice) and 1b (every worker
action in a hardened, network-less container; red-team suite) complete. Phase 2a adds
`ClaudeAPIBackend`: the model loop runs on the host on the official `anthropic` SDK, every tool
call the model makes (Anthropic's bash and text-editor tools) runs in the box, and the API key
never enters a container. It is verified end to end against an in-process fake of the API; the
opt-in check against the real API (`pytest -m live`, spends tokens) is separate. Phase 2b adds
Gate 1: `factory intake "<request>"` runs triage, a spec writer, an ambiguity critic and a test
architect; the visible acceptance tests are committed for implementers, the black-box holdout
probes are sealed in factory state, each is shown to fail on the current code (or, for behaviour
that must not change, to pass on it), and a human
answers questions and approves the frozen contract (`factory questions/answer/show/approve`).
The design, its rationale and the phase plan live in [`docs/PLAN.md`](docs/PLAN.md); the v0 bash
loop is preserved at the `legacy-v0` tag.

## Invariants

1. The ledger is the source of truth; agents never own workflow state, budgets or release decisions.
2. The oracle is independent of implementation; implementers cannot modify what judges them.
3. The sandbox is the security boundary; hooks and prompts are defense-in-depth.
4. Evidence is bound to an exact commit; no stale verification authorizes new code.
5. Autonomy is earned through measured reliability.
6. No LLM holds a credential whose scope exceeds its own sandbox.
7. Worker output crosses the boundary only as validated data; the host never runs git inside a
   worker-controlled directory.

## Development

Requires [uv](https://docs.astral.sh/uv/) and git ≥ 2.38.

```sh
uv sync                 # create the environment
uv run factory --help   # CLI
uv run factory project add PATH        # adopt a target repo (needs factory.toml with [commands] acceptance)
uv run factory intake "<request>"      # Gate 1: request -> proposed contract (needs ANTHROPIC_API_KEY, Docker)
scripts/verify          # ruff, ruff format, mypy --strict, pytest with ≥90% branch coverage
uv run pytest -m live   # opt-in: a real model on fixture tasks; needs ANTHROPIC_API_KEY, spends tokens
```

Container tests need Docker and the sandbox image (`scripts/build-images`).

## Layout

```
src/local_factory/     the factory (models, ledger, gitops, verification, policy, agents, evidence, …)
tests/acceptance/      contract tests written before implementation (protected)
tests/unit/            unit tests
tests/integration/     tests against real git repositories
tests/redteam/         attacks on the box and the model boundary that must fail (protected)
tests/live/            opt-in checks against the real Claude API (protected, spends tokens)
tests/fixtures/        fixture target repos (cli-target = v0's greet/slugify)
docs/PLAN.md           design and phase plan
docs/specs/            per-phase implementation specs
```

## License

MIT. See [LICENSE](LICENSE).
