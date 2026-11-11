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
action in a hardened, network-less container; red-team suite) complete. Next is Phase 2: a Claude
backend whose model loop runs on the host while every tool call it makes runs in the box, so the
API key never enters a container. Nothing here drives a real model yet.
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
scripts/verify          # ruff, ruff format, mypy --strict, pytest with ≥90% branch coverage
```

## Layout

```
src/local_factory/     the factory (models, ledger, gitops, verification, policy, agents, evidence, …)
tests/acceptance/      contract tests written before implementation (protected)
tests/unit/            unit tests
tests/integration/     tests against real git repositories
tests/fixtures/        fixture target repos (cli-target = v0's greet/slugify)
docs/PLAN.md           design and phase plan
docs/specs/            per-phase implementation specs
```

## License

MIT. See [LICENSE](LICENSE).
