"""Live: a real Claude model completes fixture tasks through the box (docs/specs/phase-2a.md §
Exit check; PLAN §5.3 "quality ... is measured in Phase 2, not assumed"). Protected.

Opt-in only: `uv run pytest -m live tests/live` (deselected by default). Spends real tokens (a few
US dollars at most; each run is capped by `max_run_cost_usd`) and needs `ANTHROPIC_API_KEY`,
Docker and the sandbox image. Without a key it fails; it never skips.

Each task runs the whole sandboxed slice: Sonnet (standard tier) implements, Opus (frontier)
verifies, the target's gates judge, and then a hidden check the model never saw runs on the
candidate in a fresh box. One JSON line per task goes to `$LF_LIVE_REPORT` when set (always to
stdout): outcome, hidden check, and per run the model, turns, tokens, cache reads and cost.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from local_factory.agents.claude_api import ClaudeAPIBackend
from local_factory.config import ClaudeConfig, FactoryConfig, load_project_config
from local_factory.evidence.binding import load_or_create_key
from local_factory.gitops.bare import BareRepo
from local_factory.gitops.bundles import export_base_bundle
from local_factory.ledger import Ledger
from local_factory.models import (
    Contract,
    Criterion,
    GateStatus,
    RiskTier,
    TestLayer,
    TicketStatus,
    TicketType,
)
from local_factory.pipeline.sandboxed_slice import run_sandboxed_slice
from local_factory.sandbox import Sandbox
from local_factory.sandbox.runtime import DockerRuntime
from local_factory.verification.gates import GateSpec
from tests.acceptance.conftest import Git, docker_objects, small_sandbox_config

pytestmark = [pytest.mark.live, pytest.mark.container, pytest.mark.integration]

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "cli-target"
ORACLE = "sha256:" + "c" * 64
MAX_RUN_COST_USD = 2.0
MAX_TASK_COST_USD = 5.0


@dataclass(frozen=True)
class LiveTask:
    name: str
    request: str
    criterion: str
    hidden_check: str  # bash, run on the candidate in a fresh box; exit 0 = the behaviour is right


TASKS = [
    LiveTask(
        name="greet-shout",
        request="greet.py should accept a --shout flag that prints the greeting in upper case.",
        criterion=(
            "When greet.py is run with --name NAME --shout, the system shall print the usual "
            "greeting in upper case, and without --shout it shall print it unchanged."
        ),
        hidden_check=(
            "test \"$(python3 greet.py --name 'Ada Lovelace' --shout)\" = 'HELLO, ADA LOVELACE!' "
            "&& test \"$(python3 greet.py --name 'Ada Lovelace')\" = 'Hello, Ada Lovelace!'"
        ),
    ),
    LiveTask(
        name="slugify-ascii",
        request="slugify.py should turn accented Latin letters into plain ASCII letters.",
        criterion=(
            "When slugify.py is given text with accented Latin letters, the system shall replace "
            "each with its unaccented ASCII letter (for example 'Crème Brûlée' becomes "
            "'creme-brulee'), and keep slugifying everything else as before."
        ),
        hidden_check=(
            "test \"$(python3 slugify.py 'Crème Brûlée à la Mode')\" = creme-brulee-a-la-mode "
            "&& test \"$(python3 slugify.py 'Hello,  World!')\" = hello-world "
            "&& test \"$(python3 slugify.py 'Ærøskøbing Straße')\" != ''"
        ),
    ),
]


@pytest.fixture
def api_key_present() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.fail("live tests need ANTHROPIC_API_KEY in the environment (they spend tokens)")


@pytest.fixture
def runtime() -> Iterator[DockerRuntime]:
    runtime = DockerRuntime()
    runtime.check(small_sandbox_config().image)
    yield runtime
    runtime.cleanup()
    assert docker_objects(runtime.instance) == ([], [])


def report(line: dict[str, object]) -> None:
    text = json.dumps(line, sort_keys=True)
    print(text)
    target = os.environ.get("LF_LIVE_REPORT")
    if target:
        with Path(target).open("a", encoding="utf-8") as fh:
            fh.write(text + "\n")


def turns(transcript: Path | None) -> int:
    if transcript is None or not transcript.is_file():
        return 0
    lines = transcript.read_text().splitlines()
    return sum(1 for line in lines if json.loads(line)["event"] == "response")


@pytest.mark.parametrize("task", TASKS, ids=[t.name for t in TASKS])
def test_a_real_model_completes_a_fixture_task(
    task: LiveTask, api_key_present: None, runtime: DockerRuntime, tmp_path: Path
) -> None:
    git = Git(tmp_path / "githome")
    (tmp_path / "githome").mkdir()
    target = tmp_path / "target"
    shutil.copytree(FIXTURE, target)
    git("init", "-q", cwd=target)
    git("add", "-A", cwd=target)
    git("commit", "-q", "-m", "target", cwd=target)
    state = tmp_path / "state"
    repo = BareRepo.init(state / "repo.git")
    repo.fetch_from(target, ["refs/heads/main:refs/heads/main"])
    contract = Contract(
        version=1,
        summary=task.request,
        criteria=(Criterion(id="AC-1", ears=task.criterion, layer=TestLayer.E2E),),
    )
    config = FactoryConfig(
        home=tmp_path / "home", claude=ClaudeConfig(max_run_cost_usd=MAX_RUN_COST_USD)
    )
    sandbox = Sandbox(runtime, small_sandbox_config(run_timeout_s=1_200.0, memory_mb=512))
    backend = ClaudeAPIBackend(config, transcript_dir=tmp_path / "transcripts")

    with Ledger(state / "ledger.db") as ledger:
        ticket = ledger.create_ticket(
            title=task.name,
            type=TicketType.FEATURE,
            risk=RiskTier.R1,
            request=task.request,
            actor="human",
        )
        ledger.transition(ticket.id, TicketStatus.TRIAGED, actor="triage")
        ledger.transition(ticket.id, TicketStatus.CONTRACT_PROPOSED, actor="spec")
        ledger.propose_contract(ticket.id, contract, actor="spec")
        ledger.set_oracle_version(ticket.id, ORACLE, actor="test-architect")
        ledger.freeze_contract(ticket.id, actor="human")
        outcome = asyncio.run(
            run_sandboxed_slice(
                ledger=ledger,
                repo=repo,
                project=load_project_config(target),
                backend=backend,
                sandbox=sandbox,
                ticket_id=ticket.id,
                work_root=tmp_path / "work",
                key=load_or_create_key(state / "evidence.key"),
                factory_config=config,
            )
        )
        runs = ledger.runs(ticket.id)

    hidden = None
    if outcome.candidate_sha is not None:
        bundle = tmp_path / "hidden.bundle"
        ref = export_base_bundle(repo, sha=outcome.candidate_sha, out=bundle)
        [hidden] = asyncio.run(
            sandbox.run_gates(
                [GateSpec(name="hidden", command=task.hidden_check, timeout_s=60.0)],
                bundle=bundle,
                ref=ref,
            )
        )

    cost = sum(r.usage.cost_usd for r in runs)
    report(
        {
            "task": task.name,
            "status": outcome.status,
            "hidden_check": hidden.status.value if hidden else None,
            "cost_usd": round(cost, 4),
            "runs": [
                {
                    "role": r.role,
                    "model": r.model,
                    "status": r.status,
                    "turns": turns(r.transcript_path),
                    "input_tokens": r.usage.input_tokens,
                    "output_tokens": r.usage.output_tokens,
                    "cache_read_tokens": r.usage.cache_read_tokens,
                    "cache_write_tokens": r.usage.cache_write_tokens,
                    "cost_usd": round(r.usage.cost_usd, 4),
                    "error": r.error,
                }
                for r in runs
            ],
        }
    )

    assert outcome.status == "candidate_verified", outcome
    assert hidden is not None and hidden.status is GateStatus.PASS, hidden
    assert cost < MAX_TASK_COST_USD
    assert sum(r.usage.cache_read_tokens for r in runs) > 0  # prompt caching works
