"""Live: a real model turns a request into an approved contract (docs/specs/phase-2b.md § Exit
check). Protected.

Opt-in only: `uv run pytest -m live tests/live/test_intake_live.py`. Spends real tokens (cents:
every tier maps to Claude Haiku 5.5, each run is capped by `max_run_cost_usd`) and needs
`ANTHROPIC_API_KEY`, Docker and the sandbox image. Without a key it fails; it never skips.

Gate 1 runs in the box on the cli-target fixture. Open questions get a fixed answer. The test
approves the proposed contract, then measures the oracle's quality: how many of the visible tests
and probes pass on a known-good implementation (reported, not asserted: a model's tests may
disagree with this particular implementation and still be reasonable). One JSON line goes to
`$LF_LIVE_REPORT` when set (always to stdout).
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from local_factory.agents.claude_api import ClaudeAPIBackend
from local_factory.config import ClaudeConfig, FactoryConfig
from local_factory.models import Tier
from local_factory.oracle import oracle_ref, red_first
from local_factory.pipeline.intake import (
    Intake,
    IntakeOutcome,
    answer,
    approve,
    open_questions,
    receive,
    run_intake,
)
from local_factory.pipeline.sandboxed_slice import box_site
from local_factory.sandbox import Sandbox
from local_factory.sandbox.runtime import DockerRuntime
from tests.acceptance.conftest import docker_objects, small_sandbox_config
from tests.acceptance.intake_world import ACCEPTANCE, World, make_world

pytestmark = [pytest.mark.live, pytest.mark.container, pytest.mark.integration]

LIVE_MODEL = "claude-haiku-5-5"
LIVE_TIERS = {"claude": dict.fromkeys(Tier, LIVE_MODEL)}
MAX_RUN_COST_USD = 0.40
MAX_TOTAL_COST_USD = 1.50
MAX_ROUNDS = 4

REQUEST = (
    "greet.py should accept a --shout flag that prints the greeting in upper case. "
    "Without the flag the output stays exactly as it is today."
)
ANSWER = (
    "Use your best judgment: keep today's behaviour unchanged without --shout, "
    "and record your choice as an assumption."
)

REFERENCE_GREET = '''#!/usr/bin/env python3
"""A tiny polite greeter CLI."""
import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Print a polite greeting.")
    parser.add_argument("--name", required=True, help="Name of the person to greet.")
    parser.add_argument("--shout", action="store_true", help="Print the greeting in upper case.")
    args = parser.parse_args(argv)

    if not args.name:
        print("error: --name must not be empty", file=sys.stderr)
        return 2

    message = f"Hello, {args.name}!"
    print(message.upper() if args.shout else message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


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


@pytest.fixture
def gate1(tmp_path: Path) -> Iterator[World]:
    with make_world(tmp_path) as world:
        yield world


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


def reference_commit(w: World, oracle_sha: str) -> str:
    """The oracle commit plus a known-good greet.py, as a commit in the bare repository."""
    clone = w.root / "reference"
    w.git("clone", "-q", "--no-checkout", str(w.repo.path), str(clone), cwd=w.root)
    w.git("fetch", "-q", "origin", oracle_sha, cwd=clone)
    w.git("checkout", "-q", "--detach", oracle_sha, cwd=clone)
    (clone / "greet.py").write_text(REFERENCE_GREET)
    w.git("commit", "-q", "-am", "reference implementation", cwd=clone)
    w.repo.fetch_from(clone, ["HEAD:refs/heads/reference"])
    return w.repo.rev_parse("refs/heads/reference")


def test_a_real_model_turns_a_request_into_an_approved_contract(
    api_key_present: None, runtime: DockerRuntime, gate1: World, tmp_path: Path
) -> None:
    config = FactoryConfig(
        home=tmp_path / "home",
        tiers=LIVE_TIERS,
        apex_enabled=True,  # every tier is Haiku here; nothing reaches the apex model
        claude=ClaudeConfig(max_run_cost_usd=MAX_RUN_COST_USD),
    )
    site = box_site(Sandbox(runtime, small_sandbox_config(run_timeout_s=900.0, memory_mb=512)))
    intake = Intake(
        site=site,
        ledger=gate1.ledger,
        repo=gate1.repo,
        project=gate1.project,
        backend=ClaudeAPIBackend(config, transcript_dir=tmp_path / "transcripts"),
        holdout=gate1.holdout,
        work_root=gate1.work_root,
        factory_config=config,
    )
    ticket = receive(gate1.ledger, REQUEST)
    asked = 0
    outcome: IntakeOutcome | None = None
    for _ in range(MAX_ROUNDS):
        outcome = asyncio.run(run_intake(intake, ticket.id))
        if outcome.status != "needs_info":
            break
        for question in open_questions(gate1.ledger, ticket.id):
            answer(gate1.ledger, ticket.id, question.id, ANSWER)
            asked += 1
    assert outcome is not None

    green: dict[str, object] = {}
    record = gate1.holdout.load(ticket.id)
    contract = gate1.ledger.get_ticket(ticket.id).contract
    if outcome.status == "contract_proposed" and record is not None and contract is not None:
        approve(gate1.ledger, gate1.repo, gate1.holdout, ticket.id)
        oracle_sha = gate1.repo.rev_parse(oracle_ref(ticket.id))
        reference = reference_commit(gate1, oracle_sha)
        visible = [test for c in contract.criteria for test in c.tests]
        checks = asyncio.run(
            red_first(
                site=site,
                repo=gate1.repo,
                oracle_sha=reference,
                base_sha=reference,
                visible=list(dict.fromkeys(visible)),
                probes=record.probes,
                acceptance=ACCEPTANCE,
                ticket_dir=gate1.work_root / "reference",
            )
        )
        passing = [c for c in checks if c.green]
        green = {
            "visible": f"{sum(c.kind == 'visible' for c in passing)}/{len(set(visible))}",
            "holdout": f"{sum(c.kind == 'holdout' for c in passing)}/{len(record.probes)}",
            "failing": [f"{c.id}: {c.reason}" for c in checks if c not in passing],
        }

    runs = gate1.ledger.runs(ticket.id)
    cost = sum(r.usage.cost_usd for r in runs)
    final = gate1.ledger.get_ticket(ticket.id)
    report(
        {
            "task": "intake-greet-shout",
            "status": outcome.status,
            "final_status": final.status.value,
            "error": outcome.error,
            "questions_answered": asked,
            "criteria": len(contract.criteria) if contract else 0,
            "visible_tests": sum(len(c.tests) for c in contract.criteria) if contract else 0,
            "probes": len(record.probes) if record else 0,
            "red_first": [
                f"{c.id} [{'keep' if c.keep else 'change'}]: {c.reason}" for c in outcome.red_checks
            ],
            "kinds": [c.kind.value for c in contract.criteria] if contract else [],
            "oracle_attempts": sum(
                1 for e in gate1.ledger.events(ticket.id) if e.kind == "red_first"
            ),
            "green_on_reference": green,
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

    assert outcome.status == "contract_proposed", outcome
    assert final.status.value == "approved"
    assert contract is not None and record is not None
    assert all(c.tests and c.holdout_probes >= 1 for c in contract.criteria)
    assert all(c.ok for c in outcome.red_checks)  # change checks red, keep checks green
    assert {r.model for r in runs} == {LIVE_MODEL}
    assert cost < MAX_TOTAL_COST_USD
