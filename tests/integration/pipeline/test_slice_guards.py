"""Integration: what the slice refuses to do, and the factory bugs it refuses to hide."""

from __future__ import annotations

from typing import Any

import pytest

from local_factory.agents import FakeBackend
from local_factory.evidence import build_evidence
from local_factory.gitops import GitError
from local_factory.models import Evidence, TaskStatus, TicketStatus
from local_factory.pipeline import SliceError

from .conftest import IMPLEMENT_SHOUT, VERIFIER_PASS, World

pytestmark = pytest.mark.integration

SLICE = "local_factory.pipeline.host_slice"


def _backend() -> FakeBackend:
    return FakeBackend({"implementer": IMPLEMENT_SHOUT, "verifier": VERIFIER_PASS})


def test_an_unknown_main_ref_raises_before_any_change(world: World) -> None:
    backend = _backend()
    with pytest.raises(GitError):
        world.run(backend, main_ref="refs/heads/nope")
    assert len(world.ledger.events(world.ticket_id)) == world.initial_events
    assert backend.calls == []
    assert not world.work_root.exists()
    assert world.refs("refs/heads/factory/") == []


def test_unbound_evidence_is_a_factory_bug(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    def stale_oracle(**kwargs: Any) -> Evidence:
        evidence = build_evidence(**kwargs)
        return evidence.model_copy(update={"oracle_version": "sha256:" + "c" * 64})

    monkeypatch.setattr(f"{SLICE}.build_evidence", stale_oracle)
    with pytest.raises(SliceError, match="oracle version mismatch"):
        world.run(_backend())
    assert world.ledger.get_ticket(world.ticket_id).status is TicketStatus.CONVERGED
    assert [e.kind for e in world.ledger.events(world.ticket_id)].count("evidence_recorded") == 0


def test_an_export_failure_after_the_commit_check_propagates(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_export(*args: Any, **kwargs: Any) -> None:
        raise GitError(("bundle", "create"), 128, "fatal: disk full")

    monkeypatch.setattr(f"{SLICE}.export_task_bundle", broken_export)
    with pytest.raises(GitError, match="disk full"):
        world.run(_backend())
    assert world.ledger.get_task("T-0001/task-1").status is TaskStatus.RUNNING
