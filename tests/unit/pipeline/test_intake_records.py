"""Unit tests for what intake keeps in the ledger: requests, questions and their answers, and the
approval's checks on the oracle (docs/specs/phase-2b.md § intake).

The model runs themselves are in test_intake_run.py and test_intake_oracle.py.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest

from local_factory.agents.fake import Action
from local_factory.models import TicketStatus
from local_factory.oracle import oracle_ref
from local_factory.pipeline import Intake, IntakeError, questions
from local_factory.pipeline.intake import (
    ACTOR,
    MAX_REQUEST_CHARS,
    answer,
    approve,
    open_questions,
    receive,
    run_intake,
)
from tests.acceptance.intake_world import (
    REQUEST,
    ScriptedBackend,
    World,
    critique_out,
    happy_scripts,
    make_world,
)

Scripts = dict[str, list[list[Action]]]


@pytest.fixture
def gate1(tmp_path: Path) -> Iterator[World]:
    with make_world(tmp_path) as world:
        yield world


def run(w: World, ticket_id: str, scripts: Scripts | None = None) -> str:
    backend = ScriptedBackend(happy_scripts() if scripts is None else scripts)
    intake: Intake = w.intake(backend)
    return asyncio.run(run_intake(intake, ticket_id)).status


# --------------------------------------------------------------------------- receive


def test_receive_takes_the_author_as_the_actor(gate1: World) -> None:
    ticket = receive(gate1.ledger, "  Add a flag  ", author="kaan")
    [created] = gate1.ledger.events(ticket.id)
    assert (created.kind, created.actor) == ("created", "kaan")
    assert ticket.request == "Add a flag"


def test_the_longest_allowed_request_is_accepted(gate1: World) -> None:
    ticket = receive(gate1.ledger, "z" * MAX_REQUEST_CHARS)
    assert len(ticket.request) == MAX_REQUEST_CHARS
    assert ticket.title == "z" * 80


# --------------------------------------------------------------------------- questions


def test_a_ticket_without_questions_has_none(gate1: World) -> None:
    ticket = receive(gate1.ledger, REQUEST)
    assert questions(gate1.ledger, ticket.id) == []
    assert open_questions(gate1.ledger, ticket.id) == []


def test_an_answer_to_an_unknown_question_changes_nothing(gate1: World) -> None:
    ticket = receive(gate1.ledger, REQUEST)
    gate1.ledger.record_event(
        ticket.id,
        "question_answered",
        actor="human",
        data={"id": "Q-9", "answer": "stray", "answered_by": "human"},
    )
    assert questions(gate1.ledger, ticket.id) == []


def test_questions_are_folded_from_events_in_the_order_asked(gate1: World) -> None:
    ticket = receive(gate1.ledger, REQUEST)
    scripts = happy_scripts(critic=critique_out("First?", "Second?"))
    assert run(gate1, ticket.id, scripts) == "needs_info"
    answer(gate1.ledger, ticket.id, "Q-2", "two", author="kaan")
    folded = questions(gate1.ledger, ticket.id)
    assert [(q.id, q.answer, q.answered_by) for q in folded] == [
        ("Q-1", None, None),
        ("Q-2", "two", "kaan"),
    ]
    assert [q.id for q in open_questions(gate1.ledger, ticket.id)] == ["Q-1"]
    kinds = [e.kind for e in gate1.ledger.events(ticket.id)]
    assert kinds.count("question_asked") == 2 and kinds.count("question_answered") == 1


def test_an_answer_is_stored_stripped_and_attributed(gate1: World) -> None:
    ticket = receive(gate1.ledger, REQUEST)
    run(gate1, ticket.id, happy_scripts(critic=critique_out("Which flag?")))
    answered = answer(gate1.ledger, ticket.id, "Q-1", "  --shout \n", author="kaan")
    assert (answered.answer, answered.answered_by) == ("--shout", "kaan")
    event = [e for e in gate1.ledger.events(ticket.id) if e.kind == "question_answered"][-1]
    assert (event.actor, event.data) == (
        "kaan",
        {"id": "Q-1", "answer": "--shout", "answered_by": "kaan"},
    )


def test_a_question_is_asked_by_the_factory_as_its_actor(gate1: World) -> None:
    ticket = receive(gate1.ledger, REQUEST)
    run(gate1, ticket.id, happy_scripts(critic=critique_out("Which flag?")))
    asked = [e for e in gate1.ledger.events(ticket.id) if e.kind == "question_asked"]
    assert [e.actor for e in asked] == [ACTOR]


# --------------------------------------------------------------------------- approval


def proposed(w: World) -> str:
    ticket = receive(w.ledger, REQUEST)
    assert run(w, ticket.id) == "contract_proposed"
    return ticket.id


def test_approval_refuses_an_unreadable_holdout_record(gate1: World) -> None:
    ticket_id = proposed(gate1)
    gate1.holdout.path(ticket_id).write_text("this is not a holdout record")
    with pytest.raises(IntakeError, match="unusable"):
        approve(gate1.ledger, gate1.repo, gate1.holdout, ticket_id)
    assert gate1.ledger.get_ticket(ticket_id).status is TicketStatus.CONTRACT_PROPOSED


def test_approval_refuses_a_missing_oracle_ref(gate1: World) -> None:
    ticket_id = proposed(gate1)
    gate1.repo.delete_ref(oracle_ref(ticket_id))
    with pytest.raises(IntakeError, match="visible tests changed"):
        approve(gate1.ledger, gate1.repo, gate1.holdout, ticket_id)


def test_approval_says_who_approved(gate1: World) -> None:
    ticket_id = proposed(gate1)
    approved = approve(gate1.ledger, gate1.repo, gate1.holdout, ticket_id, author="kaan")
    assert approved.status is TicketStatus.APPROVED
    last = gate1.ledger.events(ticket_id)[-1]
    assert (last.actor, last.to_status) == ("kaan", TicketStatus.APPROVED)
