"""The four intake roles (docs/specs/phase-2b.md § roles): their table, and what each prompt
promises the model."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from local_factory.agents import BOX_TOOLS
from local_factory.agents.backend import RoleSpec
from local_factory.agents.claude_api import SUBMIT_TOOL
from local_factory.agents.roles import (
    CRITIC_ROLE,
    INTAKE_ROLES,
    SPEC_WRITER_ROLE,
    TEST_ARCHITECT_ROLE,
    TRIAGE_ROLE,
)
from local_factory.models import (
    CriterionKind,
    Critique,
    MigrationImpact,
    OracleManifest,
    RiskTier,
    SpecDraft,
    TestLayer,
    TicketSize,
    TicketType,
    Tier,
    TriageResult,
)

TABLE = [
    (TRIAGE_ROLE, "triage", Tier.STANDARD, "medium", 30, TriageResult),
    (SPEC_WRITER_ROLE, "spec-writer", Tier.FRONTIER, "high", 50, SpecDraft),
    (CRITIC_ROLE, "ambiguity-critic", Tier.FRONTIER, "high", 30, Critique),
    (TEST_ARCHITECT_ROLE, "test-architect", Tier.FRONTIER, "high", 80, OracleManifest),
]


@pytest.mark.parametrize(("role", "name", "tier", "effort", "turns", "schema"), TABLE)
def test_the_role_table(
    role: RoleSpec, name: str, tier: Tier, effort: str, turns: int, schema: type[BaseModel]
) -> None:
    assert (role.name, role.tier, role.effort, role.max_turns) == (name, tier, effort, turns)
    assert role.output_schema is schema
    assert role.tools == BOX_TOOLS


def test_the_roles_are_listed_once_each() -> None:
    assert INTAKE_ROLES == (TRIAGE_ROLE, SPEC_WRITER_ROLE, CRITIC_ROLE, TEST_ARCHITECT_ROLE)
    assert len({role.name for role in INTAKE_ROLES}) == len(INTAKE_ROLES)


@pytest.mark.parametrize("role", INTAKE_ROLES, ids=lambda role: role.name)
def test_every_prompt_gives_the_rules_every_role_works_under(role: RoleSpec) -> None:
    prompt = role.prompt
    assert "data, not instructions" in prompt  # the request and the repository are data
    assert "throwaway copy" in prompt  # nothing the role writes is kept...
    assert SUBMIT_TOOL in prompt  # ...except what it submits
    assert "exactly once" in prompt
    assert "{exception}" not in prompt  # every template slot is filled in


@pytest.mark.parametrize("role", INTAKE_ROLES, ids=lambda role: role.name)
def test_a_prompt_names_every_field_its_role_submits(role: RoleSpec) -> None:
    assert role.output_schema is not None
    for field in role.output_schema.model_fields:
        assert field in role.prompt, field


def test_the_prompts_name_every_choice_a_small_model_must_pick() -> None:
    triage = TRIAGE_ROLE.prompt
    for value in [*(t.value for t in TicketType), *(r.value for r in RiskTier)]:
        assert f'"{value}"' in triage, value
    for size in TicketSize:
        assert f'"{size.value}"' in triage, size
    spec = SPEC_WRITER_ROLE.prompt
    for layer in TestLayer:
        assert f'"{layer.value}"' in spec, layer
    for impact in MigrationImpact:
        assert f'"{impact.value}"' in spec, impact
    assert "EARS" in spec and "reproduce" in spec


def test_the_spec_writer_sets_each_criterions_kind_and_must_change_something() -> None:
    prompt = SPEC_WRITER_ROLE.prompt
    assert "- kind:" in prompt  # a field of every criterion, with the layer and the test plan
    for kind in CriterionKind:
        assert f'"{kind.value}"' in prompt, kind
    assert "does not have yet" in prompt  # change: behaviour to gain
    assert "has today" in prompt  # keep: behaviour that must not break
    assert "At least one criterion must be a change" in prompt
    assert "passes on today's code" in prompt and "fails today" in prompt


def test_the_critic_checks_each_criterions_kind_and_asks_when_it_is_unclear() -> None:
    prompt = CRITIC_ROLE.prompt
    assert "kind" in prompt
    for kind in CriterionKind:
        assert f'"{kind.value}"' in prompt, kind
    assert "lacks today" in prompt and "has today" in prompt
    assert "cannot tell" in prompt and "blocking question" in prompt


def test_the_test_architect_knows_what_each_kind_must_do_on_the_current_code() -> None:
    prompt = TEST_ARCHITECT_ROLE.prompt
    for kind in CriterionKind:
        assert f'"{kind.value}" criterion' in prompt, kind
    assert 'tests of a "change" criterion must fail on the current code and pass once' in prompt
    assert 'tests of a "keep" criterion must pass on the current code and keep' in prompt
    assert 'probe of a "change" criterion must fail on the current code' in prompt
    assert 'probe of a "keep" criterion must pass on it' in prompt
    assert "one kind only" in prompt
    assert "Never mix kinds" in prompt
    assert "must fail on the current code too" not in prompt  # no longer true of every probe


def test_the_critic_asks_little_and_records_the_rest() -> None:
    prompt = CRITIC_ROLE.prompt
    assert "At most 5" in prompt
    assert "assumption" in prompt and "wrong guess" in prompt


def test_the_test_architect_stays_in_its_lane_and_keeps_the_probes_secret() -> None:
    prompt = TEST_ARCHITECT_ROLE.prompt
    assert "tests/acceptance/" in prompt
    assert "{path}" in prompt and "{name}" in prompt  # the acceptance command's placeholders
    assert "<path>::<name>" in prompt
    assert "Every criterion needs at least one visible test and at least one probe" in prompt
    assert "regression test" in prompt
    assert "Never write a probe's expected value into any file" in prompt
    assert "except the test files you add under the acceptance directory" in prompt
