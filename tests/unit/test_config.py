"""Unit tests for project and factory configuration (local_factory.config).

Tests marked "KNOWN CONTRACT BUG (reported)" fail against the current config.py on purpose: they
pin behaviour the orchestrator-owned module should have. They are not skipped or xfailed.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from local_factory.config import (
    DEFAULT_HOME,
    DEFAULT_PRICING,
    DEFAULT_TIERS,
    FACTORY_CONFIG_FILENAME,
    PROJECT_CONFIG_FILENAME,
    ClaudeConfig,
    ConfigError,
    FactoryConfig,
    ModelPrice,
    PolicyConfig,
    ProjectConfig,
    ReleaseConfig,
    ReleaseMode,
    factory_home,
    load_factory_config,
    load_project_config,
    parse_project_config,
)
from local_factory.models import RiskTier, Tier

MINIMAL = {"project": {"name": "notes"}}


def _outcome(call: Callable[[], object]) -> str | None:
    """None if `call` raises ConfigError; otherwise a description of what happened instead."""
    try:
        result = call()
    except ConfigError:
        return None
    except Exception as exc:  # reporting the wrong exception type is the point
        return f"raised {type(exc).__name__}: {exc}"
    return f"returned {result!r}"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# =========================================================================== project config


def test_minimal_project_config_has_safe_defaults() -> None:
    config = parse_project_config(MINIMAL)
    assert config.name == "notes"
    assert config.commands == {}
    assert config.protected == (PROJECT_CONFIG_FILENAME,)
    assert config.policy == PolicyConfig(
        changed_line_coverage=0.90, mutation_score=0.70, flake_runs=3
    )
    assert config.release == ReleaseConfig()


def test_full_project_config() -> None:
    config = parse_project_config(
        {
            "project": {"name": "notes-app2"},
            "commands": {"lint": "ruff check .", "unit": "pytest -q"},
            "paths": {"protected": ["migrations/**", "tests/acceptance/**"]},
            "policy": {"changed_line_coverage": 1.0, "mutation_score": 0.0, "flake_runs": 5},
            "release": {
                "policy": {"R0": "hold"},
                "r2_promotion_clean_releases": 3,
                "always_hold": ["auth_config"],
            },
        }
    )
    assert config.commands == {"lint": "ruff check .", "unit": "pytest -q"}
    assert config.protected == ("migrations/**", "tests/acceptance/**", PROJECT_CONFIG_FILENAME)
    assert config.policy.flake_runs == 5
    assert config.release.policy[RiskTier.R0] is ReleaseMode.HOLD
    assert config.release.r2_promotion_clean_releases == 3
    assert config.release.always_hold == ("auth_config",)


def test_unknown_sections_are_rejected_by_name() -> None:
    with pytest.raises(ConfigError, match=r"factory\.toml: unknown section\(s\): deploy, gates"):
        parse_project_config({**MINIMAL, "gates": {}, "deploy": {}})


def test_error_messages_name_the_source() -> None:
    with pytest.raises(ConfigError, match=r"^custom/factory\.toml: "):
        parse_project_config({"bogus": {}}, source="custom/factory.toml")


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"project": {}},
        {"project": {"name": ""}},
        {"project": {"name": "Notes"}},
        {"project": {"name": "-notes"}},
        {"project": {"name": "notes_app"}},
        {"project": {"name": "notes app"}},
        {"project": {"name": 7}},
    ],
)
def test_project_name_is_required_and_slug_shaped(raw: Mapping[str, Any]) -> None:
    with pytest.raises(ConfigError, match="name"):
        parse_project_config(raw)


@pytest.mark.parametrize(
    "protected",
    [[], ["src/**"], ["factory.toml"], ["a", "factory.toml", "b"]],
)
def test_factory_toml_is_always_protected(protected: list[str]) -> None:
    config = parse_project_config({**MINIMAL, "paths": {"protected": protected}})
    assert config.protected.count(PROJECT_CONFIG_FILENAME) == 1
    assert [p for p in config.protected if p != PROJECT_CONFIG_FILENAME] == [
        p for p in protected if p != PROJECT_CONFIG_FILENAME
    ]


def test_factory_toml_protection_also_holds_for_direct_construction() -> None:
    assert ProjectConfig(name="x", protected=("src/**",)).protected == (
        "src/**",
        PROJECT_CONFIG_FILENAME,
    )
    assert ProjectConfig(name="x").protected == (PROJECT_CONFIG_FILENAME,)


@pytest.mark.parametrize(
    "policy",
    [
        {"changed_line_coverage": 1.01},
        {"changed_line_coverage": -0.1},
        {"mutation_score": 2},
        {"flake_runs": 0},
        {"flake_runs": "three"},
        {"fail_under": 0.5},
    ],
)
def test_policy_thresholds_are_bounded_and_strict(policy: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        parse_project_config({**MINIMAL, "policy": policy})


@pytest.mark.parametrize("commands", [{"lint": ["ruff", "check"]}, {"unit": 1}, "pytest"])
def test_commands_must_be_a_table_of_strings(commands: object) -> None:
    with pytest.raises(ConfigError, match="commands"):
        parse_project_config({**MINIMAL, "commands": commands})


def test_release_policy_defaults() -> None:
    release = parse_project_config(MINIMAL).release
    assert release.policy == {
        RiskTier.R0: ReleaseMode.DARK,
        RiskTier.R1: ReleaseMode.DARK,
        RiskTier.R2: ReleaseMode.HOLD,
    }
    assert release.r2_promotion_clean_releases == 10
    assert release.always_hold == ("destructive_migration", "auth_config", "deploy_config")


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({}, ("dark", "dark", "hold")),
        ({"R1": "hold"}, ("dark", "hold", "hold")),
        ({"R2": "dark"}, ("dark", "dark", "dark")),
        ({"R0": "hold", "R1": "hold", "R2": "hold"}, ("hold", "hold", "hold")),
    ],
)
def test_partial_release_policy_merges_with_defaults(
    override: dict[str, str], expected: tuple[str, str, str]
) -> None:
    policy = parse_project_config({**MINIMAL, "release": {"policy": override}}).release.policy
    assert set(policy) == set(RiskTier)
    assert (policy[RiskTier.R0], policy[RiskTier.R1], policy[RiskTier.R2]) == expected


@pytest.mark.parametrize(
    "release",
    [
        {"policy": {"R1": "yolo"}},
        {"policy": {"R3": "dark"}},
        {"r2_promotion_clean_releases": 0},
        {"auto_rollback": True},
    ],
)
def test_invalid_release_config(release: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match="release"):
        parse_project_config({**MINIMAL, "release": release})


def test_load_project_config_from_file_or_repo_root(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "repo" / PROJECT_CONFIG_FILENAME,
        '[project]\nname = "notes"\n\n[commands]\nunit = "pytest"\n\n'
        '[paths]\nprotected = ["migrations/**"]\n',
    )
    from_file = load_project_config(path)
    assert from_file == load_project_config(path.parent)
    assert from_file.commands == {"unit": "pytest"}
    assert from_file.protected == ("migrations/**", PROJECT_CONFIG_FILENAME)


def test_load_project_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_project_config(tmp_path)
    with pytest.raises(ConfigError, match="not found"):
        load_project_config(tmp_path / "elsewhere.toml")


def test_load_project_config_invalid_toml(tmp_path: Path) -> None:
    path = _write(tmp_path / PROJECT_CONFIG_FILENAME, "[project\nname = ")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_project_config(path)


def test_load_project_config_validation_error_names_the_file(tmp_path: Path) -> None:
    path = _write(tmp_path / PROJECT_CONFIG_FILENAME, '[project]\nname = "Bad Name"\n')
    with pytest.raises(ConfigError, match=str(path)):
        load_project_config(tmp_path)


def test_config_error_is_a_plain_exception() -> None:
    assert issubclass(ConfigError, Exception)


# --------------------------------------------------------------- project config: contract bugs


def test_non_table_sections_are_config_errors() -> None:
    """KNOWN CONTRACT BUG (reported): non-table [project]/[paths] crash with Attribute/TypeError.

    `ConfigError` is documented as the error for an invalid configuration file.
    """
    cases: list[dict[str, Any]] = [
        {"project": "notes"},
        {**MINIMAL, "paths": "protected"},
        {**MINIMAL, "paths": ["src/**"]},
    ]
    wrong = {repr(raw): _outcome(partial(parse_project_config, raw)) for raw in cases}
    assert {raw: outcome for raw, outcome in wrong.items() if outcome} == {}


def test_protected_must_be_a_list_not_a_string() -> None:
    """KNOWN CONTRACT BUG (reported): `protected = "src/**"` becomes one glob per character.

    `tuple(paths["protected"])` splits the string into ("s", "r", "c", "/", "*", "*") before
    validation; pydantic itself would reject a bare string for `tuple[str, ...]`.
    """
    raw = {**MINIMAL, "paths": {"protected": "src/**"}}
    assert _outcome(lambda: parse_project_config(raw)) is None


def test_unknown_keys_in_project_and_paths_are_rejected() -> None:
    """KNOWN CONTRACT BUG (reported): typos inside [project]/[paths] are silently ignored.

    `[paths] protect = [...]` (sic) leaves only factory.toml protected without any error, while
    unknown *sections* and unknown [policy]/[release] keys are rejected.
    """
    cases: list[dict[str, Any]] = [
        {"project": {"name": "notes", "nmae": "typo"}},
        {**MINIMAL, "paths": {"protect": ["migrations/**"]}},
    ]
    wrong = {repr(raw): _outcome(partial(parse_project_config, raw)) for raw in cases}
    assert {raw: outcome for raw, outcome in wrong.items() if outcome} == {}


def test_unreadable_or_undecodable_files_are_config_errors(tmp_path: Path) -> None:
    """KNOWN CONTRACT BUG (reported): `_read_toml` lets UnicodeDecodeError / OSError escape."""
    bad_utf8 = tmp_path / "utf8" / PROJECT_CONFIG_FILENAME
    bad_utf8.parent.mkdir()
    bad_utf8.write_bytes(b'[project]\nname = "\xff"\n')
    dir_as_file = tmp_path / "dir"
    (dir_as_file / PROJECT_CONFIG_FILENAME).mkdir(parents=True)
    home = tmp_path / "home"
    (home / FACTORY_CONFIG_FILENAME).mkdir(parents=True)
    wrong = {
        "invalid utf-8": _outcome(lambda: load_project_config(bad_utf8)),
        "factory.toml is a directory": _outcome(lambda: load_project_config(dir_as_file)),
        "config.toml is a directory": _outcome(lambda: load_factory_config(home)),
    }
    assert {case: outcome for case, outcome in wrong.items() if outcome} == {}


# =========================================================================== pricing


def test_estimate_uses_per_million_rates() -> None:
    price = ModelPrice(input=2.0, output=10.0)
    assert price.estimate(0, 0) == 0.0
    assert price.estimate(1_000_000, 1_000_000) == pytest.approx(12.0)
    assert price.estimate(1_000, 500) == pytest.approx(0.002 + 0.005)


def test_long_context_rates_apply_strictly_above_the_threshold() -> None:
    price = ModelPrice(
        input=1.0, output=2.0, long_context_over=100_000, long_input=3.0, long_output=4.0
    )
    assert price.estimate(100_000, 1_000_000) == pytest.approx(0.1 + 2.0)
    assert price.estimate(100_001, 1_000_000) == pytest.approx(100_001 * 3.0 / 1e6 + 4.0)


def test_missing_long_rates_fall_back_to_base_rates() -> None:
    only_output = ModelPrice(input=1.0, output=2.0, long_context_over=10, long_output=5.0)
    assert only_output.estimate(1_000_000, 1_000_000) == pytest.approx(1.0 + 5.0)
    only_input = ModelPrice(input=1.0, output=2.0, long_context_over=10, long_input=7.0)
    assert only_input.estimate(1_000_000, 1_000_000) == pytest.approx(7.0 + 2.0)


def test_long_rates_without_a_threshold_never_apply() -> None:
    price = ModelPrice(input=1.0, output=2.0, long_input=100.0, long_output=100.0)
    assert price.estimate(10_000_000, 1_000_000) == pytest.approx(10.0 + 2.0)


@pytest.mark.parametrize(
    "fields",
    [
        {"input": -1.0, "output": 1.0},
        {"input": 1.0, "output": -1.0},
        {"input": 1.0, "output": 1.0, "long_context_over": 0},
        {"input": 1.0, "output": 1.0, "long_input": -0.5},
        {"input": 1.0, "output": 1.0, "long_output": -0.5},
        {"input": 1.0},
    ],
)
def test_model_price_constraints(fields: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="validation error"):
        ModelPrice.model_validate(fields)


def test_every_default_tier_model_has_a_default_price() -> None:
    for tiers in DEFAULT_TIERS.values():
        assert set(tiers) == set(Tier)
        for model in tiers.values():
            assert model in DEFAULT_PRICING
    long_context = [m for m, p in DEFAULT_PRICING.items() if p.long_context_over is not None]
    for model in long_context:
        price = DEFAULT_PRICING[model]
        assert price.long_context_over is not None
        above = price.long_context_over + 1
        assert price.estimate(above, 0) >= ModelPrice(
            input=price.input, output=price.output
        ).estimate(above, 0)


def test_cost_counts_cache_tokens_at_their_own_rates() -> None:
    price = DEFAULT_PRICING["claude-sonnet-5-5"]
    cost = price.cost(
        input_tokens=1_000, output_tokens=500, cache_read_tokens=2_000, cache_write_tokens=400
    )
    assert cost == pytest.approx((1_000 * 2.0 + 500 * 10.0 + 2_000 * 0.10 + 400 * 2.50) / 1e6)
    assert price.cost(input_tokens=1_000, output_tokens=500) == price.estimate(1_000, 500)


def test_cache_rates_default_to_the_standard_multipliers() -> None:
    price = ModelPrice(input=10.0, output=0.0)
    assert price.cost(input_tokens=0, output_tokens=0, cache_write_tokens=1_000_000) == (
        pytest.approx(12.5)
    )
    assert price.cost(input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000) == (
        pytest.approx(1.0)
    )


def test_long_context_is_decided_on_the_whole_prompt() -> None:
    price = DEFAULT_PRICING["claude-haiku-5-5"]
    short = price.cost(input_tokens=50_000, output_tokens=0, cache_read_tokens=50_000)
    assert short == pytest.approx((50_000 * 0.10 + 50_000 * 0.01) / 1e6)
    long = price.cost(
        input_tokens=50_000, output_tokens=1_000, cache_read_tokens=50_000, cache_write_tokens=1
    )
    assert long == pytest.approx((50_000 * 0.50 + 1_000 * 2.50 + 50_000 * 0.05 + 0.625) / 1e6)
    # A long request without explicit long cache rates derives them from the long input rate.
    derived = ModelPrice(input=1.0, output=1.0, long_context_over=10, long_input=4.0)
    assert derived.cost(input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000) == (
        pytest.approx(0.4)
    )


def test_default_cache_prices_match_the_published_rates() -> None:
    rates = {m: (p.cache_write, p.cache_read) for m, p in DEFAULT_PRICING.items()}
    assert rates == {
        "claude-haiku-5-5": (0.125, 0.01),
        "claude-sonnet-5-5": (2.50, 0.10),
        "claude-opus-5-5": (5.00, 0.20),
        "claude-fable-5-2": (10.00, 0.20),
    }


# =========================================================================== factory config


def test_model_for_maps_tiers_to_default_models(tmp_path: Path) -> None:
    config = FactoryConfig(home=tmp_path)
    for tier in (Tier.FAST, Tier.STANDARD, Tier.FRONTIER):
        assert config.model_for("claude", tier) == DEFAULT_TIERS["claude"][tier]


def test_apex_is_disabled_by_default(tmp_path: Path) -> None:
    config = FactoryConfig(home=tmp_path)
    assert config.apex_enabled is False
    with pytest.raises(ConfigError, match="apex_enabled = true"):
        config.model_for("claude", Tier.APEX)
    # The apex gate is checked before the backend lookup.
    with pytest.raises(ConfigError, match="apex"):
        config.model_for("no-such-backend", Tier.APEX)


def test_apex_when_enabled(tmp_path: Path) -> None:
    config = FactoryConfig(home=tmp_path, apex_enabled=True)
    assert config.model_for("claude", Tier.APEX) == DEFAULT_TIERS["claude"][Tier.APEX]


def test_unknown_backend_or_unmapped_tier(tmp_path: Path) -> None:
    config = FactoryConfig(home=tmp_path, tiers={"local": {Tier.FAST: "small"}})
    assert config.model_for("local", Tier.FAST) == "small"
    with pytest.raises(ConfigError, match="backend='claude'"):
        config.model_for("claude", Tier.FAST)
    with pytest.raises(ConfigError, match="tier=standard"):
        config.model_for("local", Tier.STANDARD)


def test_project_state_dir(tmp_path: Path) -> None:
    assert (
        FactoryConfig(home=tmp_path).project_state_dir("notes") == tmp_path / "projects" / "notes"
    )


@pytest.mark.parametrize("with_file", [False, True])
def test_mutating_a_config_never_changes_the_defaults(tmp_path: Path, with_file: bool) -> None:
    if with_file:
        _write(tmp_path / FACTORY_CONFIG_FILENAME, '[tiers.claude]\nfast = "custom-fast"\n')
    tiers_before, pricing_before = copy.deepcopy(DEFAULT_TIERS), copy.deepcopy(DEFAULT_PRICING)
    config = load_factory_config(tmp_path)
    assert config.tiers is not DEFAULT_TIERS and config.pricing is not DEFAULT_PRICING
    assert config.tiers["claude"] is not DEFAULT_TIERS["claude"]
    config.tiers["claude"][Tier.STANDARD] = "mutated"
    config.tiers["extra"] = {Tier.FAST: "x"}
    config.pricing["custom"] = ModelPrice(input=0.0, output=0.0)
    assert (tiers_before, pricing_before) == (DEFAULT_TIERS, DEFAULT_PRICING)
    fresh = FactoryConfig(home=tmp_path)
    assert (fresh.tiers, fresh.pricing) == (tiers_before, pricing_before)


# --------------------------------------------------------------------------- factory_home


def test_factory_home_from_env_mapping(tmp_path: Path) -> None:
    assert factory_home({"LOCAL_FACTORY_HOME": str(tmp_path)}) == tmp_path


def test_factory_home_expands_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert factory_home({"LOCAL_FACTORY_HOME": "~/factory"}) == tmp_path / "factory"


@pytest.mark.parametrize("env", [{}, {"LOCAL_FACTORY_HOME": ""}, {"OTHER": "x"}])
def test_factory_home_defaults(
    env: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert Path("~/.local-factory") == DEFAULT_HOME
    assert factory_home(env) == tmp_path / ".local-factory"


def test_factory_home_reads_the_process_environment_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_FACTORY_HOME", str(tmp_path / "fh"))
    assert factory_home() == tmp_path / "fh"
    monkeypatch.delenv("LOCAL_FACTORY_HOME")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert factory_home() == tmp_path / ".local-factory"


# --------------------------------------------------------------------------- load_factory_config


def test_missing_config_toml_means_all_defaults(tmp_path: Path) -> None:
    config = load_factory_config(tmp_path)
    assert config == FactoryConfig(home=tmp_path)
    assert (config.tiers, config.pricing, config.apex_enabled) == (
        DEFAULT_TIERS,
        DEFAULT_PRICING,
        False,
    )


def test_load_factory_config_uses_the_env_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_FACTORY_HOME", str(tmp_path))
    _write(tmp_path / FACTORY_CONFIG_FILENAME, "apex_enabled = true\n")
    config = load_factory_config()
    assert (config.home, config.apex_enabled) == (tmp_path, True)


def test_config_toml_merges_over_defaults(tmp_path: Path) -> None:
    _write(
        tmp_path / FACTORY_CONFIG_FILENAME,
        "apex_enabled = true\n\n"
        '[tiers.local]\nfast = "qwen-small"\nstandard = "qwen-large"\n\n'
        '[pricing."qwen-small"]\ninput = 0.0\noutput = 0.0\n\n'
        '[pricing."claude-opus-5-5"]\ninput = 1.0\noutput = 2.0\n',
    )
    config = load_factory_config(tmp_path)
    assert config.apex_enabled is True
    assert config.tiers["claude"] == DEFAULT_TIERS["claude"]
    assert config.model_for("local", Tier.STANDARD) == "qwen-large"
    assert config.model_for("claude", Tier.APEX) == DEFAULT_TIERS["claude"][Tier.APEX]
    assert config.pricing["qwen-small"] == ModelPrice(input=0.0, output=0.0)
    assert config.pricing["claude-opus-5-5"] == ModelPrice(input=1.0, output=2.0)
    for model, price in DEFAULT_PRICING.items():
        if model != "claude-opus-5-5":
            assert config.pricing[model] == price


@pytest.mark.parametrize(
    "text",
    [
        '[pricing."m"]\ninput = -1.0\noutput = 1.0\n',
        '[tiers.local]\nultra = "m"\n',
        "[tiers.claude]\nfast = 1\n",
        'apex_enabled = "yes please"\n',
        '[pricing."m"]\ninput = 1.0\noutput = 1.0\ncurrency = "EUR"\n',
    ],
)
def test_invalid_config_toml_names_the_file(tmp_path: Path, text: str) -> None:
    path = _write(tmp_path / FACTORY_CONFIG_FILENAME, text)
    with pytest.raises(ConfigError, match=str(path)):
        load_factory_config(tmp_path)


def test_config_toml_with_invalid_toml(tmp_path: Path) -> None:
    _write(tmp_path / FACTORY_CONFIG_FILENAME, "apex_enabled = \n")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_factory_config(tmp_path)


def test_overriding_one_tier_keeps_the_backends_other_tiers(tmp_path: Path) -> None:
    _write(tmp_path / FACTORY_CONFIG_FILENAME, '[tiers.claude]\nfast = "custom-fast"\n')
    tiers = load_factory_config(tmp_path).tiers
    assert tiers["claude"] == {**DEFAULT_TIERS["claude"], Tier.FAST: "custom-fast"}
    assert set(tiers) == set(DEFAULT_TIERS)


def test_a_new_backend_gets_only_the_tiers_it_names(tmp_path: Path) -> None:
    _write(tmp_path / FACTORY_CONFIG_FILENAME, '[tiers.local]\nstandard = "qwen"\n')
    config = load_factory_config(tmp_path)
    assert config.tiers == {**DEFAULT_TIERS, "local": {Tier.STANDARD: "qwen"}}
    assert config.model_for("local", Tier.STANDARD) == "qwen"
    with pytest.raises(ConfigError, match="backend='local' tier=fast"):
        config.model_for("local", Tier.FAST)


@pytest.mark.parametrize(
    ("text", "backend"),
    [('[tiers]\nlocal = "qwen"\n', "local"), ("[tiers]\nclaude = 1\n", "claude")],
)
def test_a_non_table_backend_under_tiers_is_a_config_error(
    tmp_path: Path, text: str, backend: str
) -> None:
    _write(tmp_path / FACTORY_CONFIG_FILENAME, text)
    with pytest.raises(ConfigError, match=rf"\[tiers\.{backend}\] must be a table"):
        load_factory_config(tmp_path)


def test_pricing_overrides_merge_with_defaults(tmp_path: Path) -> None:
    overridden = next(iter(DEFAULT_PRICING))
    _write(
        tmp_path / FACTORY_CONFIG_FILENAME,
        f'[pricing."{overridden}"]\ninput = 9.0\noutput = 9.0\n\n'
        '[pricing."new-model"]\ninput = 1.0\noutput = 2.0\nlong_context_over = 5\n',
    )
    pricing = load_factory_config(tmp_path).pricing
    assert pricing == {
        **DEFAULT_PRICING,
        overridden: ModelPrice(input=9.0, output=9.0),
        "new-model": ModelPrice(input=1.0, output=2.0, long_context_over=5),
    }


# --------------------------------------------------------------- factory config: contract bugs


def test_factory_config_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    """KNOWN CONTRACT BUG (reported): config.toml keys other than the three known ones are
    silently dropped, defeating `FactoryConfig`'s extra="forbid" (e.g. `apex_enable = true`,
    or a `[prices]` table that leaves default pricing in force).
    """
    wrong = {}
    for name, text in {
        "apex_enable": "apex_enable = true\n",
        "prices": '[prices."m"]\ninput = 1.0\noutput = 1.0\n',
    }.items():
        home = tmp_path / name
        _write(home / FACTORY_CONFIG_FILENAME, text)
        wrong[name] = _outcome(partial(load_factory_config, home))
    assert {name: outcome for name, outcome in wrong.items() if outcome} == {}


def test_factory_config_non_table_sections_are_config_errors(tmp_path: Path) -> None:
    """KNOWN CONTRACT BUG (reported): `tiers = 1` / `pricing = "x"` raise TypeError (merge)."""
    wrong = {}
    for name, text in {"tiers": "tiers = 1\n", "pricing": 'pricing = "x"\n'}.items():
        home = tmp_path / name
        _write(home / FACTORY_CONFIG_FILENAME, text)
        wrong[name] = _outcome(partial(load_factory_config, home))
    assert {name: outcome for name, outcome in wrong.items() if outcome} == {}


# --------------------------------------------------------------------------- [claude]


def test_claude_defaults(tmp_path: Path) -> None:
    claude = FactoryConfig(home=tmp_path).claude
    assert claude == ClaudeConfig()
    assert (claude.max_tokens, claude.max_retries, claude.compaction) == (64_000, 2, True)
    assert claude.compaction_trigger_tokens == 150_000
    assert (claude.max_run_cost_usd, claude.task_budget_tokens) == (None, None)
    assert "api_key" not in ClaudeConfig.model_fields


def test_claude_section_loads(tmp_path: Path) -> None:
    _write(
        tmp_path / FACTORY_CONFIG_FILENAME,
        "[claude]\nmax_tokens = 32000\ncompaction = false\nmax_run_cost_usd = 2.5\n"
        "task_budget_tokens = 40000\n",
    )
    claude = load_factory_config(tmp_path).claude
    assert (claude.max_tokens, claude.compaction) == (32_000, False)
    assert (claude.max_run_cost_usd, claude.task_budget_tokens) == (2.5, 40_000)


@pytest.mark.parametrize(
    "body",
    [
        '[claude]\napi_key = "sk-ant-nope"\n',
        "[claude]\nmax_tokens = 10\n",
        "[claude]\ncompaction_trigger_tokens = 1000\n",
        "[claude]\ntask_budget_tokens = 100\n",
        "[claude]\nmax_run_cost_usd = 0\n",
        'claude = "x"\n',
    ],
)
def test_claude_section_is_validated(tmp_path: Path, body: str) -> None:
    _write(tmp_path / FACTORY_CONFIG_FILENAME, body)
    with pytest.raises(ConfigError):
        load_factory_config(tmp_path)
