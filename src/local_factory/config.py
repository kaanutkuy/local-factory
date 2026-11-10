"""Factory-level and project-level configuration. Orchestrator-owned.

- Factory config: ``$LOCAL_FACTORY_HOME/config.toml`` (tiers -> models, pricing). Optional.
- Project config: ``factory.toml`` at the root of a target repo (commands, protected paths,
  policy thresholds, release policy). Required for every target; itself always protected.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError, field_validator

from local_factory.models import FrozenModel, RiskTier, Tier

PROJECT_CONFIG_FILENAME = "factory.toml"
FACTORY_CONFIG_FILENAME = "config.toml"
DEFAULT_HOME = Path("~/.local-factory")


class ConfigError(Exception):
    """A configuration file is missing, unparsable, or invalid."""


# --------------------------------------------------------------------------- project config


class ReleaseMode(StrEnum):
    DARK = "dark"
    HOLD = "hold"


class PolicyConfig(FrozenModel):
    changed_line_coverage: float = Field(default=0.90, ge=0.0, le=1.0)
    mutation_score: float = Field(default=0.70, ge=0.0, le=1.0)
    flake_runs: int = Field(default=3, ge=1)


def _default_release_policy() -> dict[RiskTier, ReleaseMode]:
    return {
        RiskTier.R0: ReleaseMode.DARK,
        RiskTier.R1: ReleaseMode.DARK,
        RiskTier.R2: ReleaseMode.HOLD,
    }


class ReleaseConfig(FrozenModel):
    policy: dict[RiskTier, ReleaseMode] = Field(default_factory=_default_release_policy)
    r2_promotion_clean_releases: int = Field(default=10, ge=1)
    always_hold: tuple[str, ...] = ("destructive_migration", "auth_config", "deploy_config")

    @field_validator("policy")
    @classmethod
    def _complete_policy(cls, value: dict[RiskTier, ReleaseMode]) -> dict[RiskTier, ReleaseMode]:
        return {**_default_release_policy(), **value}


class ProjectConfig(FrozenModel):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    commands: dict[str, str] = Field(default_factory=dict)
    protected: tuple[str, ...] = (PROJECT_CONFIG_FILENAME,)
    policy: PolicyConfig = PolicyConfig()
    release: ReleaseConfig = ReleaseConfig()

    @field_validator("protected")
    @classmethod
    def _config_is_always_protected(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return value if PROJECT_CONFIG_FILENAME in value else (*value, PROJECT_CONFIG_FILENAME)


_PROJECT_SECTIONS = frozenset({"project", "commands", "paths", "policy", "release"})
_PROJECT_KEYS = frozenset({"name"})
_PATHS_KEYS = frozenset({"protected"})
_FACTORY_KEYS = frozenset({"tiers", "pricing", "apex_enabled"})


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"{path}: not found") from exc
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"{path}: unreadable: {exc}") from exc


def _table(
    raw: Mapping[str, Any], key: str, source: str, allowed: frozenset[str] | None = None
) -> Mapping[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"{source}: [{key}] must be a table")
    unknown = set(value) - allowed if allowed is not None else set()
    if unknown:
        raise ConfigError(f"{source}: unknown key(s) in [{key}]: {', '.join(sorted(unknown))}")
    return value


def parse_project_config(
    raw: Mapping[str, Any], *, source: str = PROJECT_CONFIG_FILENAME
) -> ProjectConfig:
    unknown = set(raw) - _PROJECT_SECTIONS
    if unknown:
        raise ConfigError(f"{source}: unknown section(s): {', '.join(sorted(unknown))}")
    project = _table(raw, "project", source, _PROJECT_KEYS)
    paths = _table(raw, "paths", source, _PATHS_KEYS)
    data: dict[str, Any] = {
        "name": project.get("name"),
        "commands": raw.get("commands", {}),
        "policy": raw.get("policy", {}),
        "release": raw.get("release", {}),
    }
    if "protected" in paths:
        data["protected"] = paths["protected"]  # pydantic rejects a bare string
    try:
        return ProjectConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{source}: {exc}") from exc


def load_project_config(path: Path) -> ProjectConfig:
    """Load ``factory.toml`` (pass the file path or the repo root)."""
    if path.is_dir():
        path = path / PROJECT_CONFIG_FILENAME
    return parse_project_config(_read_toml(path), source=str(path))


# --------------------------------------------------------------------------- factory config


class ModelPrice(FrozenModel):
    """USD per 1M tokens. Estimates only; the model broker meters actual spend."""

    input: float = Field(ge=0.0)
    output: float = Field(ge=0.0)
    long_context_over: int | None = Field(default=None, ge=1)
    long_input: float | None = Field(default=None, ge=0.0)
    long_output: float | None = Field(default=None, ge=0.0)

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        long = self.long_context_over is not None and input_tokens > self.long_context_over
        rate_in = self.long_input if long and self.long_input is not None else self.input
        rate_out = self.long_output if long and self.long_output is not None else self.output
        return (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000


DEFAULT_TIERS: dict[str, dict[Tier, str]] = {
    "claude": {
        Tier.FAST: "claude-haiku-5-5",
        Tier.STANDARD: "claude-sonnet-5-5",
        Tier.FRONTIER: "claude-opus-5-5",
        Tier.APEX: "claude-fable-5-2",
    }
}

DEFAULT_PRICING: dict[str, ModelPrice] = {
    "claude-haiku-5-5": ModelPrice(
        input=0.10, output=0.50, long_context_over=100_000, long_input=0.50, long_output=2.50
    ),
    "claude-sonnet-5-5": ModelPrice(input=2.00, output=10.00),
    "claude-opus-5-5": ModelPrice(input=4.00, output=20.00),
    "claude-fable-5-2": ModelPrice(input=8.00, output=40.00),
}


def _default_tiers() -> dict[str, dict[Tier, str]]:
    """A fresh copy, so no config instance can mutate the module defaults."""
    return {backend: dict(mapping) for backend, mapping in DEFAULT_TIERS.items()}


def _merge_tiers(overrides: Mapping[str, Any], source: str) -> dict[str, dict[Any, str]]:
    """Merge per tier: overriding one tier of a backend keeps that backend's other tiers."""
    merged: dict[str, dict[Any, str]] = dict(_default_tiers())
    for backend, mapping in overrides.items():
        if not isinstance(mapping, Mapping):
            raise ConfigError(f"{source}: [tiers.{backend}] must be a table")
        merged[backend] = {**merged.get(backend, {}), **mapping}
    return merged


class FactoryConfig(FrozenModel):
    home: Path
    tiers: dict[str, dict[Tier, str]] = Field(default_factory=lambda: _default_tiers())
    pricing: dict[str, ModelPrice] = Field(default_factory=lambda: dict(DEFAULT_PRICING))
    apex_enabled: bool = False

    def model_for(self, backend: str, tier: Tier) -> str:
        if tier is Tier.APEX and not self.apex_enabled:
            raise ConfigError("the apex tier is disabled (set apex_enabled = true to allow it)")
        try:
            return self.tiers[backend][tier]
        except KeyError as exc:
            raise ConfigError(f"no model configured for backend={backend!r} tier={tier}") from exc

    def project_state_dir(self, name: str) -> Path:
        return self.home / "projects" / name


def factory_home(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    raw = env.get("LOCAL_FACTORY_HOME")
    return Path(raw).expanduser() if raw else DEFAULT_HOME.expanduser()


def load_factory_config(home: Path | None = None) -> FactoryConfig:
    """Load ``config.toml`` from the factory home; a missing file means all defaults."""
    home = factory_home() if home is None else home
    path = home / FACTORY_CONFIG_FILENAME
    if not path.exists():
        return FactoryConfig(home=home)
    raw = _read_toml(path)
    unknown = set(raw) - _FACTORY_KEYS
    if unknown:
        raise ConfigError(f"{path}: unknown key(s): {', '.join(sorted(unknown))}")
    try:
        return FactoryConfig.model_validate(
            {
                "home": home,
                "tiers": _merge_tiers(_table(raw, "tiers", str(path)), str(path)),
                "pricing": {**DEFAULT_PRICING, **_table(raw, "pricing", str(path))},
                "apex_enabled": raw.get("apex_enabled", False),
            }
        )
    except ValidationError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
