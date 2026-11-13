"""Factory-level and project-level configuration. Orchestrator-owned.

- Factory config: ``$LOCAL_FACTORY_HOME/config.toml`` (tiers -> models, pricing, sandbox, the
  Claude model loop). Optional.
- Project config: ``factory.toml`` at the root of a target repo (commands, protected paths,
  policy thresholds, release policy). Required for every target; itself always protected.
"""

from __future__ import annotations

import os
import posixpath
import tomllib
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from pydantic import Field, ValidationError, field_validator, model_validator

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
_FACTORY_KEYS = frozenset({"tiers", "pricing", "apex_enabled", "sandbox", "claude"})
_CLAUDE_KEYS = frozenset(
    {
        "max_tokens",
        "max_retries",
        "request_timeout_s",
        "compaction",
        "compaction_trigger_tokens",
        "max_tool_result_chars",
        "max_run_cost_usd",
        "task_budget_tokens",
    }
)
_SANDBOX_KEYS = frozenset(
    {
        "image",
        "docker",
        "uid",
        "gid",
        "memory_mb",
        "cpus",
        "pids_limit",
        "work_size_mb",
        "tmp_size_mb",
        "command_timeout_s",
        "run_timeout_s",
        "max_bundle_bytes",
        "max_output_bytes",
        "cache_mounts",
        "gate_env",
    }
)


# --------------------------------------------------------------------------- sandbox config

DEFAULT_SANDBOX_IMAGE = "local-factory/sandbox:v1"
SANDBOX_UID = 10001  # the box's user: never root, never the host user (checked at runtime)
# Paths inside the box that the factory itself uses; a cache mount may not shadow them.
RESERVED_BOX_PATHS: tuple[str, ...] = (
    "/work",
    "/in",
    "/opt/local-factory",
    "/tmp",  # noqa: S108 - a path inside the container, not a host temp file
    "/home",
)


def _box_path(value: str, what: str) -> str:
    if not value.startswith("/") or posixpath.normpath(value) != value or value == "/":
        raise ValueError(f"{what} must be a normalized absolute path other than '/': {value!r}")
    if "," in value or "=" in value:
        raise ValueError(f"{what} must not contain ',' or '=': {value!r}")
    return value


class CacheMount(FrozenModel):
    """A host directory mounted read-only into gate containers (e.g. a warm uv cache)."""

    source: Path
    target: str

    @field_validator("source")
    @classmethod
    def _absolute_source(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError(f"cache mount source must be absolute: {value}")
        if "," in str(value) or "=" in str(value):
            raise ValueError(f"cache mount source must not contain ',' or '=': {value}")
        return value

    @field_validator("target")
    @classmethod
    def _box_target(cls, value: str) -> str:
        _box_path(value, "cache mount target")
        for reserved in RESERVED_BOX_PATHS:
            if value == reserved or value.startswith(reserved + "/"):
                raise ValueError(f"cache mount target {value!r} shadows the reserved {reserved}")
        return value


class SandboxConfig(FrozenModel):
    """The box every worker action, gate and export runs in (docs/specs/phase-1b.md).

    There is no setting that grants network access, root, capabilities or host mounts: those
    are fixed by the runtime, not configurable.
    """

    image: str = Field(default=DEFAULT_SANDBOX_IMAGE, min_length=1)
    docker: str = Field(default="docker", min_length=1)  # the docker CLI to run
    uid: int = Field(default=SANDBOX_UID, ge=1)
    gid: int = Field(default=SANDBOX_UID, ge=1)
    memory_mb: int = Field(default=2048, ge=64)
    cpus: float = Field(default=2.0, gt=0.0)
    pids_limit: int = Field(default=512, ge=16)
    work_size_mb: int = Field(default=1024, ge=16)  # the task workspace (tmpfs) quota
    tmp_size_mb: int = Field(default=256, ge=16)  # /tmp and $HOME inside the box, each
    command_timeout_s: float = Field(default=300.0, gt=0.0)  # one command (RunCommand, a tool)
    run_timeout_s: float = Field(default=3600.0, gt=0.0)  # one agent run, wall clock
    max_bundle_bytes: int = Field(default=100_000_000, ge=1)  # exported task bundle cap
    max_output_bytes: int = Field(default=1_000_000, ge=1024)  # kept per command or file read
    cache_mounts: tuple[CacheMount, ...] = ()
    gate_env: dict[str, str] = Field(default_factory=dict)  # extra env for gate commands

    @field_validator("gate_env")
    @classmethod
    def _plain_env(cls, value: dict[str, str]) -> dict[str, str]:
        for key, item in value.items():
            if not key.isidentifier() or "\x00" in item:
                raise ValueError(f"gate_env: invalid variable {key!r}")
        return value

    @model_validator(mode="after")
    def _distinct_cache_targets(self) -> Self:
        targets = [m.target for m in self.cache_mounts]
        if len(targets) != len(set(targets)):
            raise ValueError(f"duplicate cache mount targets: {sorted(targets)}")
        return self


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


CACHE_WRITE_FACTOR = 1.25  # a 5-minute cache write, when no explicit rate is configured
CACHE_READ_FACTOR = 0.1  # a cache hit, when no explicit rate is configured


class ModelPrice(FrozenModel):
    """USD per 1M tokens, as published by the provider.

    The factory meters *tokens* from API responses (authoritative); cost is computed from these
    configured prices, so it is as accurate as this table. ``long_*`` rates apply to a request
    whose whole prompt (input + cache reads + cache writes) exceeds ``long_context_over``. Cache
    rates default to the provider's standard multipliers of the matching input rate.
    """

    input: float = Field(ge=0.0)
    output: float = Field(ge=0.0)
    cache_write: float | None = Field(default=None, ge=0.0)  # 5-minute cache writes
    cache_read: float | None = Field(default=None, ge=0.0)  # cache hits and refreshes
    long_context_over: int | None = Field(default=None, ge=1)
    long_input: float | None = Field(default=None, ge=0.0)
    long_output: float | None = Field(default=None, ge=0.0)
    long_cache_write: float | None = Field(default=None, ge=0.0)
    long_cache_read: float | None = Field(default=None, ge=0.0)

    def cost(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """USD for one request (one sampling iteration) with these token counts."""
        prompt = input_tokens + cache_read_tokens + cache_write_tokens
        long = self.long_context_over is not None and prompt > self.long_context_over
        rate_in = self.long_input if long and self.long_input is not None else self.input
        rate_out = self.long_output if long and self.long_output is not None else self.output
        write = self.long_cache_write if long else self.cache_write
        read = self.long_cache_read if long else self.cache_read
        rate_write = write if write is not None else rate_in * CACHE_WRITE_FACTOR
        rate_read = read if read is not None else rate_in * CACHE_READ_FACTOR
        return (
            input_tokens * rate_in
            + output_tokens * rate_out
            + cache_write_tokens * rate_write
            + cache_read_tokens * rate_read
        ) / 1_000_000

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        """``cost`` without cache tokens."""
        return self.cost(input_tokens=input_tokens, output_tokens=output_tokens)


DEFAULT_TIERS: dict[str, dict[Tier, str]] = {
    "claude": {
        Tier.FAST: "claude-haiku-5-5",
        Tier.STANDARD: "claude-sonnet-5-5",
        Tier.FRONTIER: "claude-opus-5-5",
        Tier.APEX: "claude-fable-5-2",
    }
}

# Anthropic's published first-party rates (platform.claude.com/docs/en/about-claude/pricing,
# read 2026-11-13). Cache hits are 0.05x input on Opus/Sonnet 5.5 and 0.025x on Fable 5.2.
DEFAULT_PRICING: dict[str, ModelPrice] = {
    "claude-haiku-5-5": ModelPrice(
        input=0.10,
        output=0.50,
        cache_write=0.125,
        cache_read=0.01,
        long_context_over=100_000,
        long_input=0.50,
        long_output=2.50,
        long_cache_write=0.625,
        long_cache_read=0.05,
    ),
    "claude-sonnet-5-5": ModelPrice(input=2.00, output=10.00, cache_write=2.50, cache_read=0.10),
    "claude-opus-5-5": ModelPrice(input=4.00, output=20.00, cache_write=5.00, cache_read=0.20),
    "claude-fable-5-2": ModelPrice(input=8.00, output=40.00, cache_write=10.00, cache_read=0.20),
}


class ClaudeConfig(FrozenModel):
    """The ``ClaudeAPIBackend`` model loop (docs/specs/phase-2a.md), ``[claude]`` in config.toml.

    Models come from ``[tiers.claude]`` and prices from ``[pricing]``. There is deliberately no
    key setting: the API key comes from the host environment of the factory process only, and
    never enters a config file, the box, a transcript or the ledger.
    """

    max_tokens: int = Field(default=64_000, ge=1_024, le=128_000)  # per response (streamed)
    max_retries: int = Field(default=2, ge=0, le=10)  # SDK retries (408/409/429/5xx/connection)
    request_timeout_s: float = Field(default=900.0, gt=0.0)  # one HTTP request
    compaction: bool = True  # server-side compaction at a token threshold
    compaction_trigger_tokens: int = Field(default=150_000, ge=50_000)
    max_tool_result_chars: int = Field(default=40_000, ge=1_000)  # per tool result and file view
    max_run_cost_usd: float | None = Field(default=None, gt=0.0)  # when a task sets no budget
    task_budget_tokens: int | None = Field(default=None, ge=20_000)  # advisory pacing (beta)


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
    sandbox: SandboxConfig = SandboxConfig()
    claude: ClaudeConfig = ClaudeConfig()

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
                "sandbox": _table(raw, "sandbox", str(path), _SANDBOX_KEYS),
                "claude": _table(raw, "claude", str(path), _CLAUDE_KEYS),
            }
        )
    except ValidationError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
