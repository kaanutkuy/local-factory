"""Red-team fixtures: the acceptance fixtures, re-exported. Orchestrator-owned and protected."""

from __future__ import annotations

# Imported so pytest registers them as fixtures for this directory.
from tests.acceptance.conftest import (  # noqa: F401 - pytest fixtures, registered by import
    box_config,
    docker_runtime,
    git,
    sandbox,
    seed_repo,
    seeded,
)
from tests.acceptance.test_sandboxed_slice import world  # noqa: F401 - pytest fixture
