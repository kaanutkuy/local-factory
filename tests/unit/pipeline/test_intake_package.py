"""The pipeline package's lazy intake exports (they load on first use: the oracle imports this
package, and the oracle must stay importable first)."""

from __future__ import annotations

import subprocess
import sys

import pytest

import local_factory.pipeline as pipeline
from local_factory.pipeline import intake

LAZY_GETATTR = vars(pipeline)["__getattr__"]  # mypy is not shown the runtime-only loader
INTAKE_NAMES = [
    "Intake",
    "IntakeError",
    "IntakeOutcome",
    "IntakeStatus",
    "answer",
    "approve",
    "open_questions",
    "questions",
    "receive",
    "run_intake",
]


@pytest.mark.parametrize("name", INTAKE_NAMES)
def test_the_intake_names_come_from_the_intake_module(name: str) -> None:
    assert name in pipeline.__all__
    assert LAZY_GETATTR(name) is getattr(intake, name)
    assert getattr(pipeline, name) is getattr(intake, name)


def test_unknown_names_are_attribute_errors() -> None:
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        LAZY_GETATTR("nope")


def test_dir_lists_every_export() -> None:
    assert set(pipeline.__all__) <= set(dir(pipeline))


def test_the_sites_are_exported_eagerly() -> None:
    from local_factory.pipeline.host_slice import host_site
    from local_factory.pipeline.sandboxed_slice import box_site

    assert pipeline.host_site is host_site and pipeline.box_site is box_site


@pytest.mark.parametrize("first", ["local_factory.oracle", "local_factory.pipeline"])
def test_either_package_can_be_imported_first(first: str) -> None:
    code = (
        f"import {first}; import local_factory.pipeline.intake as m; print(m.run_intake.__name__)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "run_intake"
