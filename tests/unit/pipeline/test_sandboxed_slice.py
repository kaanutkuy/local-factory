"""Unit tests for the sandboxed slice's box site: what it takes from the sandbox's config, how it
maps the box's export outcomes, and its bundle naming. No Docker, no git: the sandbox is a stub."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest

import local_factory.pipeline as pipeline
from local_factory.config import SandboxConfig
from local_factory.gitops import BundleLimits
from local_factory.pipeline.flow import Refused
from local_factory.pipeline.sandboxed_slice import IMAGE_KEY, _BoxSite, _BoxWorker, _unused
from local_factory.sandbox import ExportOutcome, Sandbox
from local_factory.sandbox.runtime import SandboxError
from local_factory.sandbox.workspace import TaskWorkspace

IMAGE_ID = "sha256:" + "e" * 64


class StubSandbox:
    """Just what the box site reads from a `Sandbox`; records how it was used."""

    def __init__(self, config: SandboxConfig, answer: ExportOutcome | None = None) -> None:
        self.config = config
        self.answer = answer or ExportOutcome("exported")
        self.checked = 0
        self.exports: list[tuple[object, str, str, Path]] = []

    def check(self) -> str:
        self.checked += 1
        return IMAGE_ID

    def image_id(self) -> str:
        return IMAGE_ID

    def executor(self, ws: object) -> str:
        return f"the executor of {ws}"

    async def export(self, ws: object, *, base_sha: str, message: str, out: Path) -> ExportOutcome:
        self.exports.append((ws, base_sha, message, out))
        return self.answer

    def sandbox(self) -> Sandbox:
        return cast("Sandbox", self)


# --------------------------------------------------------------------------- the site


def test_the_run_limit_is_the_configured_run_timeout() -> None:
    site = _BoxSite(StubSandbox(SandboxConfig(run_timeout_s=42.5)).sandbox())
    assert site.run_limit_s == 42.5


def test_the_bundle_cap_is_the_configured_one_and_nothing_else_changes() -> None:
    site = _BoxSite(StubSandbox(SandboxConfig(max_bundle_bytes=1234)).sandbox())
    assert site.limits == BundleLimits(max_bundle_bytes=1234)


def test_check_is_the_sandboxs_check() -> None:
    stub = StubSandbox(SandboxConfig())
    _BoxSite(stub.sandbox()).check()
    assert stub.checked == 1


def test_a_failing_check_propagates() -> None:
    class Down(StubSandbox):
        def check(self) -> str:
            raise SandboxError("the docker daemon is not reachable")

    with pytest.raises(SandboxError, match="daemon"):
        _BoxSite(Down(SandboxConfig()).sandbox()).check()


def test_the_evidence_names_the_sandbox_image() -> None:
    site = _BoxSite(StubSandbox(SandboxConfig()).sandbox())
    assert site.image_digests() == {IMAGE_KEY: IMAGE_ID} == {"sandbox": IMAGE_ID}


# --------------------------------------------------------------------------- export -> submission


WORKSPACE = cast("TaskWorkspace", "the-workspace")


def test_the_backend_acts_on_slash_work_through_the_workspaces_executor() -> None:
    worker = _BoxWorker(StubSandbox(SandboxConfig()).sandbox(), WORKSPACE)
    assert worker.handle.work_dir == Path("/work")
    assert str(worker.handle.executor) == "the executor of the-workspace"


def submit(stub: StubSandbox, out: Path) -> Path | Refused:
    worker = _BoxWorker(stub.sandbox(), WORKSPACE)
    return asyncio.run(worker.submit(base="a" * 40, message="T-0001/task-1: goal", out=out))


def test_an_exported_bundle_is_the_submission(tmp_path: Path) -> None:
    stub = StubSandbox(SandboxConfig())
    out = tmp_path / "task.bundle"
    assert submit(stub, out) == out
    assert stub.exports == [("the-workspace", "a" * 40, "T-0001/task-1: goal", out)]


@pytest.mark.parametrize(
    "answer",
    [
        ExportOutcome("no_changes", "the task made no changes"),
        ExportOutcome("worker_git_error", "export failed: exit status 4"),
        ExportOutcome("bundle_too_large", "the task bundle exceeded 99 bytes"),
    ],
    ids=lambda answer: answer.status,
)
def test_any_other_outcome_is_refused_under_its_own_name(
    tmp_path: Path, answer: ExportOutcome
) -> None:
    assert submit(StubSandbox(SandboxConfig(), answer), tmp_path / "x.bundle") == Refused(
        answer.status, answer.detail
    )


# --------------------------------------------------------------------------- bundle names


def test_an_unused_name_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "verify-abc-1.bundle"
    assert _unused(path) == path


def test_a_taken_name_gets_a_unique_variant(tmp_path: Path) -> None:
    taken = tmp_path / "verify-abc-1.bundle"
    taken.write_bytes(b"another run's bundle")
    first, second = _unused(taken), _unused(taken)
    assert first != taken and first.parent == tmp_path
    assert first.name.startswith("verify-abc-1-") and first.suffix == ".bundle"
    assert first != second
    assert not first.exists()


def test_the_package_exports_both_slices() -> None:
    assert callable(pipeline.run_host_slice) and callable(pipeline.run_sandboxed_slice)
    assert "run_sandboxed_slice" in pipeline.__all__
