"""`test_globs` is configuration: defaults live in `config`, `DiffPolicy.from_project` passes the
project's, and the deletion rule follows them."""

from __future__ import annotations

from local_factory.config import DEFAULT_TEST_GLOBS, ProjectConfig
from local_factory.models import ChangedFile, ChangeKind
from local_factory.policy import DiffPolicy, check_changes
from local_factory.policy import diff_policy as diff_policy_module


def deleting(path: str) -> list[ChangedFile]:
    return [ChangedFile(path=path, kind=ChangeKind.DELETED)]


def test_the_default_globs_come_from_config_and_stay_importable_from_diff_policy() -> None:
    assert diff_policy_module.DEFAULT_TEST_GLOBS is DEFAULT_TEST_GLOBS
    assert DiffPolicy(protected=()).test_globs == DEFAULT_TEST_GLOBS


def test_a_project_s_globs_decide_what_counts_as_a_test() -> None:
    web = ProjectConfig(name="web", test_globs=("web/**/*.spec.ts",))
    policy = DiffPolicy.from_project(web)
    assert policy.test_globs == ("web/**/*.spec.ts",)
    assert [v.rule for v in check_changes(deleting("web/ui/a.spec.ts"), policy)] == ["test_deleted"]
    # the defaults no longer apply: this project says its tests are spec files only
    assert check_changes(deleting("tests/test_a.py"), policy) == []
    assert (
        check_changes(deleting("tests/test_a.py"), DiffPolicy.from_project(ProjectConfig(name="p")))
        != []
    )


def test_from_project_still_protects_the_factory_config() -> None:
    policy = DiffPolicy.from_project(ProjectConfig(name="p", protected=("docs/**",)))
    assert policy.protected == ("docs/**", "factory.toml")
