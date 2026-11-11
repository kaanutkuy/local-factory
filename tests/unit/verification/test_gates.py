from __future__ import annotations

import os
import shlex
import signal
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from local_factory.config import ProjectConfig
from local_factory.models import GateResult, GateStatus
from local_factory.verification import gates
from local_factory.verification.gates import (
    NOTE_PREFIX,
    GateSpec,
    OutputTail,
    gates_from_project,
    run_gate,
    run_gates,
    skipped_result,
)

G = GateStatus


@pytest.fixture(autouse=True)
def gate_tmp_root(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Every gate's HOME/TMPDIR is created here; nothing may be left behind after any test."""
    root = tmp_path_factory.mktemp("gate-homes")
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    yield root
    assert list(root.iterdir()) == [], "a gate left its HOME/TMPDIR behind"


@pytest.fixture
def long_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make a leaked output pipe obvious: run_gate would then wait 60s instead of returning."""
    monkeypatch.setattr(gates, "_DRAIN_GRACE_S", 60.0)


# --------------------------------------------------------------------------- GateSpec


def test_gate_spec_defaults() -> None:
    spec = GateSpec(name="unit", command="pytest")
    assert spec.timeout_s == 900.0


@pytest.mark.parametrize(
    ("name", "command", "timeout_s", "message"),
    [
        ("", "true", 1.0, "name"),
        ("unit", "", 1.0, "blank"),
        ("unit", "  \n", 1.0, "blank"),
        ("unit", "true", 0.0, "timeout_s"),
        ("unit", "true", -1.0, "timeout_s"),
        ("unit", "true", float("nan"), "timeout_s"),
        ("unit", "true", float("inf"), "timeout_s"),
    ],
)
def test_gate_spec_rejects_invalid_values(
    name: str, command: str, timeout_s: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        GateSpec(name=name, command=command, timeout_s=timeout_s)


# --------------------------------------------------------------------------- gates_from_project


def test_gates_from_project_custom_names_keep_the_given_order() -> None:
    config = ProjectConfig(
        name="p", commands={"lint": "l", "unit": "u", "smoke": "s", "serve": "x"}
    )
    specs = gates_from_project(config, ("smoke", "unit", "missing", "lint"))
    assert [(s.name, s.command, s.timeout_s) for s in specs] == [
        ("smoke", "s", 900.0),
        ("unit", "u", 900.0),
        ("lint", "l", 900.0),
    ]


def test_gates_from_project_omits_blank_commands() -> None:
    config = ProjectConfig(name="p", commands={"lint": "", "unit": "   ", "e2e": "run-e2e"})
    assert [s.name for s in gates_from_project(config)] == ["e2e"]


def test_gates_from_project_with_no_names_or_no_commands() -> None:
    assert gates_from_project(ProjectConfig(name="p", commands={"unit": "u"}), ()) == []
    assert gates_from_project(ProjectConfig(name="p")) == []


def test_gates_from_project_rejects_duplicate_names() -> None:
    config = ProjectConfig(name="p", commands={"unit": "u"})
    with pytest.raises(ValueError, match="unit"):
        gates_from_project(config, ["unit", "lint", "unit"])


def test_gates_from_project_rejects_a_bare_string() -> None:
    config = ProjectConfig(name="p", commands={"unit": "u"})
    with pytest.raises(TypeError, match="single string"):
        gates_from_project(config, "unit")


# --------------------------------------------------------------------------- run_gate: outcome


def test_exit_codes_map_to_pass_and_fail(tmp_path: Path) -> None:
    passed = run_gate(GateSpec(name="lint", command="true"), cwd=tmp_path)
    failed = run_gate(GateSpec(name="lint", command="exit 1"), cwd=tmp_path)
    assert (passed.status, passed.exit_code, passed.output_tail) == (G.PASS, 0, "")
    assert (failed.status, failed.exit_code) == (G.FAIL, 1)
    assert 0 <= passed.duration_s < 10


def test_death_by_signal_is_a_failure_with_negative_exit_code(tmp_path: Path) -> None:
    result = run_gate(GateSpec(name="unit", command="kill -KILL $$"), cwd=tmp_path)
    assert (result.status, result.exit_code) == (G.FAIL, -signal.SIGKILL)


def test_command_that_cannot_start_is_an_error(tmp_path: Path) -> None:
    result = run_gate(GateSpec(name="unit", command="true"), cwd=tmp_path / "missing")
    assert (result.name, result.status, result.exit_code) == ("unit", G.ERROR, None)
    assert "could not be started" in result.output_tail
    assert "missing" in result.output_tail


def test_runs_in_cwd(tmp_path: Path) -> None:
    result = run_gate(GateSpec(name="unit", command="pwd -P"), cwd=tmp_path)
    assert result.output_tail.strip() == str(tmp_path.resolve())


def test_stdin_is_closed(tmp_path: Path) -> None:
    result = run_gate(
        GateSpec(name="unit", command="cat; read -r line || echo stdin-eof", timeout_s=30),
        cwd=tmp_path,
    )
    assert (result.status, result.output_tail) == (G.PASS, "stdin-eof\n")


# --------------------------------------------------------------------------- run_gate: env


BASH_ADDED = {"PWD", "SHLVL", "_", "OLDPWD"}  # bash exports these to its children itself


def gate_environment(result_tail: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in result_tail.splitlines())


def test_host_environment_is_not_visible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECRET_TOKEN", "hunter2")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    monkeypatch.setenv("LC_ALL", "de_DE.UTF-8")
    result = run_gate(GateSpec(name="unit", command="env"), cwd=tmp_path)
    assert result.status is G.PASS
    assert "hunter2" not in result.output_tail and "ghp_secret" not in result.output_tail
    environment = gate_environment(result.output_tail)
    assert set(environment) - BASH_ADDED == {"PATH", "LANG", "HOME", "TMPDIR"}


def test_minimal_environment_values(tmp_path: Path, gate_tmp_root: Path) -> None:
    result = run_gate(GateSpec(name="unit", command="env"), cwd=tmp_path)
    environment = gate_environment(result.output_tail)
    assert environment["PATH"] == os.environ["PATH"]
    assert environment["LANG"] == "C.UTF-8"
    home = Path(environment["HOME"])
    assert environment["TMPDIR"] == str(home)
    assert home.parent == gate_tmp_root
    assert home != Path.home()
    assert not home.exists()


def test_home_is_fresh_writable_and_removed_per_gate(tmp_path: Path) -> None:
    spec = GateSpec(name="unit", command='touch "$HOME/state" && mktemp && echo "$HOME"')
    first, second = run_gate(spec, cwd=tmp_path), run_gate(spec, cwd=tmp_path)
    assert (first.status, second.status) == (G.PASS, G.PASS)
    first_temp, first_home = first.output_tail.split()
    second_home = second.output_tail.split()[1]
    assert first_temp.startswith(first_home + "/")  # mktemp honours TMPDIR
    assert first_home != second_home
    assert not Path(first_home).exists() and not Path(second_home).exists()


def test_home_with_read_only_content_is_still_removed(tmp_path: Path) -> None:
    command = 'mkdir -p "$HOME/ro/sub" && touch "$HOME/ro/sub/f" && chmod -R a-w "$HOME/ro"'
    assert run_gate(GateSpec(name="unit", command=command), cwd=tmp_path).status is G.PASS
    # gate_tmp_root asserts on teardown that nothing was left behind


def test_home_is_removed_after_a_timeout(tmp_path: Path) -> None:
    result = run_gate(
        GateSpec(name="e2e", command='echo "$HOME"; touch "$HOME/f"; sleep 30', timeout_s=1.0),
        cwd=tmp_path,
    )
    assert result.status is G.ERROR
    assert not Path(result.output_tail.splitlines()[0]).exists()


def test_home_is_removed_when_the_gate_cannot_start(tmp_path: Path, gate_tmp_root: Path) -> None:
    result = run_gate(GateSpec(name="unit", command="true"), cwd=tmp_path / "missing")
    assert result.status is G.ERROR
    assert list(gate_tmp_root.iterdir()) == []


def test_env_overrides_go_on_top(tmp_path: Path) -> None:
    result = run_gate(
        GateSpec(name="unit", command="env"),
        cwd=tmp_path,
        env={"HOME": "/custom-home", "LANG": "C", "GATE_VAR": "x"},
    )
    environment = gate_environment(result.output_tail)
    assert (environment["HOME"], environment["LANG"], environment["GATE_VAR"]) == (
        "/custom-home",
        "C",
        "x",
    )
    assert environment["TMPDIR"] != "/custom-home"


def test_env_may_replace_path_without_breaking_bash(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    tool = bin_dir / "only-here"
    tool.write_text("#!/bin/sh\necho found-it\n")
    tool.chmod(0o755)
    result = run_gate(
        GateSpec(name="unit", command="only-here"), cwd=tmp_path, env={"PATH": str(bin_dir)}
    )
    assert (result.status, result.output_tail) == (G.PASS, "found-it\n")


def test_missing_host_path_falls_back_to_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PATH")
    result = run_gate(GateSpec(name="unit", command='echo "$PATH"'), cwd=tmp_path)
    assert result.output_tail == f"{os.defpath}\n"


def test_gate_env_helper() -> None:
    assert gates.gate_env("/h", {"X": "1"}) == {
        "PATH": os.environ["PATH"],
        "LANG": "C.UTF-8",
        "HOME": "/h",
        "TMPDIR": "/h",
        "X": "1",
    }


def test_env_does_not_leak_into_os_environ(tmp_path: Path) -> None:
    run_gate(GateSpec(name="unit", command="true"), cwd=tmp_path, env={"LF_ONLY_GATE": "1"})
    assert "LF_ONLY_GATE" not in os.environ


# --------------------------------------------------------------------------- run_gate: output


def test_stdout_and_stderr_are_interleaved_in_order(tmp_path: Path) -> None:
    result = run_gate(
        GateSpec(name="unit", command="echo out; echo err >&2; echo out2"), cwd=tmp_path
    )
    assert result.output_tail == "out\nerr\nout2\n"


@pytest.mark.parametrize(
    ("tail_bytes", "expected"),
    [(0, ""), (3, "def"), (6, "abcdef"), (100, "abcdef")],
)
def test_tail_keeps_exactly_the_last_bytes(tmp_path: Path, tail_bytes: int, expected: str) -> None:
    result = run_gate(
        GateSpec(name="unit", command="printf abcdef"), cwd=tmp_path, tail_bytes=tail_bytes
    )
    assert result.output_tail == expected


def test_negative_tail_bytes_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tail_bytes"):
        run_gate(GateSpec(name="unit", command="true"), cwd=tmp_path, tail_bytes=-1)


def test_large_output_is_streamed_and_bounded(tmp_path: Path) -> None:
    # Far more than a pipe buffer: the gate would block forever if output were not drained.
    result = run_gate(
        GateSpec(name="unit", command="yes | head -c 3000000; printf END", timeout_s=60),
        cwd=tmp_path,
        tail_bytes=100,
    )
    assert result.status is G.PASS
    assert len(result.output_tail) == 100
    assert result.output_tail.endswith("y\nEND")


def test_non_utf8_output_is_replaced_not_fatal(tmp_path: Path) -> None:
    result = run_gate(GateSpec(name="unit", command=r"printf '\377\376ok'"), cwd=tmp_path)
    assert result.status is G.PASS
    assert result.output_tail == "��ok"


@pytest.mark.parametrize(("tail_bytes", "expected"), [(5, "éé"), (4, "éé"), (3, "é"), (1, "")])
def test_truncation_drops_a_split_utf8_character(
    tmp_path: Path, tail_bytes: int, expected: str
) -> None:
    result = run_gate(
        GateSpec(name="unit", command=r"printf '\303\251\303\251\303\251'"),
        cwd=tmp_path,
        tail_bytes=tail_bytes,
    )
    assert result.output_tail == expected


def test_untruncated_output_keeps_leading_continuation_bytes(tmp_path: Path) -> None:
    # Only a cut made by the tail is repaired; invalid bytes the gate itself wrote are kept.
    result = run_gate(GateSpec(name="unit", command=r"printf '\251ok'"), cwd=tmp_path)
    assert result.output_tail == "�ok"


# --------------------------------------------------------------------------- run_gate: processes


@pytest.mark.usefixtures("long_grace")
def test_timeout_kills_background_children_too(tmp_path: Path) -> None:
    started = time.monotonic()
    result = run_gate(
        GateSpec(name="e2e", command="echo begin; sleep 30 & sleep 30 & wait", timeout_s=1.0),
        cwd=tmp_path,
    )
    # Had a child survived holding the output pipe, run_gate would have waited the 60s grace.
    assert time.monotonic() - started < 10
    assert (result.status, result.exit_code) == (G.ERROR, None)
    assert result.output_tail.startswith("begin\n")
    assert "timed out after 1s" in result.output_tail
    assert "escaped" not in result.output_tail


def test_timeout_note_respects_the_tail_bound(tmp_path: Path) -> None:
    result = run_gate(
        GateSpec(name="e2e", command="sleep 30", timeout_s=0.2), cwd=tmp_path, tail_bytes=20
    )
    assert result.status is G.ERROR
    assert len(result.output_tail) <= 20
    assert result.output_tail.endswith("was killed\n")


@pytest.mark.usefixtures("long_grace")
def test_background_stragglers_are_killed_after_a_normal_exit(tmp_path: Path) -> None:
    started = time.monotonic()
    result = run_gate(GateSpec(name="unit", command="sleep 30 & echo started"), cwd=tmp_path)
    assert time.monotonic() - started < 10
    assert (result.status, result.output_tail) == (G.PASS, "started\n")


def test_process_that_escapes_the_group_cannot_hang_the_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gates, "_DRAIN_GRACE_S", 0.2)
    os.mkfifo(tmp_path / "ready")
    (tmp_path / "escape.py").write_text(
        "import os, time\nos.setsid()\nopen('ready', 'w').close()\ntime.sleep(60)\n"
    )
    command = (
        f"{shlex.quote(sys.executable)} escape.py & echo $! > escaped.pid; "
        "cat ready; echo leader-done"
    )
    started = time.monotonic()
    try:
        result = run_gate(GateSpec(name="unit", command=command, timeout_s=30), cwd=tmp_path)
    finally:
        os.kill(int((tmp_path / "escaped.pid").read_text()), signal.SIGKILL)
    assert time.monotonic() - started < 10
    assert (result.status, result.exit_code) == (G.PASS, 0)
    assert result.output_tail.startswith("leader-done\n")
    assert "escaped the gate's process group" in result.output_tail


# --------------------------------------------------------------------------- run_gates


def test_run_gates_forwards_cwd_and_env(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x")
    results = run_gates(
        [GateSpec(name="unit", command='test -f f.txt && test "$V" = 1')],
        cwd=tmp_path,
        env={"V": "1"},
    )
    assert [(r.name, r.status) for r in results] == [("unit", G.PASS)]


def test_run_gates_with_no_specs() -> None:
    assert run_gates([], cwd=Path(), fail_fast=True) == []


def test_fail_fast_stops_on_error_and_names_the_stopper(tmp_path: Path) -> None:
    specs = [
        GateSpec(name="lint", command="sleep 30", timeout_s=0.2),
        GateSpec(name="unit", command="touch ran-unit"),
        GateSpec(name="e2e", command="touch ran-e2e"),
    ]
    results = run_gates(specs, cwd=tmp_path, fail_fast=True)
    assert [(r.name, r.status) for r in results] == [
        ("lint", G.ERROR),
        ("unit", G.SKIPPED),
        ("e2e", G.SKIPPED),
    ]
    assert not list(tmp_path.glob("ran-*"))
    for skipped in results[1:]:
        assert skipped.exit_code is None
        assert "'lint'" in skipped.output_tail and "'error'" in skipped.output_tail


def test_without_fail_fast_every_gate_runs(tmp_path: Path) -> None:
    specs = [
        GateSpec(name="lint", command="exit 2"),
        GateSpec(name="unit", command="touch ran-unit"),
    ]
    results = run_gates(specs, cwd=tmp_path)
    assert [(r.status, r.exit_code) for r in results] == [(G.FAIL, 2), (G.PASS, 0)]
    assert (tmp_path / "ran-unit").exists()


# --------------------------------------------------------------------------- shared helpers


def test_skipped_result_is_what_fail_fast_records(tmp_path: Path) -> None:
    stopper = GateResult(name="lint", status=G.ERROR)
    assert skipped_result(GateSpec("unit", "pytest"), stopper) == GateResult(
        name="unit",
        status=G.SKIPPED,
        output_tail=(
            "[local-factory] not run: gate 'lint' finished with status 'error' and fail_fast is set"
        ),
    )
    first, skipped = run_gates(
        [GateSpec("lint", "exit 1"), GateSpec("unit", "echo ran")], cwd=tmp_path, fail_fast=True
    )
    assert skipped == skipped_result(GateSpec("unit", "echo ran"), first)


def test_output_tail_keeps_whole_characters_and_prefixes_notes() -> None:
    assert NOTE_PREFIX == "[local-factory] "
    tail = OutputTail(8)
    tail.feed("é".encode() * 5)  # 10 bytes: the last 8 are four whole characters
    assert tail.text() == "éééé"
    tail.feed(b"x")  # now the cut falls inside an "é": its stray byte is dropped
    assert tail.text() == "éééx"
    noted = OutputTail(100)
    noted.feed(b"out")
    noted.note("done")
    assert noted.text() == "out\n[local-factory] done\n"
