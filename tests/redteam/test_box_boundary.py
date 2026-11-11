"""Red-team v1: the box boundary (docs/specs/phase-1b.md § red-team, PLAN §5.6). Protected.

Every check is a denial observed from outside the box: canaries planted on the host must never be
found inside or come back out, host listeners must never see a connection, and limits must hold.
"""

from __future__ import annotations

import asyncio
import base64
import os
import secrets
import socket
import subprocess
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from local_factory.agents.tools import CommandResult
from local_factory.config import SandboxConfig
from local_factory.evidence.binding import load_or_create_key
from local_factory.gitops.bare import BareRepo
from local_factory.gitops.bundles import export_base_bundle
from local_factory.ledger import Ledger
from local_factory.models import RiskTier, TicketType
from local_factory.sandbox import Sandbox
from local_factory.sandbox.runtime import DockerRuntime
from tests.acceptance.conftest import Git, Seeded, docker_objects

pytestmark = [pytest.mark.container, pytest.mark.integration]


def in_box(
    sandbox: Sandbox, bundle: Path, ref: str, *commands: Sequence[str], timeout_s: float = 120.0
) -> list[CommandResult]:
    async def scenario() -> list[CommandResult]:
        with sandbox.workspace() as ws:
            await sandbox.materialize(ws, bundle=bundle, ref=ref)
            tools = sandbox.executor(ws)
            return [await tools.run_command(argv, timeout_s=timeout_s) for argv in commands]

    return asyncio.run(scenario())


def sh(script: str) -> list[str]:
    return ["sh", "-c", script]


def search_for(value: str) -> list[list[str]]:
    """Commands that look for `value` everywhere in the box. The value is passed encoded, so it
    never appears in a command line, an environment or a file unless the box really has it."""
    encoded = base64.b64encode(value.encode()).decode()
    decode = f'c="$(echo {encoded} | base64 -d)"'
    return [
        sh(f'{decode}; grep -rlsF "$c" / --exclude-dir=proc --exclude-dir=sys --exclude-dir=dev'),
        sh(f'{decode}; for e in /proc/[0-9]*/environ; do tr "\\0" "\\n" < "$e"; done 2>/dev/null'),
        ["env"],
    ]


@pytest.fixture
def canary() -> str:
    return "lfcanary" + secrets.token_hex(16)


# --------------------------------------------------------------------------- identity


def test_the_box_is_unprivileged(
    sandbox: Sandbox, seeded: Seeded, box_config: SandboxConfig
) -> None:
    uid, status, rootfs, sockets, limits = in_box(
        sandbox,
        seeded.bundle,
        seeded.ref,
        ["id", "-u"],
        sh("grep -E '^(CapInh|CapPrm|CapEff|CapBnd|CapAmb|NoNewPrivs):' /proc/self/status"),
        sh("touch /usr/lf-probe 2>&1; touch /etc/lf-probe 2>&1; touch /opt/local-factory/x 2>&1"),
        sh("ls /var/run/docker.sock /run/docker.sock /var/run/docker 2>&1; echo checked"),
        sh("cat /sys/fs/cgroup/pids.max /sys/fs/cgroup/memory.max"),
    )
    assert int(uid.output.strip()) not in (0, os.getuid())
    fields = dict(line.split(":", 1) for line in status.output.strip().splitlines())
    for cap in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"):
        assert int(fields[cap].strip(), 16) == 0, cap
    assert fields["NoNewPrivs"].strip() == "1"
    assert rootfs.output.count("Read-only file system") == 3
    assert sockets.output.count("No such file or directory") == 3
    pids, memory = limits.output.split()
    assert int(pids) == box_config.pids_limit
    assert int(memory) == box_config.memory_mb * 1024 * 1024


def test_no_host_home_or_host_paths(sandbox: Sandbox, seeded: Seeded, tmp_path: Path) -> None:
    (home,) = in_box(
        sandbox,
        seeded.bundle,
        seeded.ref,
        sh(
            f"ls -A /home; test -e '{Path.home()}' && echo HOME-VISIBLE; "
            f"test -e '{tmp_path}' && echo TMP-VISIBLE; echo done"
        ),
    )
    assert home.output.split() == ["sandbox", "done"]


# --------------------------------------------------------------------------- visibility


def test_host_secrets_never_reach_the_box(
    box_config: SandboxConfig,
    seed_repo: Path,
    tmp_path: Path,
    git: Git,
    canary: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A file on the host, outside anything mounted.
    (tmp_path / "host-secret.txt").write_text(canary)
    # Credentials and other secrets in the factory's environment.
    for name in (
        "LF_CANARY",
        "ANTHROPIC_API_KEY",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
    ):
        monkeypatch.setenv(name, canary)
    # Factory state: the ledger, the evidence key's directory, and another ref in the bare repo.
    state = tmp_path / "state"
    with Ledger(state / "ledger.db") as ledger:
        ledger.create_ticket(
            title="t", type=TicketType.FEATURE, risk=RiskTier.R1, request=canary, actor="h"
        )
    load_or_create_key(state / "evidence.key")
    (state / "notes.txt").write_text(canary)
    git("checkout", "-q", "-b", "secret", cwd=seed_repo)
    (seed_repo / "secret.txt").write_text(canary)
    git("add", "secret.txt", cwd=seed_repo)
    git("commit", "-q", "-m", "secret", cwd=seed_repo)
    git("checkout", "-q", "main", cwd=seed_repo)
    repo = BareRepo.init(state / "repo.git")
    repo.fetch_from(
        seed_repo, ["refs/heads/main:refs/heads/main", "refs/heads/secret:refs/heads/secret"]
    )
    bundle = tmp_path / "base.bundle"
    ref = export_base_bundle(repo, sha=repo.rev_parse("refs/heads/main"), out=bundle)

    # A runtime created now, with the secrets already in the environment.
    runtime = DockerRuntime()
    try:
        results = in_box(Sandbox(runtime, box_config), bundle, ref, *search_for(canary))
    finally:
        runtime.cleanup()
    assert docker_objects(runtime.instance) == ([], [])
    assert results[0].output.strip() == ""  # no file in the box holds it
    for result in results:
        assert canary not in result.output


def test_workspaces_are_invisible_to_each_other(
    sandbox: Sandbox, seeded: Seeded, canary: str
) -> None:
    async def scenario() -> list[CommandResult]:
        with sandbox.workspace() as mine, sandbox.workspace() as theirs:
            await sandbox.materialize(mine, bundle=seeded.bundle, ref=seeded.ref)
            await sandbox.materialize(theirs, bundle=seeded.bundle, ref=seeded.ref)
            await sandbox.executor(theirs).write_file("their-secret.txt", canary)
            spy = sandbox.executor(mine)
            return [await spy.run_command(argv, timeout_s=120) for argv in search_for(canary)]

    for result in asyncio.run(scenario()):
        assert canary not in result.output
        assert "their-secret" not in result.output


# --------------------------------------------------------------------------- network


class Listener:
    """A host TCP listener on every interface that counts the connections it accepts."""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", 0))  # noqa: S104 - deliberately reachable, to prove the box cannot reach it
        self.sock.listen()
        self.port = self.sock.getsockname()[1]
        self.accepted = 0
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.accepted += 1
            conn.close()

    def close(self) -> None:
        self.sock.close()


@pytest.fixture
def listener() -> Iterator[Listener]:
    server = Listener()
    yield server
    server.close()


def bridge_gateway() -> str:
    out = subprocess.run(
        [
            "docker",
            "network",
            "inspect",
            "bridge",
            "--format",
            "{{(index .IPAM.Config 0).Gateway}}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def test_nothing_is_reachable(sandbox: Sandbox, seeded: Seeded, listener: Listener) -> None:
    targets = [
        "127.0.0.1",
        bridge_gateway(),
        "192.0.2.1",
    ]  # loopback, the host's bridge, TEST-NET-1
    probe = (
        "import socket\n"
        f"for host in {targets!r}:\n"
        "    s = socket.socket()\n"
        "    s.settimeout(3)\n"
        "    try:\n"
        f"        s.connect((host, {listener.port}))\n"
        "        print('CONNECTED', host)\n"
        "    except OSError as exc:\n"
        "        print('refused', host, exc)\n"
        "for name in ('example.com', 'host.docker.internal'):\n"
        "    try:\n"
        "        print('RESOLVED', name, socket.getaddrinfo(name, 443))\n"
        "    except OSError as exc:\n"
        "        print('unresolved', name, exc)\n"
    )
    interfaces, attempts = in_box(
        sandbox, seeded.bundle, seeded.ref, ["ls", "/sys/class/net"], ["python3", "-c", probe]
    )
    assert interfaces.output.split() == ["lo"]
    assert attempts.ok, attempts.output
    assert "CONNECTED" not in attempts.output
    assert "RESOLVED" not in attempts.output
    assert listener.accepted == 0


# --------------------------------------------------------------------------- resources


def test_limits_hold_and_nothing_outlives_its_box(
    sandbox: Sandbox, seeded: Seeded, box_config: SandboxConfig
) -> None:
    many = (
        "import os, signal\n"
        "n = 0\n"
        "while n < 10000:\n"
        "    try:\n"
        "        pid = os.fork()\n"
        "    except OSError:\n"
        "        break\n"
        "    if pid == 0:\n"
        "        signal.pause()\n"
        "        os._exit(0)\n"
        "    n += 1\n"
        "print('processes', n)\n"
    )
    hog = "x = bytearray(1024 * 1024 * 1024)"
    background, survivors, procs, memory, slow = in_box(
        sandbox,
        seeded.bundle,
        seeded.ref,
        sh("sleep 300 >/dev/null 2>&1 & echo started"),
        sh("for p in /proc/[0-9]*; do tr '\\0' ' ' < $p/cmdline; echo; done"),
        ["python3", "-c", many],
        ["python3", "-c", hog],
        ["sleep", "300"],
        timeout_s=10.0,
    )
    assert background.ok and "started" in background.output
    assert "sleep 300" not in survivors.output  # it died with its box
    assert procs.ok
    assert int(procs.output.split()[-1]) < box_config.pids_limit
    assert not memory.ok
    assert "out of memory" in memory.output
    assert slow.timed_out
    assert docker_objects(sandbox.runtime.instance) == ([], [])
