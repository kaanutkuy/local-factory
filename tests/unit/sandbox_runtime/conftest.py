"""A scriptable fake docker CLI for unit-testing the runtime without Docker.

The runtime passes only :func:`docker_env` to the CLI, so the fake cannot be steered through the
environment: its behaviour is baked into the script. Each rule maps an argv prefix (``"start
--attach"``) to what the fake does; the longest matching prefix wins. Every invocation is logged.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

EXITED = '{"Status":"exited","Running":false,"ExitCode":3,"OOMKilled":false,"Error":""}'

DEFAULT_RULES: dict[str, dict[str, Any]] = {
    "version": {"stdout": "29.1.3\n"},
    "image inspect": {"stdout": "sha256:" + "a" * 64 + "\n"},
    "create": {"stdout": "0123abcd\n"},
    "start --attach": {"stdout": "out\n", "stderr": "err\n"},
    "start": {},
    "inspect": {"stdout": EXITED + "\n"},
    "stop": {},
    "rm": {},
    "ps": {},
    "volume create": {},
    "volume inspect": {"stdout": '{"local-factory":"1"}\n'},
    "volume rm": {},
    "volume ls": {},
}

SCRIPT = """#!{python} -I
import json, os, sys
args = sys.argv[1:]
with open({log!r}, "a") as fh:
    fh.write(json.dumps(args) + "\\n")
rules = json.loads({rules!r})
rule = {{}}
for key in sorted(rules, key=lambda k: -len(k.split())):
    if args[: len(key.split())] == key.split():
        rule = rules[key]
        break
if "stdin_to" in rule:
    with open(rule["stdin_to"], "wb") as fh:
        fh.write(sys.stdin.buffer.read())
out = rule.get("stdout", "")
if "{{name}}" in out:
    out = out.replace("{{name}}", args[-1])
sys.stdout.write(out * rule.get("repeat", 1))
sys.stderr.write(rule.get("stderr", ""))
sys.stdout.flush()
sys.stderr.flush()
if "exec" in rule:
    os.execvp(rule["exec"][0], rule["exec"])
sys.exit(rule.get("exit", 0))
"""


@dataclass
class FakeDocker:
    path: Path
    log: Path

    def calls(self) -> list[list[str]]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def commands(self) -> list[str]:
        """The subcommand of every call ("image inspect", "create", "start", ...)."""
        two_word = {"image", "volume"}
        return [" ".join(c[:2]) if c[0] in two_word else c[0] for c in self.calls()]


MakeFake = Callable[..., FakeDocker]


@pytest.fixture
def fake_docker(tmp_path: Path) -> MakeFake:
    counter = iter(range(1_000))

    def make(rules: Mapping[str, Mapping[str, Any] | None] | None = None) -> FakeDocker:
        n = next(counter)
        merged = dict(DEFAULT_RULES)
        for key, rule in (rules or {}).items():
            if rule is None:
                merged.pop(key, None)
            else:
                merged[key] = dict(rule)
        path = tmp_path / f"docker-{n}"
        log = tmp_path / f"docker-{n}.log"
        path.write_text(
            SCRIPT.format(python=sys.executable, log=str(log), rules=json.dumps(merged))
        )
        path.chmod(0o755)
        return FakeDocker(path=path, log=log)

    return make
