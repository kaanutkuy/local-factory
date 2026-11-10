from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from local_factory.ledger import Ledger

from .helpers import FakeClock


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "ledger.db"


@pytest.fixture
def ledger(db_path: Path, clock: FakeClock) -> Iterator[Ledger]:
    with Ledger(db_path, clock=clock) as led:
        yield led
