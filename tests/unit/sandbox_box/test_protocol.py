from __future__ import annotations

import pytest

from local_factory.sandbox.protocol import (
    ERROR_PREFIX,
    MAX_DETAIL,
    bounded,
    error_line,
    error_reason,
)


def test_bounded_makes_one_printable_line() -> None:
    assert bounded("  two\nlines\r\n\tand \x1b[31mescapes\x00  ") == "two lines and [31mescapes"
    assert bounded("") == ""


def test_bounded_truncates_with_an_ellipsis() -> None:
    assert bounded("abcdefghij", 10) == "abcdefghij"
    assert bounded("abcdefghijk", 10) == "abcdefg..."
    assert len(bounded("x" * 5000)) == MAX_DETAIL
    with pytest.raises(ValueError, match="limit"):
        bounded("x", 2)


def test_error_line_is_bounded_and_never_empty() -> None:
    assert error_line("bad\nthing") == "error: bad thing"
    assert error_line("  ") == "error: unknown error"
    line = error_line("y" * 5000)
    assert len(line) == MAX_DETAIL
    assert line.startswith(ERROR_PREFIX) and line.endswith("...")


def test_error_reason_takes_the_last_error_line() -> None:
    stderr = "noise\nerror: first\nmore noise\nerror: second\ntrailing docker text\n"
    assert error_reason(stderr) == "second"
    assert error_reason("no reason here\n") is None
    assert error_reason("") is None
    assert error_reason("error: \n") is None
    assert error_reason("error: " + "z" * 5000) == "z" * (MAX_DETAIL - 3) + "..."
