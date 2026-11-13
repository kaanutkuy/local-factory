"""Ticket-id validation shared by the oracle's path-building helpers."""

from __future__ import annotations

import re

from local_factory.models import TICKET_ID_PATTERN

_TICKET_ID = re.compile(TICKET_ID_PATTERN)


def check_ticket_id(ticket_id: str) -> str:
    """``ticket_id`` itself, or ``ValueError`` unless it is exactly ``T-<4+ digits>``.

    Every path and ref the oracle builds from a ticket id goes through here, so an id can never
    carry a separator, a ``..`` or a newline into a file name or a ref name.
    """
    if _TICKET_ID.fullmatch(ticket_id) is None:
        raise ValueError(f"not a ticket id (expected T-0001): {ticket_id!r}")
    return ticket_id
