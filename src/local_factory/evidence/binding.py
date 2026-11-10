"""Evidence: build, hash, sign and bind the record a release decision rests on (PLAN.md §6.2).

Enforcement does not depend on the signature (GitHub binds commit statuses to SHAs); the HMAC
makes tampering with stored evidence detectable. The key is factory-held, never given to agents.
"""

from __future__ import annotations

import errno
import hashlib
import hmac
import os
import re
import secrets
import stat
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path

from local_factory.hashing import canonical_json, content_hash, utc_now
from local_factory.models import Evidence, Verdict

KEY_BYTES = 32
KEY_MODE = 0o600
_SIGNATURE = re.compile(r"[0-9a-fA-F]{64}")
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_CREATE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


class EvidenceKeyError(Exception):
    """The evidence key file exists but is unsafe or unusable. It is never silently replaced."""


def build_evidence(
    *,
    ticket_id: str,
    verdict: Verdict,
    tree_hash: str,
    base_sha: str,
    contract_hash: str,
    oracle_version: str,
    factory_version: str,
    image_digests: Mapping[str, str] | None = None,
    models: Mapping[str, str] | None = None,
    clock: Callable[[], datetime] = utc_now,
) -> Evidence:
    """Bind a verdict to the exact candidate, contract and oracle it was produced against."""
    if ticket_id != verdict.ticket_id:
        raise ValueError(f"verdict belongs to ticket {verdict.ticket_id}, not ticket {ticket_id}")
    if verdict.passed and verdict.failing:
        raise ValueError(
            f"inconsistent verdict for ticket {ticket_id}: passed but failing {verdict.failing}"
        )
    return Evidence(
        ticket_id=ticket_id,
        candidate_sha=verdict.sha,
        tree_hash=tree_hash,
        base_sha=base_sha,
        contract_hash=contract_hash,
        oracle_version=oracle_version,
        factory_version=factory_version,
        image_digests=dict(image_digests or {}),
        models=dict(models or {}),
        gates=verdict.gates,
        criteria=verdict.criteria,
        waivers=verdict.waivers,
        passed=verdict.passed,
        created_at=clock(),
    )


def evidence_hash(evidence: Evidence) -> str:
    """`sha256:<hex>` over the canonical JSON of the evidence."""
    return content_hash(evidence)


# --------------------------------------------------------------------------- key file


def load_or_create_key(path: Path) -> bytes:
    """Return the evidence key at `path`, creating a fresh random one if none exists.

    A new key appears at `path` atomically: it is written to a private temporary file, synced,
    then hard-linked into place (which never overwrites a concurrently created key). An existing
    key must be a regular file (not a symlink), exactly `KEY_BYTES` long, with no group/other
    permission bits; anything else raises `EvidenceKeyError` rather than being replaced, since a
    replaced key would orphan every signature made with the old one. Failing to create a key (an
    unwritable directory, a full disk) propagates as `OSError`.
    """
    try:
        return _read_key(path)
    except FileNotFoundError:
        return _create_key(path)


def _read_key(path: Path) -> bytes:
    """Read and check an existing key. Raises FileNotFoundError only if nothing is there."""
    try:
        fd = os.open(path, _READ_FLAGS)
    except FileNotFoundError:
        raise
    except OSError as exc:
        reason = "is a symbolic link" if exc.errno == errno.ELOOP else (exc.strerror or str(exc))
        raise EvidenceKeyError(f"cannot use evidence key {path}: {reason}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise EvidenceKeyError(f"evidence key {path} is not a regular file")
        if info.st_mode & 0o077:
            raise EvidenceKeyError(
                f"evidence key {path} has mode {stat.S_IMODE(info.st_mode):04o}; it must not be "
                f"accessible to group or others (chmod {KEY_MODE:o} it)"
            )
        with os.fdopen(fd, "rb", closefd=False) as handle:
            key = handle.read(KEY_BYTES + 1)
    finally:
        os.close(fd)
    if len(key) != KEY_BYTES:
        raise EvidenceKeyError(
            f"evidence key {path} is {'truncated' if len(key) < KEY_BYTES else 'too long'}: "
            f"expected exactly {KEY_BYTES} bytes"
        )
    return key


def _create_key(path: Path) -> bytes:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    key = secrets.token_bytes(KEY_BYTES)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    fd = os.open(temporary, _CREATE_FLAGS, KEY_MODE)
    try:
        with os.fdopen(fd, "wb") as handle:
            os.fchmod(handle.fileno(), KEY_MODE)  # exact mode, whatever the umask
            handle.write(key)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            path.hardlink_to(temporary)
        except FileExistsError:
            return _read_key(path)  # another process created it first; use theirs
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)
    return key


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# --------------------------------------------------------------------------- signatures


def sign(evidence: Evidence, key: bytes) -> str:
    """Hex HMAC-SHA256 over the canonical JSON of the evidence."""
    return _mac(evidence, key).hex()


def verify_signature(evidence: Evidence, signature: str, key: bytes) -> bool:
    """Constant-time check of `signature`; anything that is not 64 hex digits is invalid."""
    expected = _mac(evidence, key)
    if not _SIGNATURE.fullmatch(signature):
        return False
    return hmac.compare_digest(bytes.fromhex(signature), expected)


def _mac(evidence: Evidence, key: bytes) -> bytes:
    if not key:
        raise ValueError("evidence signing key must not be empty")
    return hmac.digest(key, canonical_json(evidence), hashlib.sha256)


# --------------------------------------------------------------------------- binding


def check_binding(
    evidence: Evidence, *, sha: str, tree_hash: str, contract_hash: str, oracle_version: str
) -> list[str]:
    """Every way `evidence` fails to authorize this exact candidate; `[]` means it is bound."""
    anchors = (
        ("candidate sha", evidence.candidate_sha, sha),
        ("tree hash", evidence.tree_hash, tree_hash),
        ("contract hash", evidence.contract_hash, contract_hash),
        ("oracle version", evidence.oracle_version, oracle_version),
    )
    problems = [
        f"{label} mismatch: evidence has {recorded}, candidate has {actual}"
        for label, recorded, actual in anchors
        if recorded != actual
    ]
    if not evidence.passed:
        problems.append("evidence records a verification that did not pass (passed=false)")
    return problems
