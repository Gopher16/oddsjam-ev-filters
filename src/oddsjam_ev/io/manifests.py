"""
manifests.py
===============================================================================
Purpose
-------
Provide immutable local publication and a fail-fast single-writer guard.

What this module provides
-------------------------
Atomic JSON writes, content verification, and ingestion store locking.

Design Notes
------------
A stale lock requires explicit operator review; it is never removed automatically.
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    """Write JSON exclusively; refuse to overwrite existing history.

    Parameters
    ----------
    path : pathlib.Path
        New destination.
    value : object
        JSON-compatible payload.

    Returns
    -------
    None
    """
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def verify_bytes(path: Path, expected: str) -> bytes:
    """Read an archive and reject corruption.

    Parameters
    ----------
    path : pathlib.Path
        Archived source.
    expected : str
        Expected SHA-256 digest.

    Returns
    -------
    bytes
        Verified contents.
    """
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Archive integrity failure: {path.name}")
    return data


@contextmanager
def writer_lock(root: Path) -> Iterator[None]:
    """Hold an exclusive store lock during mutations.

    Parameters
    ----------
    root : pathlib.Path
        Local ingestion store.

    Yields
    ------
    None
        Lock ownership; all writers must use this guard.
    """
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".writer.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(
            "Ingestion store is locked; review running writers or a stale lock"
        ) from exc
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink()
