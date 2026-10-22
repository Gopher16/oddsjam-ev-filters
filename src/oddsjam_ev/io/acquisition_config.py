"""
acquisition_config.py
===============================================================================
Purpose
-------
Separate non-secret acquisition limits from explicitly supplied private requests.

What this module provides
-------------------------
Validated download settings and a private, non-representable request profile.

Design Notes
------------
No browser stores are inspected. No authentication scheme or request method is
inferred from old CSVs. Only the fixed Bet Tracker export endpoint is supported.
===============================================================================
"""

from __future__ import annotations

import json
import math
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

HOST = "oddsjam.com"
EXPORT_PATH = "/api/backend/bets/export-csv"


class AcquisitionError(ValueError):
    """A safe local error category; never contains remote messages or secrets."""


@dataclass(frozen=True)
class DownloadSettings:
    """Bounded HTTP download settings; no account information.

    Attributes
    ----------
    timeout_seconds : float
        Socket timeout and response-body deadline, between 1 and 120 seconds.
    max_bytes : int
        Maximum uncompressed payload size, at most 512 MiB.
    attempts : int
        Maximum GET attempts, between one and four. POST is never retried.
    """

    timeout_seconds: float = 30
    max_bytes: int = 128 * 1024 * 1024
    attempts: int = 3

    def __post_init__(self) -> None:
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or not 1 <= self.timeout_seconds <= 120
        ):
            raise AcquisitionError("invalid_timeout")
        if type(self.max_bytes) is not int or not 1 <= self.max_bytes <= 512 * 1024 * 1024:
            raise AcquisitionError("invalid_size_limit")
        if type(self.attempts) is not int or not 1 <= self.attempts <= 4:
            raise AcquisitionError("invalid_attempt_limit")


@dataclass(frozen=True, repr=False)
class RequestProfile:
    """User-observed request details, held privately in memory.

    Attributes
    ----------
    method : str
        Explicit observed GET or POST method. No default is assumed.
    headers : dict
        Cookie and/or Authorization plus optional CSRF headers.
    query : dict
        Observed query parameters; values are private and never logged.
    form : dict or None
        Optional observed POST form fields. JSON requests are not supported.
    """

    method: str
    headers: dict[str, str] = field(repr=False)
    query: dict[str, str] = field(default_factory=dict, repr=False)
    form: dict[str, str] | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        return "RequestProfile(<redacted>)"

    def __post_init__(self) -> None:
        if self.method not in {"GET", "POST"}:
            raise AcquisitionError("request_method_not_verified")
        allowed = {"cookie", "authorization", "x-csrftoken", "x-csrf-token", "x-xsrf-token"}
        if not isinstance(self.headers, dict) or not self.headers:
            raise AcquisitionError("authentication_not_configured")
        normalized = set()
        for key, value in self.headers.items():
            if not isinstance(key, str) or key.lower() not in allowed or key.lower() in normalized:
                raise AcquisitionError("unsupported_private_header")
            normalized.add(key.lower())
            if (
                not isinstance(value, str)
                or not value.strip()
                or any(ord(c) < 32 or ord(c) > 255 for c in value)
            ):
                raise AcquisitionError("invalid_private_header")
        if not normalized & {"cookie", "authorization"}:
            raise AcquisitionError("authentication_not_configured")
        if not isinstance(self.query, dict):
            raise AcquisitionError("invalid_private_parameters")
        for mapping in [self.query, self.form]:
            if mapping is not None and (
                not isinstance(mapping, dict)
                or any(not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items())
            ):
                raise AcquisitionError("invalid_private_parameters")
        if self.method == "GET" and self.form is not None:
            raise AcquisitionError("get_body_not_supported")


def load_request_profile(path: Path) -> RequestProfile:
    """Load only a user-supplied, owner-private regular JSON file.

    Parameters
    ----------
    path : pathlib.Path
        Local profile, normally .local/oddsjam/request.json, mode 0600 or 0400.

    Returns
    -------
    RequestProfile
        Validated secrets. File contents, paths, and parse exceptions are redacted.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
                raise AcquisitionError("credential_file_must_be_owner_private")
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise AcquisitionError("credential_file_too_large")
        payload = json.loads(raw)
        if not isinstance(payload, dict) or set(payload) - {"method", "headers", "query", "form"}:
            raise AcquisitionError("invalid_request_profile")
        return RequestProfile(**payload)
    except AcquisitionError:
        raise
    except (OSError, ValueError, TypeError):
        raise AcquisitionError("request_profile_unavailable_or_invalid") from None
