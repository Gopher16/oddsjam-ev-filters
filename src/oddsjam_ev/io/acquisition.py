"""
acquisition.py
===============================================================================
Purpose
-------
Download an explicitly configured account export and reuse immutable ingestion.

What this module provides
-------------------------
Fixed-origin HTTP acquisition, bounded downloads/retries, and safe run receipts.

Design Notes
------------
Redirects and proxy environment settings are not used. HTTP bodies, request URLs,
headers, credentials, and remote exception messages are never logged or persisted.
A successful transport does not prove full history, freshness, or timezone.
===============================================================================
"""

from __future__ import annotations

import hashlib
import http.client
import tempfile
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

from oddsjam_ev.io.acquisition_config import (
    EXPORT_PATH,
    HOST,
    AcquisitionError,
    DownloadSettings,
    RequestProfile,
)
from oddsjam_ev.io.ingestion import SourceMetadata, import_csv
from oddsjam_ev.io.manifests import write_json
from oddsjam_ev.io.source_csv import read_source

RETRYABLE = {408, 429, 500, 502, 503, 504}
CSV_TYPES = {
    "text/csv",
    "application/csv",
    "application/octet-stream",
    "application/vnd.ms-excel",
    "text/plain",
}


class _TransientError(AcquisitionError):
    def __init__(self, code: str, delay: float = 1) -> None:
        super().__init__(code)
        self.delay = delay


def _download_once(profile: RequestProfile, settings: DownloadSettings) -> bytes:
    connection = None
    try:
        connection = http.client.HTTPSConnection(HOST, timeout=settings.timeout_seconds)
        query = urlencode(profile.query)
        target = EXPORT_PATH + ("?" + query if query else "")
        headers = {
            "Accept": "text/csv, application/octet-stream;q=0.9",
            "Accept-Encoding": "identity",
            "User-Agent": "oddsjam-ev-filters-export/1",
            **profile.headers,
        }
        body = urlencode(profile.form).encode() if profile.form is not None else None
        if body is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        connection.request(profile.method, target, body=body, headers=headers)
        response = connection.getresponse()
        if response.getheader("cf-mitigated", "").strip().lower() == "challenge":
            raise AcquisitionError(f"browser_challenge_status_{int(response.status)}")
        if (
            response.status == 403
            and response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
            == "text/html"
        ):
            raise AcquisitionError("access_rejected_status_403_html_origin_unverified")
        if response.status in RETRYABLE:
            retry_after = response.getheader("Retry-After", "")
            if retry_after and (
                not retry_after.isascii() or not retry_after.isdigit() or len(retry_after) > 3
            ):
                raise AcquisitionError("retry_later")
            delay = float(retry_after) if retry_after else 1
            if delay > 10:
                raise AcquisitionError("retry_later")
            raise _TransientError("http_transient_failure", delay=max(1, delay))
        if response.status in {401, 403}:
            raise AcquisitionError(f"authentication_rejected_status_{int(response.status)}")
        if 300 <= response.status < 400:
            raise AcquisitionError("redirect_rejected")
        if response.status != 200:
            raise AcquisitionError(f"http_response_rejected_status_{int(response.status)}")
        content_type = response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type not in CSV_TYPES:
            raise AcquisitionError("unexpected_content_type")
        if response.getheader("Content-Encoding", "identity").lower() != "identity":
            raise AcquisitionError("encoded_response_rejected")
        raw_headers = response.getheaders()
        lengths = [v for k, v in raw_headers if k.lower() == "content-length"]
        transfers = [v for k, v in raw_headers if k.lower() == "transfer-encoding"]
        if len(lengths) > 1 or len(transfers) > 1 or (lengths and transfers):
            raise AcquisitionError("ambiguous_response_framing")
        length = None
        if lengths:
            if not lengths[0].isascii() or not lengths[0].isdigit() or len(lengths[0]) > 12:
                raise AcquisitionError("invalid_content_length")
            length = int(lengths[0])
            if length > settings.max_bytes:
                raise AcquisitionError("download_too_large")
        elif not transfers or transfers[0].lower() != "chunked":
            raise AcquisitionError("unverifiable_response_length")
        deadline = time.monotonic() + settings.timeout_seconds
        chunks = []
        size = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _TransientError("download_timeout")
            if connection.sock is not None:
                connection.sock.settimeout(remaining)
            chunk = response.read1(min(65536, settings.max_bytes - size + 1))
            if not chunk:
                break
            size += len(chunk)
            if size > settings.max_bytes:
                raise AcquisitionError("download_too_large")
            chunks.append(chunk)
        if length is not None and size != length:
            raise _TransientError("incomplete_download")
        data = b"".join(chunks)
        prefix = data.lstrip(b"\xef\xbb\xbf \r\n\t").lower()
        if prefix.startswith((b"<", b"{", b"[")):
            raise AcquisitionError("non_csv_response")
        try:
            read_source(data)
        except (ValueError, TypeError, OverflowError):
            raise AcquisitionError("invalid_csv_response") from None
        return data
    except AcquisitionError:
        raise
    except TimeoutError:
        raise _TransientError("download_timeout") from None
    except http.client.IncompleteRead:
        raise _TransientError("incomplete_download") from None
    except (OSError, http.client.HTTPException, ValueError, UnicodeError):
        raise AcquisitionError("transport_failure") from None
    finally:
        if connection is not None:
            connection.close()


def download_export(profile: RequestProfile, settings: DownloadSettings) -> tuple[bytes, int]:
    """Download and validate bytes without following redirects or logging secrets.

    Parameters
    ----------
    profile : RequestProfile
        Explicitly supplied private method, headers, and parameters.
    settings : DownloadSettings
        Bounded size, socket timeout, read deadline, and attempt limits.

    Returns
    -------
    tuple[bytes, int]
        Validated original CSV bytes and attempts used.

    Notes
    -----
    Only GET retries explicit transient status, timeout, or incomplete transfer.
    TLS, DNS, and other transport failures fail closed. POST is tried once.
    """
    attempts = settings.attempts if profile.method == "GET" else 1
    for attempt in range(1, attempts + 1):
        try:
            return _download_once(profile, settings), attempt
        except _TransientError as exc:
            if attempt == attempts:
                error = AcquisitionError(str(exc))
                error.attempts_used = attempt
                raise error from None
            time.sleep(min(10, exc.delay * attempt))
        except AcquisitionError as exc:
            exc.attempts_used = attempt
            raise
    raise AcquisitionError("download_failed")  # pragma: no cover


def source_evidence(data: bytes) -> dict:
    """Describe observed schema without claiming coverage or exposing row values.

    Parameters
    ----------
    data : bytes
        Already validated export bytes.

    Returns
    -------
    dict
        Safe counts and availability indicators; no account data or source values.
    """
    frame, diagnostics = read_source(data)
    statuses = frame["status"].str.strip().str.lower()
    known = {
        "pending",
        "open",
        "unsettled",
        "won",
        "lost",
        "refunded",
        "cancelled",
        "half_won",
        "half_lost",
    }
    aware = frame["created_at"].str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
    return {
        "rows": len(frame),
        "column_count": len(frame.columns),
        "duplicate_rows_beyond_first": diagnostics["duplicate_rows_beyond_first"],
        "pending_or_open_rows": int(statuses.isin({"pending", "open", "unsettled"}).sum()),
        "unrecognized_status_rows": int((~statuses.isin(known)).sum()),
        "identifier_columns_present": [
            c for c in ("id", "bet_id", "wager_id", "uuid", "game_id", "parlay_id") if c in frame
        ],
        "update_timestamp_columns_present": [c for c in ("updated_at", "settled_at") if c in frame],
        "created_at_with_explicit_offset_rows": int(aware.sum()),
        "coverage_verified": False,
        "timezone_verified": False,
        "settlement_updates_verified": False,
    }


def acquire_export(
    profile: RequestProfile,
    settings: DownloadSettings,
    root: Path,
    metadata: SourceMetadata,
    *,
    compare_csv: Path | None = None,
) -> dict:
    """Download original bytes, import them, and publish a safe acquisition receipt.

    Parameters
    ----------
    profile : RequestProfile
        Private user-observed request.
    settings : DownloadSettings
        Transport limits.
    root : pathlib.Path
        Existing private ingestion store.
    metadata : SourceMetadata
        Declared scope/export time/timezone; never inferred from HTTP success.
    compare_csv : pathlib.Path or None
        Optional local browser download for a non-secret comparison summary.

    Returns
    -------
    dict
        Safe receipt with import linkage, observed counts, and explicit unknowns.

    Notes
    -----
    Receipt times describe acquisition, not server export or settlement time.
    Identical bytes reuse the existing import without changing source metadata.
    """
    receipt = {
        "schema_version": 1,
        "acquisition_id": uuid.uuid4().hex,
        "started_at": datetime.now(UTC).isoformat(),
        "method": profile.method,
        "settings": asdict(settings),
        "outcome": "failed",
    }
    failure = None
    try:
        data, attempts = download_export(profile, settings)
        receipt["attempts"] = attempts
        receipt["downloaded_at"] = datetime.now(UTC).isoformat()
        receipt["source_sha256"] = hashlib.sha256(data).hexdigest()
        receipt["byte_size"] = len(data)
        receipt["observed"] = source_evidence(data)
        if compare_csv is not None:
            try:
                browser_data = compare_csv.read_bytes()
                browser, _ = read_source(browser_data)
                downloaded, _ = read_source(data)
                receipt["browser_comparison"] = {
                    "identical_bytes": data == browser_data,
                    "same_column_names_and_order": list(browser.columns)
                    == list(downloaded.columns),
                    "browser_rows": len(browser),
                    "downloaded_rows": len(downloaded),
                }
            except (OSError, ValueError, TypeError):
                raise AcquisitionError("browser_comparison_failed") from None
        with tempfile.TemporaryDirectory(prefix="oddsjam-export-") as temporary:
            path = Path(temporary) / "bet-tracker-export.csv"
            path.write_bytes(data)
            try:
                imported = import_csv(path, root, metadata)
            except (OSError, ValueError, TypeError, KeyError):
                raise AcquisitionError("ingestion_failed_review_private_store") from None
        receipt["import_id"] = imported["import_id"]
        receipt["outcome"] = "already_imported" if imported["repeated_import"] else "imported"
    except AcquisitionError as exc:
        receipt["error_code"] = str(exc)
        receipt["attempts"] = getattr(exc, "attempts_used", receipt.get("attempts", 0))
        failure = str(exc)
    finally:
        receipt["finished_at"] = datetime.now(UTC).isoformat()
        try:
            directory = root / "acquisitions"
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".stage-", dir=directory) as staging:
                path = Path(staging) / "receipt.json"
                write_json(path, receipt)
                path.rename(directory / f"{receipt['acquisition_id']}.json")
        except OSError:
            raise AcquisitionError("acquisition_receipt_write_failed") from None
    if failure:
        raise AcquisitionError(failure) from None
    return receipt
