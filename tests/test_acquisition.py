"""Synthetic fixed-origin HTTP, private configuration, and ingestion integration."""

from __future__ import annotations

import http.client
import io
import json
import subprocess
import sys
import traceback
from pathlib import Path

import pytest

from oddsjam_ev.io import acquisition
from oddsjam_ev.io.acquisition import acquire_export, download_export
from oddsjam_ev.io.acquisition_config import (
    EXPORT_PATH,
    HOST,
    AcquisitionError,
    DownloadSettings,
    RequestProfile,
)
from oddsjam_ev.io.ingestion import SourceMetadata, load_observations

SECRET = "SYNTHETIC_SECRET_DO_NOT_LOG"
CSV = (
    b"created_at,sportsbook,market_name,bet_name,odds,stake,status,bet_profit,game_id\r\n"
    b"2026-01-01 12:00:00,Book,Moneyline,Away,120,10,pending,-10,g1\r\n"
)


def wire(body=CSV, *, status=200, content_type="text/csv", headers=None, framed=True):
    values = {"Content-Type": content_type}
    if framed:
        values["Content-Length"] = str(len(body))
    values.update(headers or {})
    head = f"HTTP/1.1 {status} Response\r\n" + "".join(f"{k}: {v}\r\n" for k, v in values.items())
    return head.encode() + b"\r\n" + body


class FakeSocket:
    def __init__(self, data):
        self.stream = io.BytesIO(data)

    def makefile(self, *args):
        return self.stream

    def settimeout(self, timeout):
        assert timeout > 0


@pytest.fixture
def transport(monkeypatch):
    responses = []
    requests = []
    connections = []

    class Connection:
        def __init__(self, host, timeout):
            assert host == HOST
            assert 1 <= timeout <= 120
            self.sock = None
            self.closed = False
            connections.append(self)

        def request(self, method, target, body, headers):
            requests.append((method, target, body, headers))

        def getresponse(self):
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            self.sock = FakeSocket(item)
            response = http.client.HTTPResponse(self.sock)
            response.begin()
            return response

        def close(self):
            self.closed = True

    monkeypatch.setattr(acquisition.http.client, "HTTPSConnection", Connection)
    monkeypatch.setattr(acquisition.time, "sleep", lambda delay: None)
    return responses, requests, connections


@pytest.fixture
def profile():
    return RequestProfile("GET", {"Cookie": SECRET}, {"private_parameter": SECRET})


def test_download_preserves_bytes_and_fixed_origin(transport, profile):
    responses, requests, connections = transport
    responses.append(wire())
    data, attempts = download_export(profile, DownloadSettings())
    assert data == CSV
    assert attempts == 1
    assert requests[0][0] == "GET"
    assert requests[0][1].startswith(EXPORT_PATH + "?")
    assert requests[0][3]["Accept-Encoding"] == "identity"
    assert requests[0][3]["Cookie"] == SECRET
    assert all(c.closed for c in connections)
    assert SECRET not in repr(profile)


@pytest.mark.parametrize(
    "status,error",
    [
        (301, "redirect_rejected"),
        (302, "redirect_rejected"),
        (307, "redirect_rejected"),
        (401, "authentication_rejected_status_401"),
        (403, "authentication_rejected_status_403"),
        (400, "http_response_rejected_status_400"),
        (406, "http_response_rejected_status_406"),
        (404, "http_response_rejected_status_404"),
        (204, "http_response_rejected_status_204"),
        (206, "http_response_rejected_status_206"),
    ],
)
def test_status_fails_without_following_or_importing(tmp_path, transport, profile, status, error):
    responses, requests, _ = transport
    responses.append(wire(status=status, headers={"Location": "https://example.invalid/" + SECRET}))
    root = tmp_path / "store"
    with pytest.raises(AcquisitionError, match=error):
        acquire_export(profile, DownloadSettings(), root, SourceMetadata("test"))
    assert len(requests) == 1
    assert not (root / "imports").exists()
    receipt = json.loads(next((root / "acquisitions").glob("*.json")).read_text())
    assert receipt["outcome"] == "failed"
    assert receipt["error_code"] == error
    assert SECRET not in json.dumps(receipt)


@pytest.mark.parametrize(
    "response,error",
    [
        (wire(content_type="text/html"), "unexpected_content_type"),
        (wire(content_type="application/json"), "unexpected_content_type"),
        (wire(body=b"<html>" + SECRET.encode()), "non_csv_response"),
        (wire(body=b'{"error":"' + SECRET.encode() + b'"}'), "non_csv_response"),
        (wire(body=b"a,b\n1,2\n"), "invalid_csv_response"),
        (wire(body=b""), "invalid_csv_response"),
        (wire(body=b"\xff"), "invalid_csv_response"),
        (wire(headers={"Content-Encoding": "gzip"}), "encoded_response_rejected"),
        (wire(framed=False), "unverifiable_response_length"),
        (wire(headers={"Content-Length": "bad"}), "invalid_content_length"),
        (wire(headers={"Content-Length": "-1"}), "invalid_content_length"),
        (wire(headers={"Transfer-Encoding": "chunked"}), "ambiguous_response_framing"),
        (
            wire(framed=False, headers={"Transfer-Encoding": "unsupported"}),
            "unverifiable_response_length",
        ),
    ],
)
def test_reject_invalid_downloads(transport, profile, response, error):
    transport[0].append(response)
    with pytest.raises(AcquisitionError, match=error) as caught:
        download_export(profile, DownloadSettings(attempts=1))
    assert SECRET not in "".join(traceback.format_exception(caught.value))


def test_truncation_and_retry(transport, profile):
    responses, requests, _ = transport
    responses.extend([wire(headers={"Content-Length": str(len(CSV) + 10)}), wire()])
    assert download_export(profile, DownloadSettings())[0] == CSV
    assert len(requests) == 2


def test_chunked_transfer_and_truncated_chunk(transport, profile):
    chunks = f"{len(CSV):x}\r\n".encode() + CSV + b"\r\n0\r\n\r\n"
    transport[0].append(wire(chunks, framed=False, headers={"Transfer-Encoding": "chunked"}))
    assert download_export(profile, DownloadSettings())[0] == CSV
    transport[0].append(wire(chunks[:-7], framed=False, headers={"Transfer-Encoding": "chunked"}))
    with pytest.raises(AcquisitionError, match="incomplete_download"):
        download_export(profile, DownloadSettings(attempts=1))


@pytest.mark.parametrize("framed", [True, False])
def test_size_bounds(transport, profile, framed):
    response = (
        wire()
        if framed
        else wire(
            f"{len(CSV):x}\r\n".encode() + CSV + b"\r\n0\r\n\r\n",
            framed=False,
            headers={"Transfer-Encoding": "chunked"},
        )
    )
    transport[0].append(response)
    with pytest.raises(AcquisitionError, match="download_too_large"):
        download_export(profile, DownloadSettings(max_bytes=10))


@pytest.mark.parametrize("failure", [wire(status=429), wire(status=503), TimeoutError(SECRET)])
def test_transient_failures_have_bounded_retries(transport, profile, failure):
    transport[0].extend([failure] * 3)
    with pytest.raises(AcquisitionError):
        download_export(profile, DownloadSettings(attempts=3))
    assert len(transport[1]) == 3
    assert all(c.closed for c in transport[2])


def test_retry_after_long_delay_fails_closed(transport, profile):
    transport[0].append(wire(status=429, headers={"Retry-After": "300"}))
    with pytest.raises(AcquisitionError, match="retry_later"):
        download_export(profile, DownloadSettings())
    assert len(transport[1]) == 1


def test_post_not_retried(transport):
    transport[0].append(wire(status=503))
    post = RequestProfile("POST", {"Authorization": SECRET}, form={"csrf": SECRET})
    with pytest.raises(AcquisitionError):
        download_export(post, DownloadSettings())
    assert len(transport[1]) == 1
    assert transport[1][0][2] == ("csrf=" + SECRET).encode()


def test_transport_exceptions_are_redacted(transport, profile):
    transport[0].append(OSError("https://private.invalid/?token=" + SECRET))
    with pytest.raises(AcquisitionError, match="transport_failure") as caught:
        download_export(profile, DownloadSettings())
    assert SECRET not in "".join(traceback.format_exception(caught.value))


def test_success_idempotency_browser_comparison_and_receipts(tmp_path, transport, profile):
    root = tmp_path / "store"
    browser = tmp_path / "browser.csv"
    browser.write_bytes(CSV)
    transport[0].extend([wire(), wire()])
    first = acquire_export(
        profile, DownloadSettings(), root, SourceMetadata("test"), compare_csv=browser
    )
    second = acquire_export(profile, DownloadSettings(), root, SourceMetadata("test"))
    assert first["outcome"] == "imported"
    assert second["outcome"] == "already_imported"
    assert first["import_id"] == second["import_id"]
    assert first["observed"]["pending_or_open_rows"] == 1
    assert first["observed"]["identifier_columns_present"] == ["game_id"]
    assert first["observed"]["created_at_with_explicit_offset_rows"] == 0
    assert not first["observed"]["coverage_verified"]
    assert not first["observed"]["settlement_updates_verified"]
    assert first["browser_comparison"]["identical_bytes"]
    observations, manifests = load_observations(root)
    assert len(observations) == len(manifests) == 1
    assert manifests[0]["source"]["exported_at"] is None
    assert manifests[0]["source"]["timestamp_timezone"] == "unknown"
    assert manifests[0]["source"]["scope"] == "unknown"
    assert (root / "objects" / (first["source_sha256"] + ".csv")).read_bytes() == CSV
    for receipt in (root / "acquisitions").glob("*.json"):
        assert SECRET not in receipt.read_text()


def test_bad_browser_comparison_does_not_import(tmp_path, transport, profile):
    transport[0].append(wire())
    with pytest.raises(AcquisitionError, match="browser_comparison_failed"):
        acquire_export(
            profile,
            DownloadSettings(),
            tmp_path / "store",
            SourceMetadata("test"),
            compare_csv=tmp_path / "missing",
        )
    assert not (tmp_path / "store" / "imports").exists()


def test_cli_private_initialization_and_safe_errors(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/acquire_bet_tracker.py"
    path = tmp_path / "private" / "profile.json"
    command = [sys.executable, str(script), "--profile", str(path)]
    result = subprocess.run([*command, "init-profile"], capture_output=True, text=True)
    assert result.returncode == 0
    assert path.stat().st_mode & 0o077 == 0
    assert json.loads(path.read_text())["method"] is None
    assert subprocess.run([*command, "init-profile"], capture_output=True).returncode == 2
    path.write_text('{"credential":"' + SECRET + '" broken')
    result = subprocess.run([*command, "fetch"], capture_output=True, text=True)
    assert result.returncode == 2
    assert SECRET not in result.stderr + result.stdout
    assert "Traceback" not in result.stderr


def test_body_deadline(transport, profile, monkeypatch):
    transport[0].append(wire())
    ticks = iter([0, 31])
    monkeypatch.setattr(acquisition.time, "monotonic", lambda: next(ticks))
    with pytest.raises(AcquisitionError, match="download_timeout"):
        download_export(profile, DownloadSettings(attempts=1))


def test_failure_receipt_attempt_count(tmp_path, transport, profile):
    transport[0].extend([wire(status=503)] * 3)
    root = tmp_path / "store"
    with pytest.raises(AcquisitionError):
        acquire_export(profile, DownloadSettings(), root, SourceMetadata("test"))
    receipt = json.loads(next((root / "acquisitions").glob("*.json")).read_text())
    assert receipt["attempts"] == 3
    assert not (root / "imports").exists()


def test_metadata_conflict_never_reports_success(tmp_path, transport, profile):
    root = tmp_path / "store"
    transport[0].extend([wire(), wire()])
    acquire_export(profile, DownloadSettings(), root, SourceMetadata("test"))
    with pytest.raises(AcquisitionError, match="ingestion_failed"):
        acquire_export(
            profile, DownloadSettings(), root, SourceMetadata("test", timestamp_timezone="UTC")
        )
    assert len(load_observations(root)[1]) == 1
    receipts = [json.loads(p.read_text()) for p in (root / "acquisitions").glob("*.json")]
    assert sorted(r["outcome"] for r in receipts) == ["failed", "imported"]


def test_server_error_values_not_in_cli_output(tmp_path, monkeypatch, capsys):
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts/acquire_bet_tracker.py"
    spec = importlib.util.spec_from_file_location("acquire_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    private = tmp_path / "profile.json"
    private.write_text(json.dumps({"method": "GET", "headers": {"Cookie": SECRET}}))
    private.chmod(0o600)

    def fail(*args, **kwargs):
        raise OSError(SECRET)

    monkeypatch.setattr(module, "acquire_export", fail)
    monkeypatch.setattr(sys, "argv", [str(script), "--profile", str(private), "fetch"])
    with pytest.raises(SystemExit) as caught:
        module.main()
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err
    assert "local_configuration_or_io_failure" in captured.err


@pytest.mark.parametrize("status", [200, 403, 503])
def test_explicit_browser_challenge_is_rejected_without_retry(tmp_path, transport, profile, status):
    transport[0].append(
        wire(
            status=status,
            content_type="text/html",
            body=SECRET.encode(),
            headers={"cf-mitigated": "challenge"},
        )
    )
    root = tmp_path / "store"
    with pytest.raises(AcquisitionError, match=f"browser_challenge_status_{status}"):
        acquire_export(profile, DownloadSettings(), root, SourceMetadata("test"))
    assert len(transport[1]) == 1
    assert not (root / "imports").exists()
    receipt = next((root / "acquisitions").glob("*.json")).read_text()
    assert SECRET not in receipt


def test_html_403_does_not_claim_verified_challenge(transport, profile):
    transport[0].append(
        wire(status=403, content_type="text/html", headers={"Server": "cloudflare"})
    )
    with pytest.raises(AcquisitionError, match="access_rejected_status_403_html_origin_unverified"):
        download_export(profile, DownloadSettings())
