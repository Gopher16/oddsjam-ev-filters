"""Private credential configuration contracts; no network or browser access."""

import json
import traceback

import pytest

from oddsjam_ev.io.acquisition_config import (
    AcquisitionError,
    DownloadSettings,
    load_request_profile,
)

SECRET = "SYNTHETIC_SECRET_DO_NOT_LOG"


def write_profile(tmp_path, payload):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload))
    path.chmod(0o600)
    return path


def test_private_profile_file(tmp_path):
    path = write_profile(tmp_path, {"method": "GET", "headers": {"Cookie": SECRET}})
    loaded = load_request_profile(path)
    assert loaded.headers["Cookie"] == SECRET
    assert SECRET not in repr(loaded)
    path.chmod(0o644)
    with pytest.raises(AcquisitionError, match="owner_private"):
        load_request_profile(path)
    path.chmod(0o600)
    symlink = tmp_path / "link"
    symlink.symlink_to(path)
    with pytest.raises(AcquisitionError):
        load_request_profile(symlink)


@pytest.mark.parametrize(
    "payload",
    [
        {"method": None, "headers": {}},
        {"method": "GET", "headers": {"Cookie": SECRET + "\r\nInjected: value"}},
        {"method": "GET", "headers": {"Host": "evil.invalid"}},
        {"method": "GET", "headers": {"X-CSRFToken": SECRET}},
        {"method": "GET", "headers": {"Cookie": SECRET, "cookie": SECRET}},
        {"method": "GET", "headers": {"Cookie": SECRET}, "url": SECRET},
        {"method": "PUT", "headers": {"Cookie": SECRET}},
        {"method": "GET", "headers": {"Cookie": SECRET}, "form": {}},
        {"method": "GET", "headers": {"Cookie": SECRET}, "query": {"x": 1}},
    ],
)
def test_invalid_profiles_never_echo_secrets(tmp_path, payload):
    with pytest.raises(AcquisitionError) as caught:
        load_request_profile(write_profile(tmp_path, payload))
    assert SECRET not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize(
    "settings",
    [
        {"timeout_seconds": 0},
        {"timeout_seconds": float("inf")},
        {"timeout_seconds": True},
        {"max_bytes": 0},
        {"max_bytes": 2**30},
        {"attempts": 5},
        {"attempts": 1.5},
    ],
)
def test_invalid_settings(settings):
    with pytest.raises(AcquisitionError):
        DownloadSettings(**settings)
