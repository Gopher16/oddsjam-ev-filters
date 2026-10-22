"""Acquire a privately configured Bet Tracker export; never print HTTP details."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from oddsjam_ev.io.acquisition import acquire_export
from oddsjam_ev.io.acquisition_config import (
    AcquisitionError,
    DownloadSettings,
    load_request_profile,
)
from oddsjam_ev.io.ingestion import SourceMetadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(".local/oddsjam/request.json"),
        help="Owner-private request file; never pass credentials as arguments",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/export_acquisition.yaml"))
    parser.add_argument("--source-config", type=Path, default=Path("configs/data_sources.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-profile", help="Create an unconfigured private template; no network")
    commands.add_parser("check-profile", help="Validate configuration locally; no network")
    fetch = commands.add_parser("fetch", help="Download, validate, and import CSV")
    fetch.add_argument(
        "--compare-csv", type=Path, help="Optional local browser export for comparison"
    )
    args = parser.parse_args()
    error = None
    try:
        if args.command == "init-profile":
            args.profile.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(args.profile, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                json.dump({"method": None, "headers": {}, "query": {}}, stream, indent=2)
                stream.write("\n")
            print("Private template created. Configure it locally using the acquisition guide.")
            return
        config = yaml.safe_load(args.config.read_text())
        if (
            not isinstance(config, dict)
            or set(config) != {"download"}
            or not isinstance(config["download"], dict)
        ):
            raise AcquisitionError("invalid_acquisition_config")
        settings = DownloadSettings(**config["download"])
        source_config = yaml.safe_load(args.source_config.read_text())
        if not isinstance(source_config, dict) or set(source_config) != {"store", "source"}:
            raise AcquisitionError("invalid_source_config")
        metadata = SourceMetadata(**source_config["source"])
        if not isinstance(source_config["store"], str) or not source_config["store"].strip():
            raise AcquisitionError("invalid_store")
        profile = load_request_profile(args.profile)
        if args.command == "check-profile":
            print(
                "Local configuration valid. Authentication and export coverage remain unverified."
            )
            return
        result = acquire_export(
            profile, settings, Path(source_config["store"]), metadata, compare_csv=args.compare_csv
        )
        print(json.dumps(result, indent=2))
    except AcquisitionError as exc:
        error = str(exc)
    except Exception:
        # Do not display parser errors, paths, HTTP messages, or credential values.
        error = "local_configuration_or_io_failure"
    if error:
        parser.exit(2, f"Acquisition failed: {error}. See docs/EXPORT_ACQUISITION.md.\n")


if __name__ == "__main__":
    main()
