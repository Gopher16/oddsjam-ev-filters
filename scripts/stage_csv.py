"""Private CSV staging, review and explicitly confirmed immutable import."""

import argparse
import json
from pathlib import Path

from oddsjam_ev.io.ingestion import SourceMetadata
from oddsjam_ev.io.staging import confirm_import, preview_csv, stage_csv


def main() -> None:
    """Run the local staging interface; no build or activation occurs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--ingestion-root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage")
    stage.add_argument("file", type=Path)
    stage.add_argument("--namespace", required=True)
    stage.add_argument("--scope", choices=["full", "partial", "unknown"], default="unknown")
    stage.add_argument("--exported-at")
    stage.add_argument("--timestamp-timezone", default="unknown")
    preview = commands.add_parser("preview")
    preview.add_argument("stage_id")
    confirm = commands.add_parser("confirm")
    confirm.add_argument("stage_id")
    confirm.add_argument("--preview-sha256", required=True)
    confirm.add_argument("--confirm-import", action="store_true")
    args = parser.parse_args()
    if args.staging_root.resolve() == args.ingestion_root.resolve():
        parser.error("Use separate staging and ingestion directories")
    if args.command == "stage":
        metadata = SourceMetadata(
            args.namespace, args.exported_at, args.timestamp_timezone, args.scope
        )
        with args.file.open("rb") as stream:
            result = stage_csv(
                stream, args.staging_root, filename=args.file.name, metadata=metadata
            )
    elif args.command == "preview":
        result = preview_csv(args.staging_root, args.stage_id, args.ingestion_root)
    else:
        result = confirm_import(
            args.staging_root,
            args.stage_id,
            args.ingestion_root,
            preview_sha256=args.preview_sha256,
            confirmed=args.confirm_import,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
