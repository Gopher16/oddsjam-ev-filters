"""Manual Bet Tracker ingestion CLI; reusable processing lives in oddsjam_ev.io."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from oddsjam_ev.io.ingestion import SourceMetadata, import_csv, load_observations
from oddsjam_ev.io.snapshots import materialize_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data_sources.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("import")
    ingest.add_argument("csv", type=Path)
    ingest.add_argument("--exported-at", help="Confirmed ISO export time including offset")
    ingest.add_argument("--timestamp-timezone", help="Confirmed naive timestamp timezone")
    ingest.add_argument("--scope")
    for command in ["inspect", "rebuild", "snapshot"]:
        sub = commands.add_parser(command)
        sub.add_argument("--as-of", help="System knowledge cutoff, including offset")
        if command != "inspect":
            sub.add_argument("--decisions", type=Path, help="Reviewed JSON identity resolutions")
    args = parser.parse_args()
    try:
        config = yaml.safe_load(args.config.read_text())
        if set(config) != {"store", "source"}:
            raise ValueError("Configuration must contain only store and source")
        source = dict(config["source"])
        root = Path(config["store"])
        if args.command == "import":
            for field in ["exported_at", "timestamp_timezone", "scope"]:
                value = getattr(args, field)
                if value is not None:
                    source[field] = value
            result = import_csv(args.csv, root, SourceMetadata(**source))
            print(json.dumps(result, indent=2))
        elif args.command == "inspect":
            observations, imports = load_observations(root, as_of=args.as_of)
            print(
                json.dumps(
                    {
                        "imports": len(imports),
                        "observations": len(observations),
                        "sources": imports,
                    },
                    indent=2,
                )
            )
        else:
            decisions = json.loads(args.decisions.read_text()) if args.decisions else []
            destination = materialize_snapshot(root, as_of=args.as_of, decisions=decisions)
            print(destination)
            print((destination / "manifest.json").read_text())
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"Ingestion failed: {exc}\n")


if __name__ == "__main__":
    main()
