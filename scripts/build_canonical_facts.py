"""Build canonical wager facts from a pinned manual-ingestion snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from oddsjam_ev.analysis.canonical_config import CanonicalConfig
from oddsjam_ev.io.canonical import build_canonical_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/canonical_facts.yaml"))
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/filter_history/oddsjam_filter_configurations_2026-10-22.yaml"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/canonical"))
    args = parser.parse_args()
    try:
        cfg = CanonicalConfig(**yaml.safe_load(args.config.read_text()))
        registry = yaml.safe_load(args.registry.read_text())
        path = build_canonical_bundle(args.snapshot, args.output_root, cfg, registry)
        print(f"Canonical bundle: {path}")
        print(json.dumps(json.loads((path / "manifest.json").read_text())["summary"], indent=2))
    except (OSError, ValueError, TypeError, KeyError, AssertionError) as exc:
        parser.exit(2, f"Canonical build failed: {exc}\n")


if __name__ == "__main__":
    main()
