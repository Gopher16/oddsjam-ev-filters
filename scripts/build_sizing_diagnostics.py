"""Publish explicit paired weighted/equal-stake and sizing-regime diagnostics."""

import argparse
import json
from pathlib import Path

import yaml

from oddsjam_ev.analysis.sizing_diagnostics import SizingDiagnosticsConfig
from oddsjam_ev.io.sizing_diagnostics import publish_sizing_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--sizing", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/sizing_diagnostics.yaml"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/sizing_diagnostics")
    )
    args = parser.parse_args()
    try:
        cfg = SizingDiagnosticsConfig(**yaml.safe_load(args.config.read_text()))
        path = publish_sizing_diagnostics(args.canonical, args.sizing, args.output_root, cfg)
        print(f"Sizing diagnostics: {path}")
        print(json.dumps(json.loads((path / "audit.json").read_text()), sort_keys=True))
    except (OSError, ValueError, TypeError, KeyError, AssertionError) as exc:
        parser.exit(2, f"Sizing diagnostics failed: {exc}\n")


if __name__ == "__main__":
    main()
