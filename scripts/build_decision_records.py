"""Publish complete configuration sheets; missing evidence produces explicit blockers."""

import argparse
from pathlib import Path

import yaml

from oddsjam_ev.io.decision_records import publish_decisions


def main():
    """Read explicit source/evidence snapshots without inferring current lifecycle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--prioritization", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--effective-at", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/decision_records"))
    args = parser.parse_args()
    try:
        result = publish_decisions(
            args.discovery,
            args.prioritization,
            yaml.safe_load(args.evidence.read_text()),
            args.output_root,
            args.as_of,
            args.effective_at,
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(
            2, f"Decision publication failed: {type(exc).__name__}; review explicit inputs.\n"
        )
    print(result)


if __name__ == "__main__":
    main()
