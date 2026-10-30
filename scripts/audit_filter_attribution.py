"""Reconcile explicit before/after canonical attribution publications."""

import argparse
from pathlib import Path

from oddsjam_ev.io.attribution_audit import publish_attribution_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/attribution_audits")
    )
    args = parser.parse_args()
    try:
        print(publish_attribution_audit(args.before, args.after, args.output_root))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Attribution audit failed: {exc}\n")


if __name__ == "__main__":
    main()
