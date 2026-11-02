"""Publish an explicitly cut off reviewed lifecycle and monitoring report."""

import argparse
from pathlib import Path

from oddsjam_ev.io.filter_lifecycle import publish_lifecycle_report


def main():
    """Pin a journal and knowledge cutoff; never infer applied scanner changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/filter_lifecycle_reports")
    )
    args = parser.parse_args()
    print(publish_lifecycle_report(args.register, args.output_root, args.as_of))


if __name__ == "__main__":
    main()
