"""Publish explicit all-history review queues; no inference rerun or filter execution."""

import argparse
from pathlib import Path

import yaml

from oddsjam_ev.analysis.review_config import ReviewConfig
from oddsjam_ev.io.review_queues import publish_review_queues


def main():
    """Build verified queue artifacts from four explicitly pinned publications."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("discovery", "prioritization", "decisions", "visuals"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/review_queues.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/review_queues"))
    args = parser.parse_args()
    try:
        cfg = ReviewConfig(**yaml.safe_load(args.config.read_text()))
        output = publish_review_queues(
            args.discovery, args.prioritization, args.decisions, args.visuals, args.output_root, cfg
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(
            2,
            f"Review queue publication failed: {type(exc).__name__}; inspect explicit inputs and failure audits.\n",
        )
    print(output)


if __name__ == "__main__":
    main()
