"""Publish private sizing-history provenance and unknown/ambiguous linkage audits."""

import argparse
import json
from pathlib import Path

from oddsjam_ev.io.sizing_history import load_sizing_history, publish_sizing_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--history", type=Path)
    source.add_argument("--knowledge-as-of", help="Aware cutoff for explicit empty-history audit")
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/sizing_history"))
    args = parser.parse_args()
    try:
        path = publish_sizing_history(
            args.canonical,
            args.output_root,
            history_path=args.history,
            previous=args.previous,
            knowledge_as_of=args.knowledge_as_of,
        )
        audit = load_sizing_history(path)[2]
        print(f"Sizing history: {path}")
        print(
            json.dumps(
                {
                    key: audit[key]
                    for key in ("candidate_count", "linkage_status_counts", "sizing_replay_blocked")
                },
                sort_keys=True,
            )
        )
    except (OSError, ValueError, TypeError, KeyError, AssertionError) as exc:
        parser.exit(2, f"Sizing history failed: {exc}\n")


if __name__ == "__main__":
    main()
