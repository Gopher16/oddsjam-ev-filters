"""Prepare a private historical identity proposal; never apply aliases."""

import argparse
import json
from pathlib import Path

from oddsjam_ev.io.filter_mapping_proposal import load_mapping_proposal, publish_mapping_proposal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/filter_mapping_proposals")
    )
    args = parser.parse_args()
    try:
        path = publish_mapping_proposal(args.inventory, args.canonical, args.output_root)
        print(f"Mapping proposal: {path}")
        print(json.dumps(load_mapping_proposal(path)["summary"], sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Mapping proposal failed: {exc}\n")


if __name__ == "__main__":
    main()
