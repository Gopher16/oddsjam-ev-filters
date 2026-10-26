"""Inventory exact historical labels from explicit verified source bundles."""

import argparse
import json
from pathlib import Path

from oddsjam_ev.io.label_inventory import load_label_inventory, publish_label_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/filter_label_inventory")
    )
    args = parser.parse_args()
    try:
        path = publish_label_inventory(args.snapshot, args.canonical, args.output_root)
        print(f"Label inventory: {path}")
        print(json.dumps(load_label_inventory(path)["summary"], sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Label inventory failed: {exc}\n")


if __name__ == "__main__":
    main()
