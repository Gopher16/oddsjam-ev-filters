"""Publish continuous EV values from an explicit verified canonical bundle."""

import argparse
import json
from pathlib import Path

from oddsjam_ev.io.continuous_ev import publish_continuous_ev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/continuous_ev"))
    args = parser.parse_args()
    try:
        path = publish_continuous_ev(args.canonical, args.output_root)
        print(f"Continuous EV: {path}")
        print(
            json.dumps(json.loads((path / "manifest.json").read_text())["summary"], sort_keys=True)
        )
    except (OSError, ValueError, TypeError, KeyError, AssertionError) as exc:
        parser.exit(2, f"Continuous EV build failed: {exc}\n")


if __name__ == "__main__":
    main()
