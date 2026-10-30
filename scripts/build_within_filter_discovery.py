"""Publish bounded descriptive parent/subgroup/complement evidence within filters."""

import argparse
import json
from pathlib import Path

import yaml

from oddsjam_ev.analysis.multidimensional_config import MultidimensionalConfig
from oddsjam_ev.analysis.multidimensional_discovery import DiscoveryBudgetExceeded
from oddsjam_ev.io.within_filter_discovery import publish_within_filter_discovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/multidimensional_discovery.yaml")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/within_filter_discovery")
    )
    args = parser.parse_args()
    try:
        cfg = MultidimensionalConfig(**yaml.safe_load(args.config.read_text()))
        output = publish_within_filter_discovery(
            args.canonical, args.atlas, args.diagnostics, args.output_root, cfg
        )
        audit = json.loads((output / "audit.json").read_text())
        print(f"Within-filter discovery: {output}")
        print(
            json.dumps(
                {
                    k: audit[k]
                    for k in (
                        "status",
                        "usage",
                        "candidate_count",
                        "all_partitions_reconcile",
                        "formal_validation_blocked",
                    )
                },
                sort_keys=True,
            )
        )
    except DiscoveryBudgetExceeded as exc:
        parser.exit(
            2, "Within-filter discovery failed: " + json.dumps(exc.audit, sort_keys=True) + "\n"
        )
    except (OSError, ValueError, TypeError, KeyError, AssertionError) as exc:
        parser.exit(2, f"Within-filter discovery failed: {exc}\n")


if __name__ == "__main__":
    main()
