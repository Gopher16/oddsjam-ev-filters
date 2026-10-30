"""Publish approved exploratory paired statistics and overlap-aware presentation."""

import argparse
import json
from pathlib import Path

import yaml

from oddsjam_ev.analysis.paired_comparison import PrioritizationBudgetExceeded
from oddsjam_ev.analysis.prioritization_config import PrioritizationConfig
from oddsjam_ev.io.signal_prioritization import publish_prioritization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/signal_prioritization.yaml"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/signal_prioritization")
    )
    args = parser.parse_args()
    try:
        cfg = PrioritizationConfig(**yaml.safe_load(args.config.read_text()))
        output = publish_prioritization(args.discovery, args.diagnostics, args.output_root, cfg)
        print(f"Signal prioritization: {output}")
        print(json.dumps(json.loads((output / "audit.json").read_text()), sort_keys=True))
    except PrioritizationBudgetExceeded as exc:
        parser.exit(2, "Prioritization failed: " + json.dumps(exc.audit, sort_keys=True) + "\n")
    except (OSError, ValueError, TypeError, KeyError, AssertionError) as exc:
        parser.exit(2, f"Prioritization failed: {exc}\n")


if __name__ == "__main__":
    main()
