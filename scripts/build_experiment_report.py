"""Build pinned paper experiment outcomes and readiness audits."""

import argparse
from pathlib import Path

from oddsjam_ev.io.experiment_reports import load_experiment_report, publish_experiment_report


def main():
    """Require explicit knowledge cutoff and canonical source; print counts only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/experiment_reports")
    )
    args = parser.parse_args()
    out = publish_experiment_report(args.register, args.canonical, args.output_root, args.as_of)
    print(f"Experiment report: {out}")
    print({k: len(v) for k, v in load_experiment_report(out).items()})


if __name__ == "__main__":
    main()
