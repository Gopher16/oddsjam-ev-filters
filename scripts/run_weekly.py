"""Run explicit local research inputs; never schedule, acquire, or place bets."""

import argparse
import json
from pathlib import Path

from oddsjam_ev.pipeline.config import read_config
from oddsjam_ev.pipeline.runner import run_pipeline


def main():
    """Print identifiers and stable status codes only; private reports stay local."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/weekly_research.yaml"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path)
    source.add_argument("--snapshot", type=Path)
    source.add_argument("--resume", help="Exact frozen run ID from a prior attempt")
    parser.add_argument("--as-of", help="Aware knowledge cutoff, required with --snapshot")
    parser.add_argument("--register", type=Path, help="Optional explicit experiment register")
    parser.add_argument("--temporal-evidence", type=Path)
    parser.add_argument(
        "--decisions", type=Path, help="Reviewed ingestion resolutions for CSV mode"
    )
    args = parser.parse_args()
    try:
        cfg = read_config(args.config)
        result = run_pipeline(
            cfg,
            csv=args.csv,
            snapshot=args.snapshot,
            as_of=args.as_of,
            resume=args.resume,
            register=args.register,
            temporal_evidence=args.temporal_evidence,
            decisions=args.decisions,
        )
    except (OSError, ValueError, TypeError, KeyError):
        parser.exit(2, "Pipeline failed: configuration_or_lock_failure\n")
    print(
        json.dumps(
            {k: result[k] for k in ("attempt_id", "run_id", "status", "exit_code", "error_code")},
            sort_keys=True,
        )
    )
    raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
