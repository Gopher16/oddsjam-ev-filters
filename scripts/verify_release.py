"""Read-only acceptance smoke for one explicitly selected dashboard report."""

import argparse
import json
import sys
from pathlib import Path

from oddsjam_ev.dashboard.acceptance import check_performance, inspect_release
from oddsjam_ev.dashboard.artifacts import read_dashboard_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-cold-seconds", type=float)
    parser.add_argument("--max-warm-seconds", type=float)
    parser.add_argument("--max-rss-mib", type=float)
    args = parser.parse_args()
    budgets = [args.max_cold_seconds, args.max_warm_seconds, args.max_rss_mib]
    if any(v is not None for v in budgets) and not all(v is not None for v in budgets):
        parser.error("Supply all three performance budgets together")
    result = inspect_release(read_dashboard_config(args.config), args.run_id, args.repeats)
    if budgets[0] is not None:
        result["performance_budget"] = check_performance(result, *budgets)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result.get("performance_budget", {}).get("passed") is False:
        sys.exit(2)


if __name__ == "__main__":
    main()
