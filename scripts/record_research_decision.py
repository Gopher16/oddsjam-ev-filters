"""Inspect or explicitly record local human research decisions; never apply filters."""

import argparse
import json
from pathlib import Path

from oddsjam_ev.io.decision_register import read_decisions, record_decision


def main():
    """Require explicit reviewer, rationale, exact proposal and expected state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", type=Path, default=Path("data/processed/decisions"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect")
    add = sub.add_parser("record")
    add.add_argument("--bundle", type=Path, required=True)
    for key in ["proposal-id", "to-status", "reviewer", "rationale", "expected-status"]:
        add.add_argument("--" + key, required=True)
    add.add_argument("--expected-event-id")
    args = parser.parse_args()
    if args.command == "inspect":
        print(json.dumps(read_decisions(args.register), indent=2))
    else:
        event = record_decision(
            args.register,
            args.bundle,
            args.proposal_id,
            args.to_status,
            args.reviewer,
            args.rationale,
            args.expected_status,
            args.expected_event_id,
        )
        print(json.dumps(event, indent=2))


if __name__ == "__main__":
    main()
