"""Explicit local reviewed decisions, configuration confirmations and monitoring commands."""

import argparse
import json
from pathlib import Path

from oddsjam_ev.io.filter_lifecycle import read_lifecycle, record_lifecycle_event


def main():
    """Require an explicit actor, reason, payload and current journal head for mutations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", type=Path, default=Path("data/processed/filter_lifecycle"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect")
    record = sub.add_parser("record")
    record.add_argument(
        "--kind",
        choices=[
            "POLICY",
            "EVALUATE_PROMOTION",
            "DECISION",
            "BASELINE",
            "CHANGE",
            "APPROVE_CHANGE",
            "REJECT_CHANGE",
            "CONFIRM_APPLICATION",
            "OBSERVE",
            "SETTLE",
            "MONITOR_REVIEW",
        ],
        required=True,
    )
    record.add_argument("--payload", type=Path, required=True)
    record.add_argument("--expected-event-id")
    record.add_argument("--actor", required=True)
    record.add_argument("--rationale", required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        result = read_lifecycle(args.register)
    else:
        result = record_lifecycle_event(
            args.register,
            args.kind,
            json.loads(args.payload.read_text()),
            args.actor,
            args.rationale,
            args.expected_event_id,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
