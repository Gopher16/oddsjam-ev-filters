"""Explicit local paper-experiment registration and evidence commands."""

import argparse
import json
from pathlib import Path

import yaml

from oddsjam_ev.analysis.experiment_config import ExperimentPolicy
from oddsjam_ev.io.experiment_register import (
    read_experiment_events,
    record_experiment_event,
    register_experiment,
)


def main():
    """Read explicit payloads; never infer launch, funding or enrollment from labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", type=Path, default=Path("data/processed/experiments"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect")
    register = sub.add_parser("register")
    register.add_argument("--bundle", type=Path, required=True)
    register.add_argument("--decisions", type=Path, required=True)
    register.add_argument("--proposal-id", required=True)
    register.add_argument("--approval-event-id", required=True)
    register.add_argument("--protocol", type=Path, required=True)
    register.add_argument("--policy", type=Path, default=Path("configs/experiment_policy.yaml"))
    event = sub.add_parser("event")
    event.add_argument("--experiment-id", required=True)
    event.add_argument(
        "--kind", choices=["TRANSITION", "ENROLL", "LINK", "CHECK", "DEVIATION"], required=True
    )
    event.add_argument("--payload", type=Path, required=True)
    event.add_argument("--expected-event-id", required=True)
    for command in [register, event]:
        command.add_argument("--actor", required=True)
        command.add_argument("--rationale", required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        result = read_experiment_events(args.register)
    elif args.command == "register":
        result = register_experiment(
            args.register,
            args.bundle,
            args.decisions,
            args.proposal_id,
            args.approval_event_id,
            yaml.safe_load(args.protocol.read_text()),
            args.actor,
            args.rationale,
            ExperimentPolicy(**yaml.safe_load(args.policy.read_text())),
        )
    else:
        result = record_experiment_event(
            args.register,
            args.experiment_id,
            args.kind,
            json.loads(args.payload.read_text()),
            args.actor,
            args.rationale,
            args.expected_event_id,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
