"""Explicit candidate-bound paper definitions; registration never launches a test."""

import argparse
import json
from pathlib import Path

import yaml

from oddsjam_ev.analysis.paper_definitions import draft_definition
from oddsjam_ev.io.paper_definitions import (
    publish_definitions,
    read_definitions,
    register_definition,
)
from oddsjam_ev.io.within_filter_discovery import load_within_filter_discovery


def main():
    """Validate explicit local inputs and print definition/publication identifiers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--register", type=Path, default=Path("data/processed/paper_definition_register")
    )
    commands = parser.add_subparsers(dest="command", required=True)
    draft = commands.add_parser("draft")
    draft.add_argument("--discovery", type=Path, required=True)
    draft.add_argument("--candidate-id", required=True)
    create = commands.add_parser("save")
    create.add_argument("--discovery", type=Path, required=True)
    create.add_argument("--prioritization", type=Path, required=True)
    create.add_argument("--definition", type=Path, required=True)
    create.add_argument("--test-key", required=True)
    create.add_argument("--actor", required=True)
    create.add_argument("--rationale", required=True)
    create.add_argument("--expected-version-id")
    publish = commands.add_parser("publish")
    publish.add_argument("--as-of", required=True)
    publish.add_argument(
        "--output-root", type=Path, default=Path("data/processed/paper_definitions")
    )
    commands.add_parser("inspect")
    args = parser.parse_args()
    try:
        if args.command in {"draft", "save"}:
            family = load_within_filter_discovery(args.discovery)
            document = (
                yaml.safe_load(args.definition.read_text()) if args.command == "save" else None
            )
            candidate_id = document["candidate_id"] if document else args.candidate_id
            candidates = family["signal_candidates"].loc[lambda f: f.candidate_id.eq(candidate_id)]
            if len(candidates) != 1:
                raise ValueError("Exact published candidate required")
            draft = draft_definition(candidates.iloc[0].to_dict())
            if args.command == "draft":
                print(yaml.safe_dump(draft, sort_keys=False))
                return
            if set(document) != set(draft) or json.loads(
                document["exact_predicate_json"]
            ) != json.loads(draft["exact_predicate_json"]):
                raise ValueError("Draft source rules changed; select a new candidate instead")
            result = register_definition(
                args.register,
                args.discovery,
                args.prioritization,
                candidate_id,
                args.test_key,
                document["protocol"],
                args.actor,
                args.rationale,
                args.expected_version_id,
            )
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in (
                            "test_key",
                            "version",
                            "version_id",
                            "status",
                            "evaluation_status",
                        )
                    },
                    indent=2,
                )
            )
        elif args.command == "publish":
            print(publish_definitions(args.register, args.output_root, args.as_of))
        else:
            print(json.dumps(read_definitions(args.register), indent=2))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(
            2,
            f"Paper definition command failed: {type(exc).__name__}. Review explicit inputs and publication integrity.\n",
        )


if __name__ == "__main__":
    main()
