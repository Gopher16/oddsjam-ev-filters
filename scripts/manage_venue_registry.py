"""Seed, revise, publish and inspect optional venue/product capability evidence."""

import argparse
import copy
import json
from pathlib import Path

import yaml

from oddsjam_ev.analysis.venue_registry import (
    KINDS,
    OPERATIONS,
    resolve_product,
    seed_from_policy,
    validate_registry,
)
from oddsjam_ev.io.manifests import write_json
from oddsjam_ev.io.source_csv import fingerprint
from oddsjam_ev.io.venue_registry import load_venue_registry, publish_venue_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    seed = sub.add_parser(
        "seed", help="Copy historical classifications; leave contracts/access unknown"
    )
    seed.add_argument("--policy", type=Path, default=Path("configs/canonical_facts.yaml"))
    seed.add_argument("--recorded-at", required=True)
    seed.add_argument("--output", type=Path, required=True)
    request = sub.add_parser(
        "request", help="Append a request to an explicit published predecessor"
    )
    request.add_argument("--previous", type=Path, required=True)
    request.add_argument("--request-id", required=True)
    request.add_argument("--name", required=True)
    request.add_argument("--venue-id")
    request.add_argument("--product-type", choices=sorted(KINDS), default="unknown")
    request.add_argument("--recorded-at", required=True)
    request.add_argument("--reference", required=True)
    request.add_argument("--notes", required=True)
    request.add_argument("--output", type=Path, required=True)
    publish = sub.add_parser("publish")
    publish.add_argument("--registry", type=Path, required=True)
    publish.add_argument("--previous", type=Path)
    publish.add_argument("--output-root", type=Path, default=Path("data/processed/venue_registry"))
    lookup = sub.add_parser("lookup")
    lookup.add_argument("--publication", type=Path, required=True)
    lookup.add_argument("--venue", required=True)
    lookup.add_argument("--source-id", required=True)
    lookup.add_argument("--version-id")
    lookup.add_argument("--label")
    lookup.add_argument("--product-type", choices=sorted(KINDS))
    lookup.add_argument("--at")
    lookup.add_argument("--namespace")
    lookup.add_argument("--operation", choices=sorted(OPERATIONS), default="manual_csv_import")
    args = parser.parse_args()
    try:
        if args.command == "publish":
            print(publish_venue_registry(args.registry, args.output_root, previous=args.previous))
            return
        if args.command == "lookup":
            result = resolve_product(
                load_venue_registry(args.publication),
                args.venue,
                source_id=args.source_id,
                version_id=args.version_id,
                label=args.label,
                product_type=args.product_type,
                at=args.at,
                namespace=args.namespace,
                operation=args.operation,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        if args.command == "seed":
            result = seed_from_policy(yaml.safe_load(args.policy.read_text()), args.recorded_at)
        else:
            prior = load_venue_registry(args.previous)
            result = copy.deepcopy(prior)
            result.update(previous_registry_id=fingerprint(prior), knowledge_as_of=args.recorded_at)
            result["requests"].append(
                dict(
                    request_id=args.request_id,
                    name=args.name,
                    venue_id=args.venue_id,
                    product_type=args.product_type,
                    recorded_at=args.recorded_at,
                    reference=args.reference,
                    status="requested",
                    notes=args.notes,
                )
            )
            validate_registry(result, prior)
        private = Path(__file__).resolve().parents[1] / "data/processed"
        target = args.output.resolve()
        if not target.is_relative_to(private.resolve()):
            raise ValueError("Generated registry must be under ignored data/processed")
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target, result)
        print(target)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"Venue registry failed: {exc}\n")


if __name__ == "__main__":
    main()
