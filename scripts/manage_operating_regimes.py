"""Explicit private operating-evidence template, migration and publication tools."""

import argparse
import json
from pathlib import Path

from oddsjam_ev.analysis.operating_regimes import migrate_evidence, regime_record
from oddsjam_ev.io.canonical import load_canonical_bundle
from oddsjam_ev.io.manifests import write_json
from oddsjam_ev.io.operating_regimes import publish_operating_regimes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    publish = sub.add_parser("publish")
    publish.add_argument("--canonical", type=Path, required=True)
    publish.add_argument("--history", type=Path, required=True)
    publish.add_argument("--previous", type=Path)
    publish.add_argument(
        "--output-root", type=Path, default=Path("data/processed/operating_regimes")
    )
    migrate = sub.add_parser("migrate")
    migrate.add_argument("--canonical", type=Path, required=True)
    migrate.add_argument("--sizing-history", type=Path)
    migrate.add_argument("--decision-evidence", type=Path)
    migrate.add_argument("--knowledge-as-of", required=True)
    migrate.add_argument("--output", type=Path, required=True)
    template = sub.add_parser("template")
    template.add_argument("--recorded-at", required=True)
    template.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "publish":
            print(
                publish_operating_regimes(
                    args.canonical, args.output_root, args.history, previous=args.previous
                )
            )
            return
        if args.command == "migrate":
            if args.sizing_history is None and args.decision_evidence is None:
                raise ValueError("Choose a legacy evidence input")
            load_canonical_bundle(args.canonical)
            cm = json.loads((args.canonical / "manifest.json").read_text())
            result = migrate_evidence(
                cm["specification"]["filter_registry"],
                args.knowledge_as_of,
                sizing=json.loads(args.sizing_history.read_text()) if args.sizing_history else None,
                decision=json.loads(args.decision_evidence.read_text())
                if args.decision_evidence
                else None,
            )
        else:
            from oddsjam_ev.analysis.sizing_config import aware

            aware(args.recorded_at)
            row = regime_record(
                "replace_evidence_id",
                "replace_private_namespace",
                args.recorded_at,
                "Replace with scoped owner confirmation/source reference",
            )
            row.update(
                precision="date",
                effective_from="2026-11-10",
                effective_to=None,
                evidence_type="user_confirmed",
                confidence="confirmed",
                funding="paper",
                selection_mode="discretionary",
                paper_sizing="kelly_recommendation",
            )
            result = dict(
                contract_version=1,
                knowledge_as_of=args.recorded_at,
                previous_history_id=None,
                records=[row],
                wager_evidence=[],
            )
        # Require an ignored private destination, never write an account ledger to configs/Git.
        repo = Path(__file__).resolve().parents[1]
        private = repo / "data/processed"
        target = args.output.resolve()
        if not target.is_relative_to(private.resolve()):
            raise ValueError("Generated ledgers must be under ignored data/processed")
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target, result)
        print(target)
    except (ValueError, OSError, KeyError, TypeError, AssertionError) as exc:
        parser.exit(2, f"Operating evidence failed: {exc}\n")


if __name__ == "__main__":
    main()
