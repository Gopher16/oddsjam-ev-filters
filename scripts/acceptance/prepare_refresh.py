"""Create isolated synthetic R10 browser fixtures; never reads private history."""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import yaml

from oddsjam_ev.analysis.canonical_config import CanonicalConfig
from oddsjam_ev.io.ingestion import SourceMetadata
from oddsjam_ev.pipeline.config import PipelineConfig


def prepare(root: Path, rows: int):
    """Write fixed synthetic inputs and trusted configs to a new private directory."""
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    store = root / "store"
    store.mkdir(mode=0o700)
    values = {
        "source_config": {"store": str(store), "source": asdict(SourceMetadata("demo"))},
        "canonical_config": asdict(
            CanonicalConfig({"fixed_odds": ["Book"], "exchange": [], "pickem": []})
        ),
        "atlas_config": {"contract_version": 1},
        "discovery_config": {"bootstrap_resamples": 100},
        "recommendation_config": {"contract_version": 1},
        "registry": {
            "schema_version": 1,
            "snapshot_date": "2026-11-24",
            "filters": [{"id": "demo", "name": "TEST - Demo"}],
        },
        "capabilities": {"schema_version": 1, "exact_values": {}},
    }
    paths = {}
    for key, value in values.items():
        path = root / (key + ".yaml")
        path.write_text(yaml.safe_dump(value))
        paths[key] = str(path)
    cfg = PipelineConfig(output_root=str(root / "reports"), **paths)
    pipeline = root / "pipeline.yaml"
    pipeline.write_text(yaml.safe_dump(asdict(cfg)))
    (root / "dashboard.yaml").write_text(
        yaml.safe_dump({"artifact_root": cfg.output_root, "refresh_pipeline_config": str(pipeline)})
    )
    row = {
        "created_at": "2026-11-01 12:00:00",
        "sportsbook": "Book",
        "game_id": "demo",
        "market_name": "Moneyline",
        "bet_name": "Away",
        "odds": "120",
        "stake": "10",
        "bet_profit": "12",
        "potential_payout": "22",
        "status": "won",
        "bet_type": "positive_ev",
        "saved_filter_names": "TEST - Demo",
        "is_live_bet": "False",
        "is_free_bet": "False",
        "is_odds_boost": "False",
    }
    for filename, revised in (("history.csv", False), ("revision.csv", True)):
        with (root / filename).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=row)
            writer.writeheader()
            for i in range(rows):
                writer.writerow(
                    {
                        **row,
                        "game_id": f"demo-{i}",
                        "bet_profit": "-10" if i % 2 or revised and i == 0 else "12",
                        "status": "lost" if i % 2 or revised and i == 0 else "won",
                    }
                )
    (root / "invalid.csv").write_text("bad,headers\nsynthetic,only\n")
    print(root / "dashboard.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--rows", type=int, default=1000)
    args = parser.parse_args()
    if not 1 <= args.rows <= 100_000:
        parser.error("rows must be between 1 and 100000")
    prepare(args.root.resolve(), args.rows)
