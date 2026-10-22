"""Build an exploratory discovery report from an explicit local atlas."""

import argparse
import json
from pathlib import Path

import yaml

from oddsjam_ev.analysis.discovery_config import DiscoveryConfig
from oddsjam_ev.io.discovery import load_discovery, publish_discovery


def main():
    """Publish a pinned report and print counts and gate status, not private money."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/discovery_policy.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/discovery"))
    parser.add_argument("--temporal-evidence", type=Path)
    args = parser.parse_args()
    cfg = DiscoveryConfig(**yaml.safe_load(args.config.read_text()))
    evidence = json.loads(args.temporal_evidence.read_text()) if args.temporal_evidence else None
    output = publish_discovery(args.atlas, args.output_root, cfg, evidence)
    tables = load_discovery(output)
    print(f"Discovery report: {output}")
    print(
        json.dumps(
            {
                "table_rows": {k: len(v) for k, v in tables.items()},
                "test_status": tables["candidate_results"].test_status.value_counts().to_dict(),
                "temporal_gate": tables["temporal_gates"].iloc[0].to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
