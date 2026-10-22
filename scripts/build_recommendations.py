"""Build research-only recommendations from explicitly linked local bundles."""

import argparse
from pathlib import Path

import yaml

from oddsjam_ev.analysis.recommendation_config import RecommendationConfig
from oddsjam_ev.io.recommendations import load_recommendations, publish_recommendations


def main():
    """Generate proposals and print action counts without private monetary evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["report", "atlas", "canonical"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path("configs/research_policy.yaml"))
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/filter_history/oddsjam_filter_configurations_2026-10-22.yaml"),
    )
    parser.add_argument(
        "--capabilities", type=Path, default=Path("configs/filter_capabilities.yaml")
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/recommendations"))
    args = parser.parse_args()

    def read(path):
        return yaml.safe_load(path.read_text())

    output = publish_recommendations(
        args.report,
        args.atlas,
        args.canonical,
        args.output_root,
        RecommendationConfig(**read(args.policy)),
        read(args.registry),
        read(args.capabilities),
    )
    print(f"Recommendation bundle: {output}")
    print(load_recommendations(output)["proposals"].action.value_counts().to_json())


if __name__ == "__main__":
    main()
