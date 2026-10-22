"""Build a private, descriptive atlas from one pinned canonical bundle."""

import argparse
from pathlib import Path

import yaml

from oddsjam_ev.analysis.atlas_config import AtlasConfig
from oddsjam_ev.io.atlas import publish_atlas


def main():
    """Parse explicit inputs and publish an atlas without notebook execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/opportunity_atlas.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/atlas"))
    args = parser.parse_args()
    cfg = AtlasConfig(**yaml.safe_load(args.config.read_text()))
    output = publish_atlas(args.bundle, args.output_root, cfg)
    print(f"Atlas bundle: {output}")


if __name__ == "__main__":
    main()
