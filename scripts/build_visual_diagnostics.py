"""Publish reusable verified chart data; no notebook execution or inferential rerun."""

import argparse
from pathlib import Path

import yaml

from oddsjam_ev.analysis.visual_config import VisualConfig
from oddsjam_ev.io.visual_diagnostics import publish_visual_diagnostics


def main():
    """Build explicit visual sources and fail without partial consumable charts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/visual_diagnostics.yaml"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/visual_diagnostics")
    )
    args = parser.parse_args()
    try:
        cfg = VisualConfig(**yaml.safe_load(args.config.read_text()))
        output = publish_visual_diagnostics(
            args.canonical, args.atlas, args.diagnostics, args.output_root, cfg
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(
            2,
            f"Visual publication failed: {type(exc).__name__}; review inputs and failure audits.\n",
        )
    print(output)


if __name__ == "__main__":
    main()
