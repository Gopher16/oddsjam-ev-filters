"""Publish one exact definition version for human approval; never launch it."""

import argparse
from pathlib import Path

from oddsjam_ev.io.prospective import publish_definition_proposal


def main():
    """Require an explicit version, scanner identity and parent capture policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definitions", type=Path, required=True)
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--filter-id", required=True)
    parser.add_argument("--sampling-rule", required=True)
    parser.add_argument(
        "--experiment-type",
        choices=["signal_screening", "execution_verification"],
        default="signal_screening",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/definition_proposals")
    )
    args = parser.parse_args()
    print(
        publish_definition_proposal(
            args.definitions,
            args.version_id,
            args.output_root,
            args.filter_id,
            args.sampling_rule,
            args.experiment_type,
        )
    )


if __name__ == "__main__":
    main()
