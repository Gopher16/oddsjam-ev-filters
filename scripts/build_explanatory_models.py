"""Publish explicit offline return models and SHAP; never train in the dashboard."""

import argparse
import json
from pathlib import Path

from oddsjam_ev.analysis.explanatory_models import ModelPolicy
from oddsjam_ev.io.explanatory_models import publish_explanatory_models


def main():
    """Build one immutable publication from explicit verified sources and JSON scope."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--visuals", type=Path, required=True)
    parser.add_argument(
        "--scope", type=Path, help="Optional JSON decision-time predicates; creates a subset refit"
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/explanatory_models")
    )
    args = parser.parse_args()
    scope = json.loads(args.scope.read_text()) if args.scope else {}
    if not isinstance(scope, dict):
        parser.error("Scope must be a JSON object")
    print(
        publish_explanatory_models(
            args.canonical, args.visuals, args.output_root, ModelPolicy(), scope
        )
    )


if __name__ == "__main__":
    main()
