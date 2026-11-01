"""Verified model lineage, sparse outcomes, safe replay and corruption rejection."""

import json

import pytest
from test_within_filter_publication import sources

from oddsjam_ev.analysis.explanatory_models import ModelPolicy
from oddsjam_ev.analysis.visual_config import VisualConfig
from oddsjam_ev.io.explanatory_models import load_model_publication, publish_explanatory_models
from oddsjam_ev.io.visual_diagnostics import publish_visual_diagnostics


def test_sparse_publication_and_replay(tmp_path):
    canonical, atlas, sizing = sources(tmp_path)
    visuals = publish_visual_diagnostics(
        canonical, atlas, sizing, tmp_path / "visuals", VisualConfig()
    )
    originals = {
        p: p.read_bytes() for root in (canonical, visuals) for p in root.iterdir() if p.is_file()
    }
    root = tmp_path / "models"
    output = publish_explanatory_models(canonical, visuals, root, ModelPolicy())
    loaded = load_model_publication(output)
    assert loaded["predictions"].empty
    assert loaded["audit"]["source_wagers"] == 1
    assert loaded["audit"]["populations"][0]["status"] == "declined"
    assert len(loaded["feature_inventory"]) > 100
    assert publish_explanatory_models(canonical, visuals, root, ModelPolicy()) == output
    assert all(p.read_bytes() == data for p, data in originals.items())
    assert not list(output.glob("*.pkl"))
    (output / "models.json").write_text("{} corrupted")
    with pytest.raises(ValueError, match="integrity"):
        load_model_publication(output)


def test_scope_failure_has_no_partial_models(tmp_path):
    canonical, atlas, sizing = sources(tmp_path)
    visuals = publish_visual_diagnostics(
        canonical, atlas, sizing, tmp_path / "visuals", VisualConfig()
    )
    root = tmp_path / "models"
    # A duplicate unsupported scope never gets interpreted as a predictive feature.
    with pytest.raises(ValueError, match="Unsupported"):
        publish_explanatory_models(canonical, visuals, root, ModelPolicy(), {"profit": {"gt": 0}})
    output = publish_explanatory_models(
        canonical, visuals, root, ModelPolicy(), {"sportsbook": ["not present"]}
    )
    assert load_model_publication(output)["audit"]["scope_empty"]
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["specification"]["scope"] = {}
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="specification integrity"):
        load_model_publication(output)


def test_failure_audit_does_not_publish_partial_output(tmp_path, monkeypatch):
    import oddsjam_ev.io.explanatory_models as module

    canonical, atlas, sizing = sources(tmp_path)
    visuals = publish_visual_diagnostics(
        canonical, atlas, sizing, tmp_path / "visuals", VisualConfig()
    )
    root = tmp_path / "models"

    def fail(*args):
        raise ValueError("synthetic resource failure")

    monkeypatch.setattr(module, "audit_features", fail)
    with pytest.raises(ValueError, match="resource"):
        publish_explanatory_models(canonical, visuals, root, ModelPolicy())
    failures = list(root.glob("*.failure.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text())["exception"] == "ValueError"
    assert not list(root.glob("*/manifest.json"))
