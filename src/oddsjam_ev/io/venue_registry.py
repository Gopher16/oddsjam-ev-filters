"""
venue_registry.py
===============================================================================
Purpose
-------
Publish explicit venue capability evidence without modifying historical economics.

What this module provides
-------------------------
Byte-verified immutable snapshots, predecessor-pinned revisions and deterministic replay.

Design Notes
------------
No current-policy fallback, registry activation, account connection or execution.
===============================================================================
"""

import hashlib
import json
import platform
import tempfile
from pathlib import Path

from oddsjam_ev.analysis.venue_registry import contract_blockers, validate_registry
from oddsjam_ev.io.manifests import verify_bytes, write_json, writer_lock
from oddsjam_ev.io.source_csv import fingerprint


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _summary(registry):
    return dict(
        venue_count=len(registry["venues"]),
        product_version_count=len(registry["products"]),
        request_count=len(registry["requests"]),
        access_assertion_count=len(registry["access"]),
        incomplete_contract_count=sum(bool(contract_blockers(p)) for p in registry["products"]),
        canonical_economics_unchanged=True,
        execution_authorized=False,
    )


def load_venue_registry(path: Path) -> dict:
    """Load a verified detached publication without external-state substitution.

    Parameters
    ----------
    path : pathlib.Path
        Explicit content-addressed registry publication directory.

    Returns
    -------
    dict
        Validated ledger; predecessor integrity was checked at publication time.
    """
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    spec = manifest["specification"]
    if (
        type(spec.get("contract_version")) is not int
        or spec["contract_version"] != 1
        or fingerprint(spec) != manifest["publication_id"]
    ):
        raise ValueError("Venue publication specification integrity failure")
    if set(manifest["output_sha256"]) != {"registry.json"}:
        raise ValueError("Incomplete venue publication")
    data = verify_bytes(path / "registry.json", manifest["output_sha256"]["registry.json"])
    registry = json.loads(data)
    validate_registry({**registry, "previous_registry_id": None})
    if fingerprint(registry) != spec["registry_id"] or _summary(registry) != manifest["summary"]:
        raise ValueError("Venue publication identity/coverage mismatch")
    return registry


def publish_venue_registry(
    registry_path: Path, output_root: Path, *, previous: Path | None = None
) -> Path:
    """Publish a validated immutable registry or verify an identical replay.

    Parameters
    ----------
    registry_path : pathlib.Path
        Explicit JSON ledger; its exact source bytes are pinned.
    output_root : pathlib.Path
        Private publication root outside predecessor publications.
    previous : pathlib.Path or None
        Verified prior publication required for append-only revisions.

    Returns
    -------
    pathlib.Path
        Verified new/replayed content-addressed publication.
    """
    registry_path, output_root = Path(registry_path), Path(output_root)
    if previous is not None and output_root.resolve().is_relative_to(Path(previous).resolve()):
        raise ValueError("Output must not be inside predecessor publication")
    data = registry_path.read_bytes()
    registry = json.loads(data)
    prior = load_venue_registry(previous) if previous is not None else None
    ident = validate_registry(registry, prior)
    base = Path(__file__).parents[1]
    code = [
        Path(__file__),
        base / "analysis/venue_registry.py",
        base / "analysis/canonical_config.py",
        base / "analysis/sizing_config.py",
        base / "analysis/experiment_config.py",
        base / "io/source_csv.py",
    ]
    spec = dict(
        contract_version=1,
        registry_id=ident,
        source_sha256=hashlib.sha256(data).hexdigest(),
        previous_manifest_sha256=_digest(Path(previous) / "manifest.json") if previous else None,
        code_sha256={str(p.relative_to(base)): _digest(p) for p in code},
        runtime=dict(python=platform.python_version()),
    )
    publication_id = fingerprint(spec)
    with writer_lock(output_root):
        dest = output_root / publication_id
        if dest.exists():
            if load_venue_registry(dest) != registry:
                raise ValueError("Venue replay mismatch")
            return dest
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=output_root) as tmp:
            stage = Path(tmp)
            write_json(stage / "registry.json", registry)
            write_json(
                stage / "manifest.json",
                dict(
                    publication_id=publication_id,
                    specification=spec,
                    summary=_summary(registry),
                    output_sha256={"registry.json": _digest(stage / "registry.json")},
                ),
            )
            if load_venue_registry(stage) != registry:
                raise ValueError("Venue staging mismatch")
            stage.rename(dest)
    return dest
