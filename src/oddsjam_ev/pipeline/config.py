"""Explicit local pipeline policy and pinned configuration loading."""

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from oddsjam_ev.analysis.atlas_config import AtlasConfig
from oddsjam_ev.analysis.canonical_config import CanonicalConfig
from oddsjam_ev.analysis.discovery_config import DiscoveryConfig
from oddsjam_ev.analysis.filter_attribution import validate_registry
from oddsjam_ev.analysis.recommendation_config import RecommendationConfig
from oddsjam_ev.io.ingestion import SourceMetadata


@dataclass(frozen=True)
class PipelineConfig:
    """Paths are relative to repository working directory; scheduling is forbidden."""

    contract_version: int = 1
    scheduling_enabled: bool = False
    max_export_age_days: int = 8
    transient_retries: int = 1
    output_root: str = "data/processed/weekly_research"
    source_config: str = "configs/data_sources.yaml"
    canonical_config: str = "configs/canonical_facts.yaml"
    atlas_config: str = "configs/opportunity_atlas.yaml"
    discovery_config: str = "configs/discovery_policy.yaml"
    recommendation_config: str = "configs/research_policy.yaml"
    registry: str = "configs/filter_history/oddsjam_filter_configurations_2026-10-22.yaml"
    capabilities: str = "configs/filter_capabilities.yaml"

    def __post_init__(self):
        if self.contract_version != 1 or self.scheduling_enabled is not False:
            raise ValueError("Only unscheduled local pipeline contract 1 is supported")
        if type(self.max_export_age_days) is not int or self.max_export_age_days < 1:
            raise ValueError("Positive export age limit required")
        if type(self.transient_retries) is not int or not 0 <= self.transient_retries <= 2:
            raise ValueError("Retry limit must be an integer between zero and two")
        for key, value in asdict(self).items():
            if key.endswith("config") or key in {"output_root", "registry", "capabilities"}:
                if not isinstance(value, str) or not value or ".local" in Path(value).parts:
                    raise ValueError("Explicit non-credential paths required")


def read_config(path: Path) -> PipelineConfig:
    """Load only known policy fields; reject accidental scheduling."""
    return PipelineConfig(**yaml.safe_load(path.read_text()))


def configuration_bundle(cfg: PipelineConfig) -> dict:
    """Capture and validate configuration contents before processing any input."""
    names = (
        "source_config",
        "canonical_config",
        "atlas_config",
        "discovery_config",
        "recommendation_config",
        "registry",
        "capabilities",
    )
    bundle = {name: yaml.safe_load(Path(getattr(cfg, name)).read_text()) for name in names}
    source = bundle["source_config"]
    if set(source) != {"source", "store"} or not isinstance(source["store"], str):
        raise ValueError("Source configuration requires source and store")
    SourceMetadata(**source["source"])
    CanonicalConfig(**bundle["canonical_config"])
    AtlasConfig(**bundle["atlas_config"])
    DiscoveryConfig(**bundle["discovery_config"])
    RecommendationConfig(**bundle["recommendation_config"])
    validate_registry(bundle["registry"])
    return bundle
