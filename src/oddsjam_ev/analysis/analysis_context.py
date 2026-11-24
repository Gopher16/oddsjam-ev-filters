"""
analysis_context.py
===============================================================================
Purpose
-------
Preserve exact investigation scope across optional evidence views.

What this module provides
-------------------------
Frozen versioned scope, strict portable serialization and explicit transitions.

Design Notes
------------
Pins are identities, not evidence verification. Callers bind verified publications.
===============================================================================
"""

import base64
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime

VIEWS = {"charts", "explanations", "risk"}
MAX_BYTES = 16384


def _ids(values):
    if values is not None and (
        not isinstance(values, tuple)
        or any(not isinstance(v, str) or not v.strip() for v in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("IDs must be unique nonempty strings")


def _time(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Time must be an aware ISO timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("Time must be timezone-aware")
    return parsed


@dataclass(frozen=True)
class AnalysisContext:
    """Version-1 scope; no field is inferred from a filter name.

    Parameters
    ----------
    source_id, configuration_version, investigation_id : str
        Explicit stable identities supplied by the consuming workflow.
    publications : tuple of (str, str)
        Named immutable publication/manifest fingerprints, including regime/venue
        evidence when used. Caller must verify the referenced bytes independently.
    funding, product, selection_regime : str
        One exact population; mixed populations require separate contexts.
    cohort_ids, wager_ids, event_ids : tuple of str or None
        OR within each field, AND across fields. None is unrestricted; () is empty.
    start, end, time_column : str or None
        Optional half-open aware interval on an explicitly identified time basis.
    weighting : str
        equal_stake or stake_weighted; carried unchanged, never silently converted.
    view, focus : str
        Destination and selection origin (none/event_cluster/drawdown/cohort).
    contract_version : int
        Serialization schema version, exactly 1.
    """

    source_id: str
    configuration_version: str
    investigation_id: str
    publications: tuple[tuple[str, str], ...]
    funding: str
    product: str
    selection_regime: str
    cohort_ids: tuple[str, ...] | None = None
    wager_ids: tuple[str, ...] | None = None
    event_ids: tuple[str, ...] | None = None
    start: str | None = None
    end: str | None = None
    time_column: str | None = None
    weighting: str = "equal_stake"
    view: str = "charts"
    focus: str = "none"
    contract_version: int = 1

    def __post_init__(self):
        if type(self.contract_version) is not int or self.contract_version != 1:
            raise ValueError("Unsupported analysis context version")
        for field in (
            "source_id",
            "configuration_version",
            "investigation_id",
            "selection_regime",
            "product",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip() or value in {"all", "mixed"}:
                raise ValueError(f"Explicit {field} required")
        if self.funding not in {"paper", "real_money", "unknown"}:
            raise ValueError("One funding population required")
        if self.view not in VIEWS or self.focus not in {
            "none",
            "event_cluster",
            "drawdown",
            "cohort",
        }:
            raise ValueError("Unsupported view/focus")
        if self.weighting not in {"equal_stake", "stake_weighted"}:
            raise ValueError("Explicit supported weighting required")
        if not isinstance(self.publications, tuple) or not self.publications:
            raise ValueError("Publication pins required")
        names = []
        for pin in self.publications:
            if not isinstance(pin, tuple) or len(pin) != 2:
                raise ValueError("Invalid publication pin")
            _ids(pin)
            if pin[0] in {"source_id", "configuration_version"}:
                raise ValueError("Reserved publication name")
            names.append(pin[0])
        _ids(tuple(names))
        for values in (self.cohort_ids, self.wager_ids, self.event_ids):
            _ids(values)
        lo, hi = _time(self.start), _time(self.end)
        if (lo is None) != (hi is None) or (lo is not None and lo >= hi):
            raise ValueError("Window requires ordered start and end")
        if (lo is None) != (self.time_column is None):
            raise ValueError("Window requires an explicit time column")
        if self.time_column is not None and (
            not isinstance(self.time_column, str) or not self.time_column.strip()
        ):
            raise ValueError("Invalid time column")

    def to_json(self):
        """Return canonical session JSON, bounded to 16 KiB of UTF-8."""
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        if len(raw.encode()) > MAX_BYTES:
            raise ValueError("Context exceeds portable size limit")
        return raw

    @classmethod
    def from_json(cls, raw):
        """Parse strict session JSON; reject unknown fields and duplicate keys."""
        if not isinstance(raw, str) or len(raw.encode()) > MAX_BYTES:
            raise ValueError("Invalid context size")

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate context field")
                result[key] = value
            return result

        try:
            data = json.loads(raw, object_pairs_hook=unique)
            if not isinstance(data, dict) or "contract_version" not in data:
                raise ValueError("Explicit versioned context object required")
            for field in ("cohort_ids", "wager_ids", "event_ids"):
                if data.get(field) is not None:
                    if not isinstance(data[field], list):
                        raise ValueError("ID arrays required")
                    data[field] = tuple(data[field])
            if not isinstance(data.get("publications"), list) or any(
                not isinstance(pin, list) for pin in data["publications"]
            ):
                raise ValueError("Publication arrays required")
            data["publications"] = tuple(tuple(pin) for pin in data["publications"])
            return cls(**data)
        except (TypeError, KeyError, AttributeError) as exc:
            raise ValueError("Invalid context schema") from exc

    def to_token(self):
        """Return URL-safe encoding; this is not encryption or a public share link."""
        return base64.urlsafe_b64encode(self.to_json().encode()).decode().rstrip("=")

    @classmethod
    def from_token(cls, token):
        """Decode a bounded URL token without resolving any external references."""
        if not isinstance(token, str) or len(token) > 4 * MAX_BYTES // 3 + 4:
            raise ValueError("Invalid token size")
        try:
            raw = base64.b64decode(token + "=" * (-len(token) % 4), altchars=b"-_", validate=True)
            return cls.from_json(raw.decode())
        except (UnicodeError, ValueError) as exc:
            raise ValueError("Invalid context token") from exc

    def transition(self, **changes):
        """Return a new validated context and visible before/after field changes.

        Parameters
        ----------
        **changes : object
            Explicit replacements. Scope changes never clear or widen IDs implicitly.

        Returns
        -------
        tuple
            New context and changed-field mapping for the workspace scope indicator.
        """
        updated = replace(self, **changes)
        before, after = asdict(self), asdict(updated)
        return updated, {k: (before[k], after[k]) for k in before if before[k] != after[k]}

    def reset_selection(self):
        """Explicitly clear IDs/window/focus, retaining pinned population/investigation."""
        return self.transition(
            cohort_ids=None,
            wager_ids=None,
            event_ids=None,
            start=None,
            end=None,
            time_column=None,
            focus="none",
        )
