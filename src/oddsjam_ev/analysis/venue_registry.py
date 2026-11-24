"""
venue_registry.py
===============================================================================
Purpose
-------
Keep venue identity, historical product classification and capability evidence distinct.

What this module provides
-------------------------
Append-only registry validation, explicit historical seeds and conservative lookups.

Design Notes
------------
A request, brand, account assertion or contract record never authorizes execution.
No canonical classification or economic calculation consumes this registry by default.
===============================================================================
"""

from __future__ import annotations

import math

from oddsjam_ev.analysis.canonical_config import CanonicalConfig
from oddsjam_ev.analysis.sizing_config import aware
from oddsjam_ev.io.source_csv import fingerprint

KINDS = {"fixed_odds", "exchange", "pickem", "dfs", "unknown"}
OPERATIONS = {
    "manual_csv_import",
    "quotes_read",
    "orders_submit",
    "orders_cancel",
    "automated_export",
}
ARRAYS = ("venues", "venue_aliases", "products", "access", "requests")
SCOPE = {"effective_from", "effective_to", "open_ended", "scope_verified"}
CONTRACT_KEYS = {
    "evidence_state",
    "currency",
    "price_unit",
    "stake_unit",
    "payout_unit",
    "unit_payout",
    "fee_model",
    "fee_rate",
    "fee_basis",
    "settlement_model",
    "settlement_rules",
    "reference",
}


def _exact(value, keys):
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("Exact venue registry fields required")


def _text(value):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("Nonempty exact registry text required")


def _scope(row):
    if any(type(row[k]) is not bool for k in ("scope_verified", "open_ended")):
        raise ValueError("Explicit scope flags required")
    start, end = row["effective_from"], row["effective_to"]
    if start is None:
        if end is not None or row["open_ended"] or row["scope_verified"]:
            raise ValueError("Unknown effective scope must remain unverified")
    else:
        at = aware(start)
        if end is not None and aware(end) <= at:
            raise ValueError("Invalid effective scope")
        if row["open_ended"] and end is not None:
            raise ValueError("Open scope cannot have an end")
        if row["scope_verified"] and end is None and not row["open_ended"]:
            raise ValueError("Verified scope needs an explicit end or open flag")


def _overlap(a, b):
    # Missing bounds cannot establish disjoint ownership of an alias.
    return not (
        a["effective_to"] is not None
        and b["effective_from"] is not None
        and aware(a["effective_to"]) <= aware(b["effective_from"])
        or b["effective_to"] is not None
        and a["effective_from"] is not None
        and aware(b["effective_to"]) <= aware(a["effective_from"])
    )


def _active(row, at):
    return (
        at is not None
        and row["scope_verified"]
        and aware(row["effective_from"]) <= at
        and (row["open_ended"] or at < aware(row["effective_to"]))
    )


def unknown_contract() -> dict:
    """Return explicit unknown metadata, never zero-fee assumptions.

    Returns
    -------
    dict
        Contract placeholders that cannot establish economic readiness.
    """
    return dict(
        evidence_state="unknown",
        currency="unknown",
        price_unit="unknown",
        stake_unit="unknown",
        payout_unit="unknown",
        unit_payout=None,
        fee_model="unknown",
        fee_rate=None,
        fee_basis="unknown",
        settlement_model="unknown",
        settlement_rules=None,
        reference=None,
    )


def product_version(
    version_id: str,
    product_id: str,
    venue_id: str,
    source_id: str,
    product_type: str,
    aliases: list[str],
    recorded_at: str,
    reference: str,
) -> dict:
    """Create an unverified product version for explicit evidence completion.

    Parameters
    ----------
    version_id, product_id, venue_id, source_id : str
        Version identity, stable product/venue identities and exact source scope.
    product_type : str
        Declared product class; independent of brand and request wording.
    aliases : list[str]
        Exact source labels, not fuzzy brand stems.
    recorded_at, reference : str
        Aware receipt time and supporting evidence locator.

    Returns
    -------
    dict
        Product metadata with unknown contract and source capabilities.
    """
    return dict(
        version_id=version_id,
        product_id=product_id,
        venue_id=venue_id,
        source_id=source_id,
        product_type=product_type,
        aliases=list(aliases),
        effective_from=None,
        effective_to=None,
        open_ended=False,
        scope_verified=False,
        recorded_at=recorded_at,
        reference=reference,
        contract=unknown_contract(),
        operations={op: dict(support="unknown", permission="unknown") for op in sorted(OPERATIONS)},
    )


def validate_registry(registry: dict, previous: dict | None = None) -> str:
    """Validate explicit evidence and immutable prefix revisions.

    Parameters
    ----------
    registry : dict
        Version-1 identities, scoped product/access assertions and requests.
    previous : dict or None
        Prior ledger required for a revision; old records cannot be edited/deleted.

    Returns
    -------
    str
        Logical registry fingerprint, without mutating inputs.
    """
    _exact(
        registry,
        {
            "contract_version",
            "knowledge_as_of",
            "previous_registry_id",
            "source_policy_id",
            *ARRAYS,
        },
    )
    if type(registry["contract_version"]) is not int or registry["contract_version"] != 1:
        raise ValueError("Unsupported venue registry version")
    cutoff = aware(registry["knowledge_as_of"])
    if registry["source_policy_id"] is not None:
        _text(registry["source_policy_id"])
    if any(not isinstance(registry[k], list) for k in ARRAYS):
        raise ValueError("Registry arrays required")
    if previous is None:
        if registry["previous_registry_id"] is not None:
            raise ValueError("Registry predecessor required")
    elif (
        registry["previous_registry_id"] != fingerprint(previous)
        or cutoff < aware(previous["knowledge_as_of"])
        or registry["source_policy_id"] != previous["source_policy_id"]
        or any(registry[k][: len(previous[k])] != previous[k] for k in ARRAYS)
    ):
        raise ValueError("Registry revision must preserve append-only history and source")

    def receipt(row):
        _text(row["reference"])
        if aware(row["recorded_at"]) > cutoff:
            raise ValueError("Evidence postdates knowledge cutoff")

    venues = {}
    for v in registry["venues"]:
        _exact(v, {"venue_id", "name", "recorded_at", "reference"})
        _text(v["venue_id"])
        _text(v["name"])
        receipt(v)
        if v["venue_id"] in venues:
            raise ValueError("Duplicate venue ID")
        venues[v["venue_id"]] = v["name"]
    aliases = {}
    for v in registry["venue_aliases"]:
        _exact(v, {"venue_id", "alias", "recorded_at", "reference"})
        receipt(v)
        _text(v["alias"])
        if (
            v["venue_id"] not in venues
            or v["alias"] in aliases
            or (v["alias"] in venues and v["alias"] != v["venue_id"])
        ):
            raise ValueError("Unknown venue or alias collision")
        aliases[v["alias"]] = v["venue_id"]
    versions, owners = set(), {}
    for p in registry["products"]:
        _exact(
            p,
            {
                "version_id",
                "product_id",
                "venue_id",
                "source_id",
                "product_type",
                "aliases",
                "recorded_at",
                "reference",
                "contract",
                "operations",
                *SCOPE,
            },
        )
        receipt(p)
        _scope(p)
        for key in ("version_id", "product_id", "venue_id", "source_id"):
            _text(p[key])
        if (
            p["version_id"] in versions
            or p["venue_id"] not in venues
            or p["product_type"] not in KINDS
        ):
            raise ValueError("Invalid product identity/type")
        versions.add(p["version_id"])
        identity = (p["venue_id"], p["product_type"])
        if owners.setdefault(p["product_id"], identity) != identity:
            raise ValueError("Stable product ownership/type changed")
        if not isinstance(p["aliases"], list) or not p["aliases"]:
            raise ValueError("Product aliases required")
        for alias in p["aliases"]:
            _text(alias)
        if len(set(p["aliases"])) != len(p["aliases"]):
            raise ValueError("Duplicate product alias")
        c = p["contract"]
        _exact(c, CONTRACT_KEYS)
        choices = dict(
            evidence_state={"unknown", "confirmed", "unsupported"},
            currency={"unknown", "USD"},
            price_unit={
                "unknown",
                "american_odds",
                "decimal_odds",
                "usd_per_contract",
                "payout_multiplier",
            },
            stake_unit={"unknown", "usd", "contracts", "ticket_usd"},
            payout_unit={"unknown", "gross_usd", "usd_per_contract"},
            fee_model={"unknown", "none", "rate"},
            fee_basis={"unknown", "stake", "gross_profit", "payout"},
            settlement_model={
                "unknown",
                "fixed_odds_win_loss",
                "binary_contract",
                "ticket_paytable",
                "unsupported",
            },
        )
        if any(c[k] not in values for k, values in choices.items()):
            raise ValueError("Unsupported contract metadata value")
        for key in ("unit_payout", "fee_rate"):
            val = c[key]
            if val is not None and (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(val)
                or val < 0
                or key == "fee_rate"
                and val > 1
            ):
                raise ValueError("Invalid contract amount/rate")
        if c["fee_model"] != "rate" and c["fee_rate"] is not None:
            raise ValueError("Only explicit rate fees may have a rate")
        for key in ("reference", "settlement_rules"):
            if c[key] is not None:
                _text(c[key])
        if c["evidence_state"] == "confirmed" and c["reference"] is None:
            raise ValueError("Confirmed contract requires evidence reference")
        _exact(p["operations"], OPERATIONS)
        for op in p["operations"].values():
            _exact(op, {"support", "permission"})
            if op["support"] not in {"supported", "unsupported", "unknown"} or op[
                "permission"
            ] not in {"permitted", "denied", "unknown"}:
                raise ValueError("Invalid operation support/permission")
    for i, a in enumerate(registry["products"]):
        for b in registry["products"][i + 1 :]:
            if (
                a["venue_id"] == b["venue_id"]
                and a["source_id"] == b["source_id"]
                and set(a["aliases"]) & set(b["aliases"])
                and a["product_id"] != b["product_id"]
                and _overlap(a, b)
            ):
                raise ValueError("Product alias collision in source/effective scope")
    ids = set()
    for a in registry["access"]:
        _exact(
            a,
            {
                "evidence_id",
                "namespace",
                "product_id",
                "source_id",
                "state",
                "recorded_at",
                "reference",
                *SCOPE,
            },
        )
        receipt(a)
        _scope(a)
        for key in ("evidence_id", "namespace", "product_id", "source_id"):
            _text(a[key])
        if (
            a["evidence_id"] in ids
            or a["product_id"] not in owners
            or (a["product_id"], a["source_id"])
            not in {(p["product_id"], p["source_id"]) for p in registry["products"]}
            or a["state"] not in {"confirmed", "denied", "unknown"}
        ):
            raise ValueError("Invalid account-access evidence")
        ids.add(a["evidence_id"])
    ids = set()
    for r in registry["requests"]:
        _exact(
            r,
            {
                "request_id",
                "name",
                "venue_id",
                "product_type",
                "recorded_at",
                "reference",
                "status",
                "notes",
            },
        )
        receipt(r)
        for key in ("request_id", "name", "notes"):
            _text(r[key])
        if (
            r["request_id"] in ids
            or r["venue_id"] is not None
            and r["venue_id"] not in venues
            or r["product_type"] not in KINDS
            or r["status"] != "requested"
        ):
            raise ValueError("Invalid new-venue request")
        ids.add(r["request_id"])
    return fingerprint(registry)


def seed_from_policy(policy: dict, recorded_at: str) -> dict:
    """Adapt a pinned canonical policy into unverified historical metadata.

    Parameters
    ----------
    policy : dict
        Existing canonical configuration dictionary; never modified.
    recorded_at : str
        Explicit aware evidence receipt/knowledge time.

    Returns
    -------
    dict
        Registry with unknown economics/access and independent onboarding requests.
    """
    CanonicalConfig(**policy)
    result = dict(
        contract_version=1,
        knowledge_as_of=recorded_at,
        previous_registry_id=None,
        source_policy_id=fingerprint(policy),
        **{k: [] for k in ARRAYS},
    )
    reference = "canonical-policy:" + fingerprint(policy)
    # Explicit shared brand identity only; never strip labels to guess a brand/product.
    brands = {"DraftKings (Pick 3)": "DraftKings", "DraftKings (Pick 6)": "DraftKings"}
    known = {}
    for kind, labels in sorted(policy["products"].items()):
        for label in sorted(labels):
            name = brands.get(label, label)
            vid = "venue-" + fingerprint(name)[:16]
            if vid not in known:
                result["venues"].append(
                    dict(venue_id=vid, name=name, recorded_at=recorded_at, reference=reference)
                )
                result["venue_aliases"].append(
                    dict(venue_id=vid, alias=name, recorded_at=recorded_at, reference=reference)
                )
                known[vid] = name
            pid = "product-" + fingerprint([kind, label])[:16]
            p = product_version(
                "historical-" + pid, pid, vid, "oddsjam_csv", kind, [label], recorded_at, reference
            )
            p["operations"]["manual_csv_import"] = dict(support="supported", permission="permitted")
            result["products"].append(p)
    for name in ("Robinhood", "DraftKings", "FanDuel"):
        result["requests"].append(
            dict(
                request_id="request-" + name.lower(),
                name=name,
                venue_id=next((vid for vid, n in known.items() if n == name), None),
                product_type="unknown",
                recorded_at=recorded_at,
                reference="roadmap:R06",
                status="requested",
                notes="Evaluation request only; specify product, account access, source and contract evidence.",
            )
        )
    validate_registry(result)
    return result


def contract_blockers(product: dict) -> list[str]:
    """Report contract evidence gaps without calculating payouts or profit.

    Parameters
    ----------
    product : dict
        Validated scoped product version and explicit contract metadata.

    Returns
    -------
    list[str]
        Missing or unsupported contract requirements, not trading advice.
    """
    c = product["contract"]
    reasons = []
    if c["evidence_state"] != "confirmed":
        reasons.append("contract_unverified")
    if c["currency"] != "USD":
        reasons.append("currency_unknown")
    units = {
        "fixed_odds": (
            {"american_odds", "decimal_odds"},
            "usd",
            "gross_usd",
            "fixed_odds_win_loss",
        ),
        "exchange": ({"usd_per_contract"}, "contracts", "usd_per_contract", "binary_contract"),
    }
    u = units.get(product["product_type"])
    if u is None or c["settlement_model"] != u[3]:
        reasons.append("settlement_unsupported")
    if u is None or c["price_unit"] not in u[0] or (c["stake_unit"], c["payout_unit"]) != u[1:3]:
        reasons.append("units_unsupported")
    if c["settlement_rules"] is None:
        reasons.append("settlement_rules_missing")
    if product["product_type"] == "exchange" and (
        c["unit_payout"] is None or c["unit_payout"] <= 0
    ):
        reasons.append("unit_payout_missing")
    if (
        c["fee_model"] == "unknown"
        or c["fee_model"] == "rate"
        and (c["fee_rate"] is None or c["fee_basis"] == "unknown")
    ):
        reasons.append("fees_unknown")
    return reasons


def resolve_product(
    registry: dict,
    venue: str,
    *,
    source_id: str,
    version_id: str | None = None,
    label: str | None = None,
    product_type: str | None = None,
    at: str | None = None,
    namespace: str | None = None,
    operation: str = "manual_csv_import",
) -> dict:
    """Resolve a scoped product conservatively and explain readiness gates.

    Parameters
    ----------
    registry : dict
        Validated publication ledger (detached revisions accepted for read-only use).
    venue, source_id : str
        Exact stable venue ID/alias and source; no fuzzy matching.
    version_id, label, product_type : str or None
        Optional explicit version pin or exact label/type; never latest-wins.
    at : str or None
        Independent aware effective time; absent time cannot establish scope.
    namespace : str or None
        Explicit account namespace for accessible-account evidence.
    operation : str
        Requested source operation; support and permission are independent.

    Returns
    -------
    dict
        Candidate versions, declared type, blockers and evidence readiness only.
    """
    validate_registry({**registry, "previous_registry_id": None})
    if operation not in OPERATIONS or product_type is not None and product_type not in KINDS:
        raise ValueError("Unsupported query operation/product")
    instant = aware(at) if at is not None else None
    vid = next((v["venue_id"] for v in registry["venues"] if v["venue_id"] == venue), None)
    if vid is None:
        vid = next((a["venue_id"] for a in registry["venue_aliases"] if a["alias"] == venue), None)
    candidates = [
        p
        for p in registry["products"]
        if vid is not None
        and p["venue_id"] == vid
        and p["source_id"] == source_id
        and (version_id is None or p["version_id"] == version_id)
        and (label is None or label in p["aliases"])
        and (product_type is None or p["product_type"] == product_type)
    ]
    # Exclude only proven out-of-scope versions; unknown dates still prevent guessing.
    if instant is not None:
        candidates = [p for p in candidates if not p["scope_verified"] or _active(p, instant)]
    reasons = []
    if not candidates:
        reasons.append("unknown_product")
    elif len(candidates) != 1:
        reasons.append("ambiguous_product")
    product = candidates[0] if len(candidates) == 1 else None
    econ = False
    if product is not None:
        if not _active(product, instant):
            reasons.append("effective_scope_unverified")
        gaps = contract_blockers(product)
        reasons.extend(gaps)
        econ = _active(product, instant) and not gaps
        access = [
            a
            for a in registry["access"]
            if a["namespace"] == namespace
            and a["product_id"] == product["product_id"]
            and a["source_id"] == source_id
            and (not a["scope_verified"] or _active(a, instant))
        ]
        if len(access) != 1 or access[0]["state"] != "confirmed" or not _active(access[0], instant):
            reasons.append("access_unverified_or_conflicting")
        op = product["operations"][operation]
        if op["support"] != "supported":
            reasons.append("operation_unsupported_or_unknown")
        if op["permission"] != "permitted":
            reasons.append("operation_not_permitted")
    return dict(
        status="unknown" if not candidates else "ambiguous" if product is None else "resolved",
        candidate_version_ids=sorted(p["version_id"] for p in candidates),
        product_id=product["product_id"] if product else None,
        product_type=product["product_type"] if product else "unknown",
        economic_contract_ready=econ,
        evidence_ready=not reasons,
        blockers=reasons,
        request_ids=[
            r["request_id"]
            for r in registry["requests"]
            if r["name"] == venue or vid is not None and r["venue_id"] == vid
        ],
        execution_authorized=False,
    )
