"""Receipt recency, export recency, scope and novelty are independent evidence."""

from oddsjam_ev.analysis.experiment_config import aware


def assess_freshness(imports: list[dict], as_of: str, max_age_days: int) -> dict:
    """Assess every selected import conservatively, never using filename or bet time.

    A snapshot may include several overlapping exports of unknown completeness.
    No recent import is allowed to conceal undated or old selected sources.
    """
    cutoff = aware(as_of)
    rows = []
    for item in imports:
        received = aware(item["imported_at"])
        if received > cutoff:
            raise ValueError("Source knowledge exceeds cutoff")
        exported = item["source"].get("exported_at")
        age = None if exported is None else (cutoff - aware(exported)).total_seconds() / 86400
        if age is not None and (age < 0 or aware(exported) > received):
            raise ValueError("Export time exceeds receipt or cutoff")
        rows.append(
            {
                "import_id": item["import_id"],
                "receipt_age_days": (cutoff - received).total_seconds() / 86400,
                "export_age_days": age,
                "export_status": "unverified"
                if age is None
                else ("stale" if age > max_age_days else "recent"),
                "scope": item["source"].get("scope", "unknown"),
                "timestamp_timezone": item["source"].get("timestamp_timezone", "unknown"),
            }
        )
    statuses = {row["export_status"] for row in rows}
    status = (
        "missing"
        if not rows
        else (
            "stale"
            if "stale" in statuses
            else "unverified"
            if "unverified" in statuses
            else "recent"
        )
    )
    return {
        "export_status": status,
        "scope_verified": bool(rows) and all(r["scope"] not in {"unknown", "", None} for r in rows),
        "complete_coverage_verified": False,
        "max_export_age_days": max_age_days,
        "sources": rows,
    }
