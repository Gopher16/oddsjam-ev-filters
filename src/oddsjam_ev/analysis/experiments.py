"""
experiments.py
===============================================================================
Purpose
-------
Replay immutable experiment state and separate intent from launch evidence.

What this module provides
-------------------------
Experiment lifecycle and enrollment-time classification.

Design Notes
------------
Registration, launch, enrollment and observed source funding are separate facts.
===============================================================================
"""

from oddsjam_ev.analysis.experiment_config import aware

STATES = {
    "REGISTERED": {"RUNNING", "CANCELLED"},
    "RUNNING": {"PAUSED", "AWAITING_OUTCOMES"},
    "PAUSED": {"RUNNING", "AWAITING_OUTCOMES"},
    "AWAITING_OUTCOMES": {"READY_FOR_REVIEW"},
    "READY_FOR_REVIEW": {"CLOSED"},
    "CLOSED": set(),
    "CANCELLED": set(),
}


def replay_experiments(events: list[dict]) -> dict:
    """Reconstruct experiment state without mutating protocol history.

    Parameters
    ----------
    events : list[dict]
        Integrity-checked append-only experiment events.

    Returns
    -------
    dict
        Registered protocols, lifecycle, enrollments, checks and link decisions.
    """
    state = {}
    for event in events:
        ident = event["experiment_id"]
        kind = event["kind"]
        payload = event["payload"]
        if kind == "REGISTER":
            if ident in state:
                raise ValueError("Duplicate experiment registration")
            state[ident] = {
                "protocol": payload["protocol"],
                "registration": event,
                "status": "REGISTERED",
                "launch": None,
                "enrollments": {},
                "links": {},
                "checks": {},
                "events": [],
            }
        elif ident not in state:
            raise ValueError("Unknown experiment")
        item = state[ident]
        if kind == "TRANSITION":
            if payload["to_status"] not in STATES.get(item["status"], set()):
                raise ValueError("Invalid experiment transition history")
            item["status"] = payload["to_status"]
            if item["status"] == "RUNNING" and item["launch"] is None:
                item["launch"] = event
        elif kind == "ENROLL":
            if payload["enrollment_id"] in item["enrollments"]:
                raise ValueError("Duplicate enrollment")
            item["enrollments"][payload["enrollment_id"]] = event
        elif kind == "LINK":
            item["links"][payload["enrollment_id"]] = event
        elif kind == "CHECK":
            item["checks"].setdefault(payload["enrollment_id"], []).append(event)
        elif kind == "DEVIATION":
            if item["status"] == "RUNNING":
                item["status"] = "PAUSED"
        elif kind != "REGISTER":
            raise ValueError("Unknown experiment event kind")
        item["events"].append(event)
    return state


def enrollment_timing(recorded_at, launch_at, event_start, observed_at) -> str:
    """Distinguish timely enrollment from later or unsupported observations.

    Parameters
    ----------
    recorded_at, launch_at, event_start, observed_at : str
        Explicit times for enrollment, launch, event and scanner capture.

    Returns
    -------
    str
        prospective, retrospective, or timing_unverified. Naive timestamps are rejected.
    """
    if event_start is None or observed_at is None:
        return "timing_unverified"
    recorded, launch, start, observed = map(
        aware, [recorded_at, launch_at, event_start, observed_at]
    )
    return "prospective" if launch <= observed <= recorded < start else "retrospective"
