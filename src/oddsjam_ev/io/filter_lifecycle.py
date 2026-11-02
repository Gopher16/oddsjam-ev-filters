"""
filter_lifecycle.py
===============================================================================
Purpose
-------
Record reviewed outcomes and external configuration confirmations without deployment.

What this module provides
-------------------------
Append-only policy/decision/change/application/monitoring evidence and immutable reports.

Design Notes
------------
Production requires a predeclared prospective policy and explicit bounded-pilot review.
No state transition executes a scanner action, places a wager or infers a fill.
===============================================================================
"""

import hashlib
import json
import os
import tempfile
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

from oddsjam_ev.analysis.experiment_config import aware
from oddsjam_ev.analysis.experiments import replay_experiments
from oddsjam_ev.analysis.filter_lifecycle import (
    ACTIONS,
    FEATURES,
    finite,
    monitoring_tables,
    nonempty,
    replay_lifecycle,
    setting_diff,
    settings_snapshot,
    validate_policy,
)
from oddsjam_ev.io.experiment_register import read_experiment_events
from oddsjam_ev.io.experiment_reports import load_experiment_report
from oddsjam_ev.io.manifests import verify_bytes, write_json, writer_lock
from oddsjam_ev.io.paper_definitions import load_definition_publication
from oddsjam_ev.io.source_csv import fingerprint


def utc_now():
    """Return the recording clock; the CLI cannot backdate journal knowledge."""
    return datetime.now(UTC).isoformat()


def read_lifecycle(root: Path, as_of: str | None = None) -> list[dict]:
    """Verify the complete chain before applying a knowledge cutoff.

    Parameters
    ----------
    root : pathlib.Path
        Private append-only journal.
    as_of : str, optional
        Aware recording cutoff. An absent register has no recorded evidence.

    Returns
    -------
    list[dict]
        Verified events. Tail deletion needs an external head anchor to detect.
    """
    events, last, stamp = [], None, None
    for number, path in enumerate(sorted(root.glob("*.json")), 1):
        e = json.loads(path.read_text())
        body = {k: v for k, v in e.items() if k != "event_id"}
        current = aware(e["recorded_at"])
        if (
            e.get("contract_version") != 1
            or e.get("sequence") != number
            or path.name != f"{number:08d}.json"
            or e.get("previous_event_id") != last
            or fingerprint(body) != e.get("event_id")
            or (stamp is not None and current < stamp)
            or e.get("external_execution_performed") is not False
            or not nonempty(e.get("actor"))
            or not nonempty(e.get("rationale"))
        ):
            raise ValueError("Filter lifecycle register integrity failure")
        events.append(e)
        last, stamp = e["event_id"], current
    replay_lifecycle(events)
    return [e for e in events if aware(e["recorded_at"]) <= aware(as_of)] if as_of else events


def _required(p, fields):
    if not set(fields) <= set(p) or any(not nonempty(p[k]) for k in fields):
        raise ValueError("Explicit identity and evidence fields required: " + ", ".join(fields))


def _definition(path, version_id, now):
    snapshot = load_definition_publication(Path(path))
    matches = [e for e in snapshot["events"] if e["version_id"] == version_id]
    if len(matches) != 1 or aware(snapshot["manifest"]["specification"]["as_of"]) > aware(now):
        raise ValueError("Known exact follow-up definition required")
    definition = matches[0]
    latest = [e for e in snapshot["events"] if e["test_key"] == definition["test_key"]][-1]
    if latest["version_id"] != version_id or aware(
        definition["protocol"]["planned_start_utc"]
    ) <= aware(now):
        raise ValueError("Follow-up must be a current future definition version")
    return {
        "definition": definition,
        "publication_id": snapshot["manifest"]["publication_id"],
        "manifest_sha256": hashlib.sha256((Path(path) / "manifest.json").read_bytes()).hexdigest(),
    }


def _review_source(p, now):
    root, path = Path(p["experiment_register"]), Path(p["review_report"])
    live = replay_experiments(read_experiment_events(root))
    item = live.get(p["experiment_id"])
    if not item or item["protocol"].get("contract_version") != 2:
        raise ValueError("Definition-bound experiment required")
    tables = load_experiment_report(path)
    manifest = json.loads((path / "manifest.json").read_text())
    pinned = [
        e for e in manifest["specification"]["events"] if e["experiment_id"] == p["experiment_id"]
    ]
    if (
        not pinned
        or pinned != item["events"]
        or aware(manifest["specification"]["as_of"]) > aware(now)
    ):
        raise ValueError("Current exact reviewed experiment report required")
    reviews = [e for e in item["reviews"] if e["event_id"] == p["review_event_id"]]
    if len(reviews) != 1:
        raise ValueError("Explicit recorded scheduled review required")
    d = item["protocol"]["definition"]
    assessment = tables["prospective_reviews"].loc[
        lambda f: f.experiment_id.eq(p["experiment_id"])
        & f.scheduled_review_utc.eq(reviews[0]["payload"]["scheduled_review_utc"])
    ]
    if (
        len(assessment) != 1
        or not assessment.iloc[0].review_recorded
        or assessment.iloc[0].definition_version_id != d["version_id"]
        or assessment.iloc[0].review_event_id != p["review_event_id"]
    ):
        raise ValueError("Report must include the exact completed scheduled review")
    row = assessment.iloc[0]
    review_index = next(
        i for i, e in enumerate(item["events"]) if e["event_id"] == p["review_event_id"]
    )
    if any(
        e["kind"] != "TRANSITION"
        or e["payload"].get("to_status") not in {"READY_FOR_REVIEW", "CLOSED"}
        for e in item["events"][review_index + 1 :]
    ):
        raise ValueError(
            "Evidence changed after scheduled review; explicit review amendment required"
        )
    return {
        "experiment_id": p["experiment_id"],
        "definition_version_id": d["version_id"],
        "definition": d,
        "experiment_register": str(root.resolve()),
        "experiment_head": item["events"][-1]["event_id"],
        "review_event_id": p["review_event_id"],
        "review_report_id": manifest["report_id"],
        "review_manifest_sha256": hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest(),
        "readiness": row.readiness,
        "blockers": json.loads(row.blockers_json),
        "review_scheduled_utc": row.scheduled_review_utc,
    }


def _current_decision(state, decision_id):
    decision = state["decisions"].get(decision_id)
    if not decision:
        raise ValueError("Reviewed decision required")
    p = decision["payload"]
    matching = [
        d
        for d in state["decisions"].values()
        if d["payload"]["source"]["experiment_id"] == p["source"]["experiment_id"]
    ]
    if matching[-1]["event_id"] != decision_id:
        raise ValueError("Decision was superseded")
    live = replay_experiments(read_experiment_events(Path(p["source"]["experiment_register"])))
    if (
        live[p["source"]["experiment_id"]]["events"][-1]["event_id"]
        != p["source"]["experiment_head"]
    ):
        raise ValueError("Experiment evidence changed; new reviewed decision required")
    return decision


def _production_blockers(state, decision, change, now):
    """Keep a corrected result separate from permission to review/apply a bounded pilot."""
    from oddsjam_ev.analysis.promotion import CAPS
    from oddsjam_ev.io.decision_records import load_decisions

    reasons = []
    policy = state["policies"][decision["policy_event_id"]]["payload"]
    if policy["scope"] != "bounded_prod_pilot_review":
        return ["production_policy_not_approved"]
    assessment = state["evaluations"].get(change.get("promotion_assessment_event_id"))
    if not assessment or assessment["payload"]["policy_event_id"] != decision["policy_event_id"]:
        return ["exact_terminal_promotion_assessment_required"]
    source = decision["source"]
    qualified = [
        r
        for r in assessment["payload"]["assessment"]["results"]
        if r["definition_version_id"] == source["definition_version_id"]
    ]
    if len(qualified) != 1 or not qualified[0]["eligible_for_human_pilot_review"]:
        reasons.append("prospective_corrected_family_did_not_qualify")
    for anchor in assessment["payload"]["source_anchors"]:
        live = replay_experiments(read_experiment_events(Path(anchor["experiment_register"])))
        history = live[anchor["experiment_id"]]["events"]
        exact = history[-1]["event_id"] == anchor["experiment_head"]
        closure = (
            len(history) >= 2
            and history[-1]["kind"] == "TRANSITION"
            and history[-1]["payload"].get("to_status") == "CLOSED"
            and history[-2]["event_id"] == anchor["experiment_head"]
        )
        if not exact and not closure:
            reasons.append("complete_family_evidence_changed_after_terminal_look")
    reference = source["definition"]["protocol"].get("configuration_reference")
    path = change.get("configuration_publication")
    if not reference or not nonempty(path):
        reasons.append("tested_complete_scanner_configuration_required")
    else:
        publication = load_decisions(Path(path))
        records = [
            r
            for r in publication["records"]
            if r["decision_id"] == reference["decision_id"]
            and r["record_sha256"] == reference["record_sha256"]
        ]
        if (
            publication["manifest"]["publication_id"] != reference["publication_id"]
            or len(records) != 1
        ):
            raise ValueError("Exact tested configuration publication required")
        record = records[0]
        values = {k: v.get("value") for k, v in change["after_settings"].items()}
        final = {r["control"]: r["final_value"] for r in record["settings"]}
        if (
            record["setup_status"] != "ready_for_operator_configuration_review"
            or record["candidate_id"] != source["definition"]["source"]["candidate_id"]
            or values != final
        ):
            reasons.append("proposed_settings_differ_from_tested_verified_trial")
    if not nonempty(change.get("pilot_budget_attestation")):
        reasons.append("explicit_pilot_budget_and_duration_attestation_required")
    for app_id in state["current"].values():
        app = state["applications"][app_id]["payload"]
        if app.get("pilot_caps") and app["status"] == "PROD":
            reasons.append("another_confirmed_pilot_requires_explicit_retirement")
    for app in state["applications"].values():
        if app["payload"].get("pilot_caps"):
            prior_change = state["changes"][app["payload"]["change_event_id"]]["event"]["payload"]
            prior_decision = state["decisions"][prior_change["decision_event_id"]]["payload"]
            if prior_decision["source"]["definition_version_id"] == source["definition_version_id"]:
                reasons.append(
                    "definition_already_used_for_a_pilot_new_prospective_evidence_required"
                )
    deployed = sum(
        e["payload"]["stake_usd"]
        for e in state["observations"].values()
        if e["payload"]["execution_mode"] == "operator_reported_real_money"
        and state["applications"][e["payload"]["application_id"]]["payload"].get("pilot_caps")
    )
    if deployed >= CAPS["max_total_deployed_stake_usd"]:
        reasons.append("program_pilot_budget_exhausted")
    return sorted(set(reasons))


def _prepare(state, kind, p, now):
    p = json.loads(json.dumps(p, allow_nan=False))
    if kind == "POLICY":
        validate_policy(p)
        if p["scope"] == "bounded_prod_pilot_review":
            prior_versions = {
                r["version_id"]
                for e in state["policies"].values()
                if e["payload"]["scope"] == "bounded_prod_pilot_review"
                for r in e["payload"]["promotion"]["family_definitions"]
            }
            if prior_versions & {r["version_id"] for r in p["promotion"]["family_definitions"]}:
                raise ValueError(
                    "A definition version cannot be redeclared in another promotion family"
                )
            p["promotion"]["family"] = [
                _definition(r["publication"], r["version_id"], now)
                for r in p["promotion"]["family_definitions"]
            ]
            for row, d in zip(
                p["promotion"]["family_definitions"], p["promotion"]["family"], strict=True
            ):
                expected_arm = (
                    "subgroup"
                    if d["definition"]["source"]["trial_arm"] == "subgroup"
                    else "complement"
                )
                if row["trial_arm"] != expected_arm:
                    raise ValueError("Predeclared trial arm differs from frozen definition")
        versions = [
            e["payload"]["version"]
            for e in state["policies"].values()
            if e["payload"]["policy_key"] == p["policy_key"]
        ]
        if p["version"] != (max(versions) + 1 if versions else 1):
            raise ValueError("Policy versions must advance without replacement")
    elif kind == "EVALUATE_PROMOTION":
        import pandas as pd

        from oddsjam_ev.analysis.promotion import evaluate_family

        policy_event = state["policies"].get(p.get("policy_event_id"))
        if not policy_event or policy_event["payload"]["scope"] != "bounded_prod_pilot_review":
            raise ValueError("Declared prospective production review policy required")
        if any(
            e["payload"]["policy_event_id"] == p["policy_event_id"]
            for e in state["evaluations"].values()
        ):
            raise ValueError("Fixed terminal look already evaluated; no repeated optional looks")
        policy = policy_event["payload"]["promotion"]
        sources = [_review_source(row, now) for row in p.get("sources", [])]
        if (
            len(sources) != len(policy["family"])
            or len({s["definition_version_id"] for s in sources}) != len(sources)
            or {s["definition_version_id"] for s in sources}
            != {d["definition"]["version_id"] for d in policy["family"]}
        ):
            raise ValueError("Complete exact predeclared prospective family required")
        populations, blockers = {}, {}
        for raw, source in zip(p["sources"], sources, strict=True):
            ident = source["definition_version_id"]
            definition = source["definition"]
            declared = next(
                d["definition"] for d in policy["family"] if d["definition"]["version_id"] == ident
            )
            if definition != declared:
                raise ValueError("Prospective definition identity changed")
            item = replay_experiments(read_experiment_events(Path(raw["experiment_register"])))[
                raw["experiment_id"]
            ]
            if not item["launch"] or aware(policy_event["recorded_at"]) >= aware(
                item["launch"]["recorded_at"]
            ):
                raise ValueError("Promotion policy must be registered before actual launch")
            plan = definition["protocol"]
            terminal = aware(plan["planned_end_utc"]) + pd.Timedelta(
                days=plan["settlement_followup_days"]
            )
            manifest = json.loads((Path(raw["review_report"]) / "manifest.json").read_text())
            if aware(manifest["specification"]["as_of"]) < terminal:
                raise ValueError("Only the fixed terminal review after follow-up can be evaluated")
            populations[ident] = (
                load_experiment_report(Path(raw["review_report"]))["prospective_observations"]
                .loc[lambda f, ident=ident: f.definition_version_id.eq(ident)]
                .copy()
            )
            blockers[ident] = list(source["blockers"])
            if any(
                e["kind"] == "SETTLE" and aware(e["recorded_at"]) > terminal for e in item["events"]
            ):
                blockers[ident].append("settlement_recorded_after_fixed_terminal_cutoff")
        p.update(
            source_anchors=sources,
            assessment=evaluate_family(policy, populations, blockers),
            external_execution_performed=False,
        )
    elif kind == "DECISION":
        _required(
            p,
            [
                "policy_event_id",
                "experiment_register",
                "experiment_id",
                "review_report",
                "review_event_id",
                "evidence_reference",
            ],
        )
        if p.get("action") not in ACTIONS or p["policy_event_id"] not in state["policies"]:
            raise ValueError("Declared review policy and explicit human action required")
        source = _review_source(p, now)
        prior = [
            d
            for d in state["decisions"].values()
            if d["payload"]["source"]["experiment_id"] == p["experiment_id"]
        ]
        if p.get("supersedes_decision_id") != (prior[-1]["event_id"] if prior else None):
            raise ValueError("Explicit latest decision supersession required")
        followup = None
        if p["action"] in {"continue", "modify"}:
            followup = _definition(p.get("followup_publication"), p.get("followup_version_id"), now)
            if followup["definition"]["version_id"] == source["definition_version_id"]:
                raise ValueError(
                    "Continue/modify requires a new definition, never reopening the old version"
                )
        p.update(
            source=source,
            followup=followup,
            production_authorized=False,
            interpretation="human_descriptive_review_not_statistical_pass",
        )
    elif kind == "BASELINE":
        _required(p, ["filter_id", "policy_event_id", "evidence_reference", "legacy_context"])
        if (
            p["filter_id"] in state["current"]
            or p["policy_event_id"] not in state["policies"]
            or p.get("status") not in {"TEST", "PROD", "retired"}
        ):
            raise ValueError("One initial operator-confirmed baseline and declared policy required")
        if aware(p.get("effective_at_utc")) > aware(now):
            raise ValueError("Baseline application cannot be future-dated")
        p["settings"] = settings_snapshot(p.get("settings"), p["evidence_reference"], now)
        p.update(previous_application_id=None, production_approval="legacy_unknown_not_a_promotion")
    elif kind == "CHANGE":
        _required(p, ["filter_id", "decision_event_id", "evidence_reference"])
        decision = _current_decision(state, p["decision_event_id"])["payload"]
        prior_id = state["current"].get(p["filter_id"])
        if p.get("previous_application_id") != prior_id:
            raise ValueError("Exact current applied baseline required")
        prior = state["applications"][prior_id]["payload"] if prior_id else None
        action = p.get("change_kind")
        if action not in {"new_test", "modify", "stop", "rollback", "promote"} or p.get(
            "target_status"
        ) not in {"TEST", "PROD", "retired"}:
            raise ValueError("Explicit supported change kind/status required")
        if action == "new_test" and (prior or p["target_status"] != "TEST"):
            raise ValueError("A new paper test needs an unused filter identity and TEST status")
        if action not in {"new_test", "promote"} and not prior:
            raise ValueError("Existing applied version required for modification/stop/rollback")
        if action == "stop" and (decision["action"] != "stop" or p["target_status"] != "retired"):
            raise ValueError("Stopping needs an explicit stop decision and retired target")
        if action in {"new_test", "modify", "promote"} and decision["action"] not in {
            "continue",
            "modify",
        }:
            raise ValueError("Settings changes require a reviewed continue/modify decision")
        before = prior["settings"] if prior else {}
        after = settings_snapshot(p.get("after_settings"), p["evidence_reference"], now)
        blockers = []
        touches_production = p["target_status"] == "PROD" or (prior and prior["status"] == "PROD")
        policy = state["policies"][decision["policy_event_id"]]["payload"]
        if touches_production:
            if policy["scope"] != "bounded_prod_pilot_review":
                blockers.append("production_policy_not_approved")
            elif p["target_status"] != "retired":
                blockers.extend(_production_blockers(state, decision, p, now))
        if any(s["state"] == "unknown" for s in after.values()):
            blockers.append("incomplete_target_settings")
        if action == "stop" and after != before:
            # Evidence timestamps may differ; semantic configuration must be identical.
            if setting_diff(before, after):
                raise ValueError("Stop cannot hide settings changes")
        diff = setting_diff(before, after)
        translations = p.get("ui_evidence", [])
        if not isinstance(translations, list):
            raise ValueError("Scanner translation evidence must be a list")
        for t in translations:
            if (
                set(t)
                != {
                    "control",
                    "semantics_verified",
                    "platform_version",
                    "evidence_reference",
                    "observed_at_utc",
                    "state",
                    "value",
                }
                or t["control"] not in after
                or t["semantics_verified"] is not True
                or t["state"] != after[t["control"]]["state"]
                or t["value"] != after[t["control"]]["value"]
                or not nonempty(t["platform_version"])
                or not nonempty(t["evidence_reference"])
                or aware(t["observed_at_utc"]) > aware(now)
            ):
                raise ValueError("Exact supported scanner UI evidence required")
        if action != "stop" and any(
            sum(t["control"] == r["control"] for t in translations) != 1 for r in diff
        ):
            blockers.append("exact_ui_translation_evidence_missing_or_ambiguous")
        if action in {"new_test", "modify", "promote"}:
            followup = decision["followup"]
            values = {k: v["value"] for k, v in after.items()}
            if (
                not followup
                or followup["definition"]["protocol"]["baseline"]["filter_config"] != values
            ):
                blockers.append("followup_definition_settings_mismatch_or_incomplete")
            elif aware(followup["definition"]["protocol"]["planned_start_utc"]) <= aware(now):
                blockers.append("followup_start_elapsed_requires_new_definition")
        if p["target_status"] == "TEST":
            stake = state["policies"][decision["policy_event_id"]]["payload"]["monitoring"][
                "paper_stake_usd"
            ]
            sizing = after["paper_sizing"]
            if sizing["state"] != "observed" or sizing["value"] != {
                "method": "fixed_paper_stake",
                "stake_usd": stake,
                "applies_to": "both_arms",
            }:
                blockers.append("paper_sizing_monitoring_policy_mismatch")
        if action == "rollback":
            target = state["applications"].get(p.get("rollback_application_id"))
            if (
                not target
                or target["payload"]["filter_id"] != p["filter_id"]
                or setting_diff(target["payload"]["settings"], after)
            ):
                raise ValueError("Rollback must name an exact earlier configuration of this filter")
            if target["event_id"] == prior_id:
                raise ValueError("Rollback target is already current")
        p.update(
            policy_event_id=decision["policy_event_id"],
            before_settings=before,
            after_settings=after,
            diff=diff,
            approval_blockers=sorted(set(blockers)),
        )
        aware(p.get("planned_effective_at_utc"))
        if aware(p["planned_effective_at_utc"]) < aware(now):
            raise ValueError("Planned change date cannot precede proposal recording")
    elif kind in {"APPROVE_CHANGE", "REJECT_CHANGE", "CONFIRM_APPLICATION"}:
        _required(p, ["change_event_id", "evidence_reference"])
        change = state["changes"].get(p["change_event_id"])
        if not change:
            raise ValueError("Unknown settings change")
        target = change["event"]["payload"]
        if kind != "REJECT_CHANGE":
            current_decision = _current_decision(state, target["decision_event_id"])["payload"]
            if target["target_status"] == "PROD":
                reasons = _production_blockers(state, current_decision, target, now)
                if reasons:
                    raise ValueError("Blocked production review: " + ", ".join(reasons))
        if state["current"].get(target["filter_id"]) != target["previous_application_id"]:
            raise ValueError("Applied baseline changed since proposal")
        if kind == "APPROVE_CHANGE":
            if change["status"] != "proposed" or target["approval_blockers"]:
                raise ValueError("Blocked or already reviewed change cannot be approved")
            followup = state["decisions"][target["decision_event_id"]]["payload"]["followup"]
            if target["change_kind"] in {"new_test", "modify", "promote"} and aware(
                followup["definition"]["protocol"]["planned_start_utc"]
            ) <= aware(now):
                raise ValueError("Follow-up start elapsed; new definition and review required")
        if kind == "CONFIRM_APPLICATION":
            if (
                change["status"] != "approved"
                or p.get("approval_event_id") != change["approval"]["event_id"]
            ):
                raise ValueError("Exact current approval required for confirmed application")
            effective = aware(p.get("effective_at_utc"))
            if not aware(change["approval"]["recorded_at"]) <= effective <= aware(
                now
            ) or effective < aware(target["planned_effective_at_utc"]):
                raise ValueError(
                    "Actual application must follow approval/planned date and cannot be future-dated"
                )
            if (
                p.get("status") != target["target_status"]
                or p.get("filter_id") != target["filter_id"]
            ):
                raise ValueError("Explicit actual filter identity and status must match approval")
            actual = settings_snapshot(p.get("settings"), p["evidence_reference"], now)
            if setting_diff(target["after_settings"], actual):
                raise ValueError("Confirmed scanner settings differ from approval")
            p.update(
                filter_id=target["filter_id"],
                settings=target["after_settings"],
                status=target["target_status"],
                previous_application_id=target["previous_application_id"],
                policy_event_id=target["policy_event_id"],
            )
            if target["target_status"] == "PROD":
                p["pilot_caps"] = state["policies"][target["policy_event_id"]]["payload"][
                    "promotion"
                ]["pilot_caps"]
                p["production_approval"] = "explicit_human_bounded_pilot_only"
            prior = state["applications"].get(target["previous_application_id"])
            if prior and effective <= aware(prior["payload"]["effective_at_utc"]):
                raise ValueError("Application periods must advance strictly")
    elif kind == "OBSERVE":
        _required(
            p,
            [
                "application_id",
                "observation_key",
                "event_namespace",
                "event_id",
                "evidence_reference",
            ],
        )
        application = state["applications"].get(p["application_id"])
        if (
            not application
            or state["current"].get(application["payload"]["filter_id"]) != p["application_id"]
            or application["payload"]["status"] == "retired"
        ):
            raise ValueError("Current active applied configuration required")
        a = application["payload"]
        if not aware(application["recorded_at"]) <= aware(p.get("observed_at_utc")) <= aware(
            now
        ) < aware(p.get("event_start_utc")) or aware(p["observed_at_utc"]) < aware(
            a["effective_at_utc"]
        ):
            raise ValueError("Monitoring cannot retrospectively enroll observations")
        if p.get("configuration_sha256") != fingerprint(a["settings"]):
            raise ValueError("Exact applied configuration identity required")
        if any(
            e["payload"]["observation_key"] == p["observation_key"]
            for e in state["observations"].values()
        ):
            raise ValueError("Duplicate monitoring observation across application versions")
        if (
            p.get("execution_mode") not in {"paper", "operator_reported_real_money"}
            or not finite(p.get("american_odds"))
            or abs(p["american_odds"]) < 100
            or not finite(p.get("stake_usd"))
            or p["stake_usd"] <= 0
        ):
            raise ValueError("Explicit execution mode, positive stake and American odds required")
        if p["execution_mode"] == "operator_reported_real_money" and (
            a["status"] != "PROD" or not nonempty(p.get("fill_reference"))
        ):
            raise ValueError(
                "Only confirmed existing PROD with operator fill evidence can record funded observations"
            )
        if (
            p["execution_mode"] == "paper"
            and p["stake_usd"]
            != state["policies"][a["policy_event_id"]]["payload"]["monitoring"]["paper_stake_usd"]
        ):
            raise ValueError("Paper stake differs from declared monitoring policy")
        if p.get("availability_status") not in {
            "not_checked",
            "checked_available",
            "checked_unavailable",
            "inconclusive",
        }:
            raise ValueError("Explicit availability, including missing/failed checks required")
        if not nonempty(p.get("selection")) or p.get("line") is None:
            raise ValueError("Original selection and line required")
        if p["availability_status"] == "checked_available":
            check = p.get("availability_check", {})
            if (
                not isinstance(check, dict)
                or not nonempty(check.get("evidence_reference"))
                or not aware(p["observed_at_utc"])
                <= aware(check.get("checked_at_utc"))
                <= aware(now)
                or check.get("selection") != p["selection"]
                or str(check.get("line")) != str(p["line"])
                or check.get("american_odds") != p["american_odds"]
                or not finite(check.get("available_stake_usd"))
                or check["available_stake_usd"] < p["stake_usd"]
            ):
                raise ValueError("Timely unchanged quote and sufficient capacity evidence required")
        elif p["availability_status"] != "not_checked" and not nonempty(
            p.get("availability_reason")
        ):
            raise ValueError("Failed or inconclusive availability reason required")
        if "pilot_caps" in a:
            import pandas as pd

            from oddsjam_ev.analysis.promotion import CAPS

            deployed = sum(
                e["payload"]["stake_usd"]
                for e in state["observations"].values()
                if e["payload"]["execution_mode"] == "operator_reported_real_money"
                and "pilot_caps" in state["applications"][e["payload"]["application_id"]]["payload"]
            )
            breaches = []
            if p["execution_mode"] == "operator_reported_real_money":
                if p["stake_usd"] > CAPS["max_stake_per_wager_usd"]:
                    breaches.append("stake_per_wager_cap_exceeded")
                if deployed + p["stake_usd"] > CAPS["max_total_deployed_stake_usd"]:
                    breaches.append("program_deployed_stake_cap_exceeded")
                if aware(p["observed_at_utc"]) >= aware(a["effective_at_utc"]) + pd.Timedelta(
                    days=CAPS["max_duration_days"]
                ):
                    breaches.append("pilot_duration_cap_exceeded")
            p["pilot_breaches"] = breaches
        features = p.get("features", {})
        if (
            not isinstance(features, dict)
            or set(features) - FEATURES
            or any(not finite(v) for v in features.values())
        ):
            raise ValueError("Supported finite decision-time monitoring features required")
        features = {
            **features,
            "american_odds": p["american_odds"],
            "time_to_event_minutes": (
                aware(p["event_start_utc"]) - aware(p["observed_at_utc"])
            ).total_seconds()
            / 60,
        }
        p.update(
            features=features,
            filter_id=a["filter_id"],
            policy_event_id=a["policy_event_id"],
            fill_reference=p.get("fill_reference"),
            availability_reference=p.get("availability_reference"),
        )
    elif kind == "SETTLE":
        _required(p, ["observation_event_id", "evidence_reference"])
        observation = state["observations"].get(p["observation_event_id"])
        if not observation or p.get("result") not in {"won", "lost", "push", "void", "pending"}:
            raise ValueError("Existing monitoring observation and explicit outcome required")
        obs = observation["payload"]
        if (
            not aware(obs["event_start_utc"])
            <= aware(p.get("outcome_available_at_utc"))
            <= aware(now)
        ):
            raise ValueError("Outcome must follow event start and be known by recording time")
        prior = state["settlements"].get(p["observation_event_id"])
        if p.get("supersedes_event_id") != (prior["event_id"] if prior else None) or (
            prior and not nonempty(p.get("correction_reason"))
        ):
            raise ValueError("Explicit settlement correction identity and reason required")
        if obs["execution_mode"] == "paper":
            odds = obs["american_odds"]
            unit = {
                "won": odds / 100 if odds > 0 else 100 / abs(odds),
                "lost": -1.0,
                "push": 0.0,
                "void": 0.0,
                "pending": None,
            }[p["result"]]
            profit = None if unit is None else obs["stake_usd"] * unit
        else:
            profit = p.get("reported_net_profit_usd")
            if p["result"] == "pending":
                if profit is not None:
                    raise ValueError("Pending outcomes require unknown profit")
            elif not finite(profit) or not nonempty(p.get("net_profit_evidence_reference")):
                raise ValueError(
                    "Reported funded net profit and evidence required; quote returns are not fills"
                )
            unit = profit / obs["stake_usd"] if profit is not None else None
        p.update(
            unit_return=unit,
            profit_usd=profit,
            application_id=obs["application_id"],
            configuration_sha256=obs["configuration_sha256"],
        )
    elif kind == "MONITOR_REVIEW":
        _required(p, ["application_id", "evidence_reference", "monitoring_report"])
        if (
            p.get("action") not in {"continue", "pause", "investigate", "rollback", "retest"}
            or p["application_id"] not in state["applications"]
        ):
            raise ValueError("Explicit monitoring review action required")
        report = load_lifecycle_report(Path(p["monitoring_report"]))
        spec = report["manifest"]["specification"]
        if spec["events"] != list(state.get("_events", [])) or aware(spec["as_of"]) > aware(now):
            raise ValueError("Current pinned monitoring report required")
        p["monitoring_publication_id"] = report["manifest"]["publication_id"]
        p["automatic_action"] = False
    else:
        raise ValueError("Unknown filter lifecycle event kind")
    json.dumps(p, allow_nan=False)
    return p


def record_lifecycle_event(
    root: Path,
    kind: str,
    payload: dict,
    actor: str,
    rationale: str,
    expected_event_id: str | None = None,
) -> dict:
    """Append an explicit human event with optimistic concurrency and source locks.

    Parameters
    ----------
    root : pathlib.Path
        Private lifecycle journal; existing files are never replaced.
    kind, payload : str, dict
        Explicit command and evidence. Recording time always comes from the system.
    actor, rationale : str
        Human attribution and reason; not an authenticated identity.
    expected_event_id : str, optional
        Required latest global journal head after the first event.

    Returns
    -------
    dict
        Immutable event. No external action is executed.
    """
    if not nonempty(actor) or not nonempty(rationale):
        raise ValueError("Actor and rationale required")
    with ExitStack() as stack:
        # Serialize dependent decisions/applications against new experiment outcomes.
        preview = replay_lifecycle(read_lifecycle(root))
        experiment_root = payload.get("experiment_register") if kind == "DECISION" else None
        if kind in {"CHANGE", "APPROVE_CHANGE", "REJECT_CHANGE", "CONFIRM_APPLICATION"}:
            decision_id = payload.get("decision_event_id")
            if kind != "CHANGE":
                change = preview["changes"].get(payload.get("change_event_id"))
                decision_id = change["event"]["payload"]["decision_event_id"] if change else None
            if decision_id in preview["decisions"]:
                experiment_root = preview["decisions"][decision_id]["payload"]["source"][
                    "experiment_register"
                ]
        source_roots = {str(Path(experiment_root).resolve())} if experiment_root else set()
        if kind in {"CHANGE", "APPROVE_CHANGE", "CONFIRM_APPLICATION"}:
            target = payload if kind == "CHANGE" else (change["event"]["payload"] if change else {})
            evaluation = preview["evaluations"].get(target.get("promotion_assessment_event_id"))
            if evaluation:
                source_roots.update(
                    str(Path(s["experiment_register"]).resolve())
                    for s in evaluation["payload"]["source_anchors"]
                )
        if kind == "EVALUATE_PROMOTION":
            source_roots.update(
                str(Path(row["experiment_register"]).resolve())
                for row in payload.get("sources", [])
            )
        for source_root in sorted(source_roots):
            stack.enter_context(writer_lock(Path(source_root)))
        stack.enter_context(writer_lock(root))
        events = read_lifecycle(root)
        if expected_event_id != (events[-1]["event_id"] if events else None):
            raise ValueError("Stale lifecycle journal head")
        now = utc_now()
        if events and aware(now) < aware(events[-1]["recorded_at"]):
            raise ValueError("Recording clock moved backwards")
        state = replay_lifecycle(events)
        state["_events"] = events
        p = _prepare(state, kind, payload, now)
        event = {
            "contract_version": 1,
            "sequence": len(events) + 1,
            "previous_event_id": expected_event_id,
            "kind": kind,
            "payload": p,
            "actor": actor.strip(),
            "rationale": rationale.strip(),
            "recorded_at": now,
            "external_execution_performed": False,
        }
        event["event_id"] = fingerprint(event)
        replay_lifecycle([*events, event])
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=root) as staging:
            file = Path(staging) / "event.json"
            write_json(file, event)
            os.link(file, root / f"{len(events) + 1:08d}.json")
        return event


def load_lifecycle_report(path: Path) -> dict:
    """Verify a standalone immutable lifecycle/monitoring publication.

    Parameters
    ----------
    path : pathlib.Path
        Exact published directory, never a latest-file lookup.

    Returns
    -------
    dict
        Manifest, pinned journal and monitoring tables.
    """
    from oddsjam_ev.io.parquet import read_parquet_file

    manifest = json.loads((path / "manifest.json").read_text())
    if fingerprint(manifest["specification"]) != manifest["publication_id"]:
        raise ValueError("Lifecycle report integrity failure")
    spec = manifest["specification"]
    if spec["contract_version"] != 1:
        raise ValueError("Unsupported lifecycle report contract")
    replay_lifecycle(spec["events"])
    result = {}
    for name in ("applications", "observations", "metrics", "drift"):
        file = name + ".parquet"
        verify_bytes(path / file, manifest["output_sha256"][file])
        result[name] = read_parquet_file(path / file)
        if len(result[name]) != manifest["table_rows"][name]:
            raise ValueError("Lifecycle report count mismatch")
    return {"manifest": manifest, "events": spec["events"], "tables": result}


def publish_lifecycle_report(root: Path, output_root: Path, as_of: str) -> Path:
    """Pin policy, decisions, settings/application periods and monitoring knowledge.

    Parameters
    ----------
    root, output_root : pathlib.Path
        Journal and immutable private publication root.
    as_of : str
        Explicit aware knowledge cutoff, no later than the system recording clock.

    Returns
    -------
    pathlib.Path
        Verified report directory. Earlier publications remain unchanged.
    """
    if aware(as_of) > aware(utc_now()):
        raise ValueError("Lifecycle report cutoff cannot be in the future")
    if output_root.resolve().is_relative_to(root.resolve()):
        raise ValueError("Publication output cannot modify the journal")
    events = read_lifecycle(root, as_of)
    from oddsjam_ev.analysis import filter_lifecycle

    files = (
        Path(__file__),
        Path(filter_lifecycle.__file__),
        Path(__file__).parents[1] / "analysis" / "decision_config.py",
    )
    spec = {
        "contract_version": 1,
        "as_of": aware(as_of).isoformat(),
        "events": events,
        "register_head": events[-1]["event_id"] if events else None,
        "code_sha256": {
            str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in files
        },
    }
    ident = fingerprint(spec)
    tables = monitoring_tables(events, as_of)
    # Nested feature dicts are exported as JSON rather than inferred Arrow struct schemas.
    if "features" in tables["observations"]:
        tables["observations"]["features"] = tables["observations"].features.map(
            lambda v: json.dumps(v, sort_keys=True)
        )
    with writer_lock(output_root):
        destination = output_root / ident
        if destination.exists():
            load_lifecycle_report(destination)
            return destination
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=output_root) as staging:
            stage = Path(staging)
            hashes = {}
            for name, frame in tables.items():
                file = name + ".parquet"
                frame.to_parquet(stage / file, index=False)
                hashes[file] = hashlib.sha256((stage / file).read_bytes()).hexdigest()
            write_json(
                stage / "manifest.json",
                {
                    "publication_id": ident,
                    "specification": spec,
                    "output_sha256": hashes,
                    "table_rows": {k: len(v) for k, v in tables.items()},
                },
            )
            load_lifecycle_report(stage)
            stage.rename(destination)
    return destination


def validate_experiment_close(item: dict, payload: dict, now: str) -> dict:
    """Bind explicit experiment closure to a published current human review decision.

    Parameters
    ----------
    item, payload : dict
        Current experiment state and selected lifecycle publication/decision reference.
    now : str
        System recording time.

    Returns
    -------
    dict
        Unchanged evidence with the reviewed action attached; no external stop is inferred.
    """
    _required(
        payload, ["review_decision_reference", "review_decision_event_id", "evidence_reference"]
    )
    report = load_lifecycle_report(Path(payload["review_decision_reference"]))
    if aware(report["manifest"]["specification"]["as_of"]) > aware(now):
        raise ValueError("Future lifecycle decision report")
    state = replay_lifecycle(report["events"])
    decision = _current_decision(state, payload["review_decision_event_id"])
    source = decision["payload"]["source"]
    if (
        source["experiment_id"] != item["registration"]["experiment_id"]
        or source["definition_version_id"] != item["protocol"]["definition"]["version_id"]
        or source["experiment_head"] != item["events"][-1]["event_id"]
    ):
        raise ValueError("Closing decision must match the exact current experiment version")
    return {
        **payload,
        "review_outcome": decision["payload"]["action"],
        "lifecycle_publication_id": report["manifest"]["publication_id"],
    }
