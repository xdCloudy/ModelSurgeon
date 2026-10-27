"""Validate and render ModelSurgeon's evidence-backed progress inventory.

The JSON inventory is the sole source for progress percentages and the generated
progress matrices. This tool deliberately uses only the Python standard library
so it can run in local development and the lightweight CI environment.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path("docs/progress/modelsurgeon-progress-v1.json")
README_PATH = Path("README.md")
START_MARKER = "<!-- BEGIN GENERATED MODELSURGEON PROGRESS -->"
END_MARKER = "<!-- END GENERATED MODELSURGEON PROGRESS -->"

DIMENSIONS = ("research", "implementation", "effectiveness", "readiness")
DIMENSION_WEIGHTS = {
    "research": 20,
    "implementation": 25,
    "effectiveness": 35,
    "readiness": 20,
}
RESEARCH_SCORES = {
    "not_investigated": 0,
    "prior_art_reviewed": 12.5,
    "hypothesis_established": 25,
    "feasibility_evaluated": 37.5,
    "experimental_implementation": 50,
    "physical_experiment_conducted": 62.5,
    "baselines_compared": 75,
    "reproducibility_validated": 87.5,
    "research_concluded": 100,
}
IMPLEMENTATION_SCORES = {
    "not_implemented": 0,
    "experimental": 10,
    "prototype": 20,
    "functional": 35,
    "locally_tested": 45,
    "ci_validated": 60,
    "physically_benchmarked": 70,
    "integrated": 80,
    "regression_tested": 90,
    "production_validated": 100,
}
EFFECTIVENESS_SCORES = {
    "not_assessed": 0,
    "measured_negative": 0,
    "measured_inconclusive": 10,
    "quality_gate_passed": 60,
    "replicated_advantage": 85,
    "production_validated": 100,
}
READINESS_SCORES = {
    "not_assessed": 0,
    "experimental": 10,
    "locally_validated": 30,
    "ci_validated": 50,
    "integrated_main": 70,
    "release_validated": 90,
    "production_validated": 100,
}
OUTCOMES = {"positive", "negative", "inconclusive", "infeasible", "superseded", "blocked"}
COMPATIBILITY_SCORES = {
    "verified": 100,
    "experimental": 25,
    "research_only": 10,
    "unknown": 0,
    "unsupported": 0,
}
EVIDENCE_REQUIRED_RESEARCH = set(RESEARCH_SCORES) - {"not_investigated"}
EVIDENCE_REQUIRED_IMPLEMENTATION = set(IMPLEMENTATION_SCORES) - {"not_implemented"}


class ProgressError(ValueError):
    """Invalid canonical progress data or generated-document state."""


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _as_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _markdown(value: object) -> str:
    text = str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")
    return text.strip()


def _label(value: str) -> str:
    return value.replace("_", " ").capitalize()


def _progress_bar(value: float, width: int = 20) -> str:
    bounded = min(100.0, max(0.0, value))
    filled = round(width * bounded / 100.0)
    return f"`{'█' * filled}{'░' * (width - filled)}` {bounded:.1f}%"


def _local_path(root: Path, ref: dict[str, Any]) -> Path:
    raw = ref.get("path")
    if not isinstance(raw, str) or not raw:
        raise ProgressError("evidence reference needs a repository-relative path")
    relative = PurePosixPath(raw.split("#", 1)[0])
    if relative.is_absolute() or ".." in relative.parts:
        raise ProgressError(f"evidence path escapes the repository: {raw}")
    resolved = (root / Path(*relative.parts)).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ProgressError(f"evidence path escapes the repository: {raw}")
    return resolved


def _reference_markdown(refs: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for ref in refs:
        path = str(ref.get("path", ""))
        label = str(ref.get("label", PurePosixPath(path.split("#", 1)[0]).name))
        fragment = "#" + path.split("#", 1)[1] if "#" in path else ""
        relative_path = posixpath.relpath(path.split("#", 1)[0], "docs/progress")
        rendered.append(f"[{_markdown(label)}]({relative_path}{fragment})")
    return ", ".join(rendered) if rendered else "—"


def score_dimension(milestone: dict[str, Any], dimension: str) -> float:
    status_keys: dict[str, str] = {
        "research": "research_status",
        "implementation": "implementation_status",
        "effectiveness": "effectiveness_status",
        "readiness": "readiness_status",
    }
    score_maps: dict[str, Mapping[str, float]] = {
        "research": RESEARCH_SCORES,
        "implementation": IMPLEMENTATION_SCORES,
        "effectiveness": EFFECTIVENESS_SCORES,
        "readiness": READINESS_SCORES,
    }
    status_key = status_keys[dimension]
    score_map = score_maps[dimension]
    status = milestone.get(status_key)
    if not isinstance(status, str) or status not in score_map:
        return 0.0
    return float(score_map[status])


def calculate_progress(data: dict[str, Any]) -> dict[str, Any]:
    workstreams = {item["id"]: item for item in data["workstreams"]}
    milestones_by_workstream: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for milestone in data["milestones"]:
        milestones_by_workstream[milestone["workstream"]].append(milestone)

    overall_dimension: dict[str, float] = {}
    dimension_denominators: dict[str, float] = defaultdict(float)
    dimension_numerators: dict[str, float] = defaultdict(float)
    workstream_scores: dict[str, dict[str, Any]] = {}
    for workstream_id, workstream in workstreams.items():
        items = milestones_by_workstream[workstream_id]
        dim_scores: dict[str, float] = {}
        for dimension in DIMENSIONS:
            applicable = [m for m in items if dimension in m["scored_dimensions"]]
            denominator = sum(float(m["weight"]) for m in applicable)
            if denominator == 0:
                continue
            local_score = (
                sum(float(m["weight"]) * score_dimension(m, dimension) for m in applicable)
                / denominator
            )
            dim_scores[dimension] = local_score
            global_weight = float(workstream["weight"])
            dimension_denominators[dimension] += global_weight
            dimension_numerators[dimension] += global_weight * local_score
        workstream_dim_weights = {
            dim: float(workstream["dimension_weights"][dim]) for dim in dim_scores
        }
        dim_weight_sum = sum(workstream_dim_weights.values())
        stream_score = (
            sum(dim_scores[dim] * weight for dim, weight in workstream_dim_weights.items())
            / dim_weight_sum
            if dim_weight_sum
            else 0.0
        )
        workstream_scores[workstream_id] = {
            "overall": stream_score,
            "dimensions": dim_scores,
            "weight": workstream["weight"],
        }
    for dimension in DIMENSIONS:
        denominator = dimension_denominators[dimension]
        overall_dimension[dimension] = (
            dimension_numerators[dimension] / denominator if denominator else 0.0
        )
    # Aggregate the dimension-weighted score within each workstream first,
    # then apply the explicit workstream weights. This keeps absent dimensions
    # from being silently renormalized across unrelated workstreams.
    overall = sum(
        workstream_scores[workstream_id]["overall"] * float(workstream["weight"])
        for workstream_id, workstream in workstreams.items()
    ) / sum(float(workstream["weight"]) for workstream in workstreams.values())

    compatibility = data["compatibility_cells"]
    compatibility_weight = sum(float(cell["weight"]) for cell in compatibility)
    compatibility_coverage = (
        sum(float(cell["weight"]) * COMPATIBILITY_SCORES[cell["status"]] for cell in compatibility)
        / compatibility_weight
        if compatibility_weight
        else 0.0
    )
    return {
        "overall": overall,
        "dimensions": overall_dimension,
        "workstreams": workstream_scores,
        "compatibility": compatibility_coverage,
        "compatibility_counts": dict(Counter(cell["status"] for cell in compatibility)),
    }


def validate(data: dict[str, Any], root: Path = ROOT) -> list[str]:
    if not isinstance(data, dict):
        return ["canonical progress data must be a JSON object"]
    errors: list[str] = []

    def err(message: str) -> None:
        errors.append(message)

    top_level_required = {
        "schema_version",
        "inventory_id",
        "scope_version",
        "assessment_date",
        "assessment_main_commit",
        "scope_note",
        "scoring",
        "scope_history",
        "workstreams",
        "milestones",
        "research_sources",
        "compatibility_cells",
        "benchmark_records",
        "blockers",
        "priority_order",
        "retired_milestones",
        "backlog_audit",
    }
    missing_top_level = sorted(top_level_required - set(data))
    if missing_top_level:
        err(f"canonical progress data missing top-level fields: {', '.join(missing_top_level)}")
    for field in (
        "scope_history",
        "workstreams",
        "milestones",
        "research_sources",
        "compatibility_cells",
        "benchmark_records",
        "blockers",
        "priority_order",
        "retired_milestones",
    ):
        if not isinstance(data.get(field), list):
            err(f"{field} must be a JSON array")
    schema_path = root / "docs/progress/schema-v1.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema_required = set(schema["required"])
        schema_milestone_required = set(schema["$defs"]["milestone"]["required"])
        schema_source_required = set(schema["$defs"]["researchSource"]["required"])
        source_milestone_fields = {
            "id",
            "workstream",
            "title",
            "description",
            "scope",
            "weight",
            "research_status",
            "research_outcome",
            "implementation_status",
            "effectiveness_status",
            "readiness_status",
            "scored_dimensions",
            "research_questions",
            "acceptance_criteria",
            "dependencies",
            "evidence_refs",
            "source_refs",
            "related_prs",
            "related_commits",
            "remaining_work",
            "limitations",
            "assessment_date",
            "history",
        }
        if schema_required != top_level_required:
            err("schema-v1.json top-level required fields disagree with the validator")
        if schema_milestone_required != source_milestone_fields:
            err("schema-v1.json milestone required fields disagree with the validator")
        source_fields = {
            "id",
            "project",
            "url",
            "pinned_revision",
            "license_spdx",
            "licence_status",
            "method",
            "relevance",
            "research_status",
            "integration_approach",
            "hypothesis",
            "research_outcome",
            "requirements",
            "findings",
            "remaining_investigation",
            "source_paths",
            "assessment_date",
            "history",
        }
        if schema_source_required != source_fields:
            err("schema-v1.json research-source fields disagree with the validator")
        if not schema.get("$schema", "").endswith("2020-12/schema"):
            err("schema-v1.json must declare JSON Schema 2020-12")
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        err(f"cannot validate JSON Schema document: {exc}")

    if data.get("schema_version") != 1:
        err("schema_version must be 1")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(data.get("assessment_date", ""))):
        err("assessment_date must use YYYY-MM-DD")
    else:
        try:
            assessment_day = date.fromisoformat(data["assessment_date"])
            if assessment_day > date.today():
                err("assessment_date cannot be in the future")
            elif date.today() - assessment_day > timedelta(days=180):
                err("assessment is stale (>180 days); reassess milestones and append history")
        except ValueError:
            err("assessment_date is not a valid calendar date")
    if not re.fullmatch(r"[0-9a-f]{40}", str(data.get("assessment_main_commit", ""))):
        err("assessment_main_commit must be a full 40-character lowercase Git SHA")
    if not _is_text(data.get("scope_version")) or not _is_text(data.get("scope_note")):
        err("scope_version and scope_note are required")
    scoring = data.get("scoring")
    if not isinstance(scoring, dict):
        err("scoring must be an object")
    else:
        if scoring.get("dimension_weights") != DIMENSION_WEIGHTS:
            err("scoring.dimension_weights must match the documented fixed dimension weights")
        if (
            scoring.get("workstream_weight_total") != 100
            or scoring.get("milestone_weight_total_per_workstream") != 100
        ):
            err("scoring must declare 100-point workstream and per-workstream milestone totals")

    workstreams = _as_list(data.get("workstreams"))
    stream_ids = [
        item.get("id")
        for item in workstreams
        if isinstance(item, dict) and _is_text(item.get("id"))
    ]
    if len(stream_ids) != len(set(stream_ids)):
        err("workstream identifiers must be unique")
    if (
        not workstreams
        or sum(item.get("weight", 0) for item in workstreams if isinstance(item, dict)) != 100
    ):
        err("workstream weights must be positive integers totaling 100")
    for workstream in workstreams:
        if not isinstance(workstream, dict):
            err("every workstream must be an object")
            continue
        if not _is_text(workstream.get("id")) or not _is_text(workstream.get("title")):
            err("every workstream needs an id and title")
        if not isinstance(workstream.get("weight"), int) or workstream["weight"] <= 0:
            err(f"workstream {workstream.get('id')} needs a positive integer weight")
        if not _is_text(workstream.get("description")):
            err(f"workstream {workstream.get('id')} needs a description")
        dimension_weights = workstream.get("dimension_weights")
        if not isinstance(dimension_weights, dict) or set(dimension_weights) != set(DIMENSIONS):
            err(f"workstream {workstream.get('id')} must define all four dimension weights")
        elif (
            any(not isinstance(v, int) or v < 0 for v in dimension_weights.values())
            or sum(dimension_weights.values()) != 100
        ):
            err(
                f"workstream {workstream.get('id')} dimension weights must be nonnegative integers totaling 100"
            )

    source_list = _as_list(data.get("research_sources"))
    source_ids = [
        item.get("id")
        for item in source_list
        if isinstance(item, dict) and _is_text(item.get("id"))
    ]
    if len(source_ids) != len(set(source_ids)):
        err("research source identifiers must be unique")
    sources = {
        item.get("id"): item
        for item in source_list
        if isinstance(item, dict) and _is_text(item.get("id"))
    }
    for source in source_list:
        if not isinstance(source, dict):
            err("every research source must be an object")
            continue
        for key in (
            "id",
            "project",
            "url",
            "method",
            "relevance",
            "research_status",
            "research_outcome",
            "integration_approach",
            "hypothesis",
            "requirements",
            "findings",
            "remaining_investigation",
            "source_paths",
            "assessment_date",
            "history",
        ):
            if key not in source:
                err(f"research source missing {key}: {source.get('id')}")
        if (
            not isinstance(source.get("research_status"), str)
            or source.get("research_status") not in RESEARCH_SCORES
        ):
            err(f"research source {source.get('id')} has an invalid research status")
        if source.get("research_outcome") is not None and (
            not isinstance(source.get("research_outcome"), str)
            or source.get("research_outcome") not in OUTCOMES
        ):
            err(f"research source {source.get('id')} has an invalid research outcome")
        if (source.get("research_status") == "research_concluded") != (
            source.get("research_outcome") is not None
        ):
            err(f"research source {source.get('id')} must pair research_concluded with an outcome")
        revision = source.get("pinned_revision")
        if revision is not None and not re.fullmatch(r"[0-9a-f]{7,64}", str(revision)):
            err(f"research source {source.get('id')} pinned_revision must be a Git SHA or null")
        if source.get("research_status") != "not_investigated" and not revision:
            err(f"researched source {source.get('id')} must pin its revision")
        if source.get("assessment_date") != data.get("assessment_date"):
            err(f"research source {source.get('id')} assessment_date differs from the dataset")
        if source.get("licence_status") not in {
            "verified",
            "declared_unverified",
            "unknown",
            "not_applicable",
        }:
            err(f"research source {source.get('id')} has an invalid licence_status")
        if source.get("licence_status") == "verified" and not _is_text(source.get("license_spdx")):
            err(f"research source {source.get('id')} needs license_spdx when verified")
        if not isinstance(source.get("source_paths"), list) or any(
            not _is_text(path) for path in source.get("source_paths", [])
        ):
            err(
                f"research source {source.get('id')} source_paths must be a list of non-empty repository paths"
            )
        history = _as_list(source.get("history"))
        if not history:
            err(f"research source {source.get('id')} needs append-only research history")
        else:
            previous_day = ""
            for event in history:
                if not isinstance(event, dict):
                    err(f"research source {source.get('id')} has a non-object history event")
                    continue
                if event.get("date", "") < previous_day:
                    err(f"research source {source.get('id')} history dates must be ordered")
                previous_day = event.get("date", "")
                if (
                    not isinstance(event.get("research_status"), str)
                    or event.get("research_status") not in RESEARCH_SCORES
                ):
                    err(f"research source {source.get('id')} history has invalid research_status")
                if event.get("research_outcome") is not None and (
                    not isinstance(event.get("research_outcome"), str)
                    or event.get("research_outcome") not in OUTCOMES
                ):
                    err(f"research source {source.get('id')} history has invalid research_outcome")
                if (event.get("research_status") == "research_concluded") != (
                    event.get("research_outcome") is not None
                ):
                    err(
                        f"research source {source.get('id')} history must pair conclusion and outcome"
                    )
            latest = history[-1]
            for key in (
                "research_status",
                "research_outcome",
                "findings",
                "pinned_revision",
                "licence_status",
                "source_paths",
            ):
                if latest.get(key) != source.get(key):
                    err(
                        f"research source {source.get('id')} current {key} differs from latest history"
                    )

    milestones = _as_list(data.get("milestones"))
    milestone_ids = [
        item.get("id") for item in milestones if isinstance(item, dict) and _is_text(item.get("id"))
    ]
    if len(milestone_ids) != len(set(milestone_ids)):
        err("milestone identifiers must be unique")
    known_ids = set(milestone_ids)
    stream_id_set = set(stream_ids)
    required_milestone_fields = (
        "id",
        "workstream",
        "title",
        "description",
        "scope",
        "weight",
        "research_status",
        "research_outcome",
        "implementation_status",
        "effectiveness_status",
        "readiness_status",
        "scored_dimensions",
        "research_questions",
        "acceptance_criteria",
        "dependencies",
        "evidence_refs",
        "source_refs",
        "related_prs",
        "related_commits",
        "remaining_work",
        "limitations",
        "assessment_date",
        "history",
    )
    milestone_by_id: dict[str, dict[str, Any]] = {}
    for item in milestones:
        if isinstance(item, dict) and _is_text(item.get("id")):
            milestone_by_id[item["id"]] = item
    for milestone in milestones:
        if not isinstance(milestone, dict):
            err("every milestone must be an object")
            continue
        missing = [key for key in required_milestone_fields if key not in milestone]
        if missing:
            err(f"milestone {milestone.get('id')} missing required fields: {', '.join(missing)}")
            continue
        array_fields = (
            "scored_dimensions",
            "research_questions",
            "acceptance_criteria",
            "dependencies",
            "evidence_refs",
            "source_refs",
            "related_prs",
            "related_commits",
            "limitations",
            "history",
        )
        bad_arrays = [field for field in array_fields if not isinstance(milestone[field], list)]
        if bad_arrays:
            err(f"milestone {milestone.get('id')} fields must be arrays: {', '.join(bad_arrays)}")
            continue
        if (
            not isinstance(milestone["workstream"], str)
            or milestone["workstream"] not in stream_id_set
        ):
            err(
                f"milestone {milestone['id']} references unknown workstream {milestone['workstream']}"
            )
        if not isinstance(milestone["weight"], int) or milestone["weight"] <= 0:
            err(f"milestone {milestone['id']} weight must be a positive integer")
        if (
            not isinstance(milestone["research_status"], str)
            or milestone["research_status"] not in RESEARCH_SCORES
        ):
            err(f"milestone {milestone['id']} has an invalid research_status")
        if (
            not isinstance(milestone["implementation_status"], str)
            or milestone["implementation_status"] not in IMPLEMENTATION_SCORES
        ):
            err(f"milestone {milestone['id']} has an invalid implementation_status")
        if (
            not isinstance(milestone["effectiveness_status"], str)
            or milestone["effectiveness_status"] not in EFFECTIVENESS_SCORES
        ):
            err(f"milestone {milestone['id']} has an invalid effectiveness_status")
        if (
            not isinstance(milestone["readiness_status"], str)
            or milestone["readiness_status"] not in READINESS_SCORES
        ):
            err(f"milestone {milestone['id']} has an invalid readiness_status")
        outcome = milestone["research_outcome"]
        if outcome is not None and (not isinstance(outcome, str) or outcome not in OUTCOMES):
            err(f"milestone {milestone['id']} has an invalid research_outcome")
        if (milestone["research_status"] == "research_concluded") != (outcome is not None):
            err(
                f"milestone {milestone['id']} must have an outcome exactly when research is concluded"
            )
        if (
            not milestone["scored_dimensions"]
            or any(
                not isinstance(dimension, str) or dimension not in DIMENSIONS
                for dimension in milestone["scored_dimensions"]
            )
            or len(set(milestone["scored_dimensions"])) != len(milestone["scored_dimensions"])
        ):
            err(f"milestone {milestone['id']} has invalid scored_dimensions")
        for field in ("description", "scope", "remaining_work"):
            if not _is_text(milestone[field]):
                err(f"milestone {milestone['id']} needs a non-empty {field}")
        if not _as_list(milestone["acceptance_criteria"]):
            err(f"milestone {milestone['id']} needs measurable acceptance criteria")
        if not _as_list(milestone["research_questions"]) or any(
            not _is_text(question) for question in milestone["research_questions"]
        ):
            err(f"milestone {milestone['id']} needs specific research questions")
        if milestone["assessment_date"] != data.get("assessment_date"):
            err(f"milestone {milestone['id']} assessment_date differs from the dataset")
        if (
            sum(
                1
                for item in milestones
                if isinstance(item, dict) and item.get("workstream") == milestone["workstream"]
            )
            == 0
        ):
            err(f"workstream {milestone['workstream']} has no milestones")
        if (
            milestone["research_status"] in EVIDENCE_REQUIRED_RESEARCH
            and not milestone["evidence_refs"]
            and not milestone["source_refs"]
        ):
            err(
                f"milestone {milestone['id']} progressed research without evidence or source references"
            )
        if (
            milestone["implementation_status"] in EVIDENCE_REQUIRED_IMPLEMENTATION
            and not milestone["evidence_refs"]
        ):
            err(f"milestone {milestone['id']} progressed implementation without local evidence")
        for source_id in milestone["source_refs"]:
            if not isinstance(source_id, str) or source_id not in sources:
                err(f"milestone {milestone['id']} references unknown research source {source_id}")
        for dependency in milestone["dependencies"]:
            if not isinstance(dependency, str) or dependency not in known_ids:
                err(f"milestone {milestone['id']} references unknown dependency {dependency}")
        for pr in milestone["related_prs"]:
            if not isinstance(pr, int) or pr <= 0:
                err(f"milestone {milestone['id']} has invalid pull request reference {pr}")
        for commit in milestone["related_commits"]:
            if not re.fullmatch(r"[0-9a-f]{7,40}", str(commit)):
                err(f"milestone {milestone['id']} has invalid commit reference {commit}")
        for ref in milestone["evidence_refs"]:
            _validate_evidence_ref(ref, root, err, f"milestone {milestone['id']}")
        history = milestone["history"]
        if not isinstance(history, list) or not history:
            err(f"milestone {milestone['id']} needs append-only assessment history")
            continue
        previous: dict[str, Any] | None = None
        last_date = ""
        for event in history:
            if not isinstance(event, dict):
                err(f"milestone {milestone['id']} has a non-object history event")
                continue
            if event.get("date", "") < last_date:
                err(f"milestone {milestone['id']} history dates must be ordered")
            last_date = event.get("date", "")
            for field in (
                "research_status",
                "research_outcome",
                "implementation_status",
                "effectiveness_status",
                "readiness_status",
            ):
                if field not in event:
                    err(f"milestone {milestone['id']} history event missing {field}")
            if event.get("research_status") not in RESEARCH_SCORES:
                err(f"milestone {milestone['id']} history has invalid research_status")
            if event.get("implementation_status") not in IMPLEMENTATION_SCORES:
                err(f"milestone {milestone['id']} history has invalid implementation_status")
            if event.get("effectiveness_status") not in EFFECTIVENESS_SCORES:
                err(f"milestone {milestone['id']} history has invalid effectiveness_status")
            if event.get("readiness_status") not in READINESS_SCORES:
                err(f"milestone {milestone['id']} history has invalid readiness_status")
            if (
                event.get("research_outcome") is not None
                and event.get("research_outcome") not in OUTCOMES
            ):
                err(f"milestone {milestone['id']} history has invalid research_outcome")
            if (event.get("research_status") == "research_concluded") != (
                event.get("research_outcome") is not None
            ):
                err(
                    f"milestone {milestone['id']} history must pair research_concluded with an outcome"
                )
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(event.get("date", ""))):
                err(f"milestone {milestone['id']} history date must use YYYY-MM-DD")
            else:
                try:
                    date.fromisoformat(event["date"])
                except ValueError:
                    err(f"milestone {milestone['id']} history date is invalid")
            if previous is not None:
                for field, score_map in (
                    ("research_status", RESEARCH_SCORES),
                    ("implementation_status", IMPLEMENTATION_SCORES),
                    ("readiness_status", READINESS_SCORES),
                ):
                    if (
                        event.get(field) in score_map
                        and previous.get(field) in score_map
                        and score_map[event[field]] < score_map[previous[field]]
                        and (
                            not event.get("reassessment")
                            or not _is_text(event.get("reason"))
                            or not event.get("evidence_refs")
                        )
                    ):
                        err(
                            f"milestone {milestone['id']} regressed {field} without an evidence-backed reassessment"
                        )
                if (
                    previous.get("research_outcome") != event.get("research_outcome")
                    and previous.get("research_outcome") is not None
                    and (not event.get("reassessment") or not _is_text(event.get("reason")))
                ):
                    err(
                        f"milestone {milestone['id']} changed a concluded research outcome without reassessment rationale"
                    )
            for ref in _as_list(event.get("evidence_refs")):
                _validate_evidence_ref(ref, root, err, f"milestone {milestone['id']} history")
            previous = event
        if previous:
            for field in (
                "research_status",
                "research_outcome",
                "implementation_status",
                "effectiveness_status",
                "readiness_status",
            ):
                if previous.get(field) != milestone.get(field):
                    err(
                        f"milestone {milestone['id']} current {field} does not match its latest history event"
                    )
        if (
            milestone["readiness_status"] == "integrated_main"
            and IMPLEMENTATION_SCORES[milestone["implementation_status"]]
            < IMPLEMENTATION_SCORES["integrated"]
        ):
            err(
                f"milestone {milestone['id']} claims main readiness without integrated implementation"
            )

    stream_milestone_weights: dict[str, int] = Counter()
    for milestone in milestones:
        if (
            isinstance(milestone, dict)
            and isinstance(milestone.get("workstream"), str)
            and milestone.get("workstream") in stream_id_set
            and isinstance(milestone.get("weight"), int)
        ):
            stream_milestone_weights[milestone["workstream"]] += milestone["weight"]
    for workstream in workstreams:
        if isinstance(workstream, dict) and isinstance(workstream.get("id"), str) and stream_milestone_weights[workstream["id"]] != 100:
            err(
                f"milestone weights for {workstream['id']} must total 100; got {stream_milestone_weights[workstream['id']]}"
            )
    _validate_dependency_graph(milestone_by_id, err)

    cells = _as_list(data.get("compatibility_cells"))
    cell_ids = [
        cell.get("id") for cell in cells if isinstance(cell, dict) and _is_text(cell.get("id"))
    ]
    if len(cell_ids) != len(set(cell_ids)):
        err("compatibility cell identifiers must be unique")
    for cell in cells:
        if not isinstance(cell, dict):
            err("every compatibility cell must be an object")
            continue
        for key in (
            "id",
            "architecture",
            "model_scope",
            "format",
            "quantization",
            "operation",
            "model_type",
            "hardware_backend",
            "runtime",
            "recovery_method",
            "theoretical_status",
            "implementation_level",
            "validation_level",
            "status",
            "evidence_refs",
            "limitations",
            "assessment_date",
            "weight",
        ):
            if key not in cell:
                err(f"compatibility cell missing {key}: {cell.get('id')}")
        for key in (
            "id", "architecture", "model_scope", "format", "quantization", "operation",
            "hardware_backend", "runtime", "recovery_method", "assessment_date",
        ):
            if not _is_text(cell.get(key)):
                err(f"compatibility cell {cell.get('id')} needs a non-empty {key}")
        if not isinstance(cell.get("weight"), int) or cell["weight"] <= 0:
            err(f"compatibility cell {cell.get('id')} needs a positive integer weight")
        if not isinstance(cell.get("evidence_refs"), list) or not isinstance(cell.get("limitations"), list):
            err(f"compatibility cell {cell.get('id')} evidence_refs and limitations must be arrays")
        if (
            not isinstance(cell.get("status"), str)
            or cell.get("status") not in COMPATIBILITY_SCORES
        ):
            err(f"compatibility cell {cell.get('id')} has an invalid status")
        if cell.get("status") == "verified" and not cell.get("evidence_refs"):
            err(f"verified compatibility cell {cell.get('id')} needs evidence")
        if cell.get("model_type") not in {"dense", "moe", "multimodal", "unknown"}:
            err(f"compatibility cell {cell.get('id')} has an invalid model_type")
        if cell.get("theoretical_status") not in {"plausible", "unknown", "unsupported"}:
            err(f"compatibility cell {cell.get('id')} has an invalid theoretical_status")
        if cell.get("implementation_level") not in {
            "not_implemented", "in_pr", "experimental", "integrated_main"
        }:
            err(f"compatibility cell {cell.get('id')} has an invalid implementation_level")
        if cell.get("validation_level") not in {
            "not_tested", "synthetic", "physical", "independent_reproduction"
        }:
            err(f"compatibility cell {cell.get('id')} has an invalid validation_level")
        if cell.get("status") == "verified" and (
            cell.get("implementation_level") != "integrated_main"
            or cell.get("validation_level") not in {"physical", "independent_reproduction"}
            or cell.get("theoretical_status") != "plausible"
        ):
            err(
                f"verified compatibility cell {cell.get('id')} requires plausible theory, main integration, and physical validation"
            )
        if cell.get("implementation_level") == "in_pr" and cell.get("status") == "verified":
            err(f"unmerged compatibility cell {cell.get('id')} cannot be verified support")
        if cell.get("assessment_date") != data.get("assessment_date"):
            err(f"compatibility cell {cell.get('id')} assessment_date differs from the dataset")
        for ref in _as_list(cell.get("evidence_refs")):
            _validate_evidence_ref(ref, root, err, f"compatibility cell {cell.get('id')}")

    benchmarks = _as_list(data.get("benchmark_records"))
    benchmark_ids = [
        record.get("id")
        for record in benchmarks
        if isinstance(record, dict) and _is_text(record.get("id"))
    ]
    if len(benchmark_ids) != len(set(benchmark_ids)):
        err("benchmark record identifiers must be unique")
    for record in benchmarks:
        if not isinstance(record, dict):
            err("every benchmark record must be an object")
            continue
        for key in (
            "id",
            "name",
            "method",
            "model_scope",
            "dataset",
            "split",
            "outcome",
            "finding",
            "limitations",
            "evidence_refs",
            "integrated_on_main",
            "assessment_date",
        ):
            if key not in record:
                err(f"benchmark record missing {key}: {record.get('id')}")
        if not isinstance(record.get("integrated_on_main"), bool):
            err(f"benchmark record {record.get('id')} integrated_on_main must be boolean")
        if record.get("integrated_on_main") is False and not record.get("pull_requests"):
            err(f"benchmark record {record.get('id')} must cite its pending PR if not on main")
        if record.get("assessment_date") != data.get("assessment_date"):
            err(f"benchmark record {record.get('id')} assessment_date differs from the dataset")
        if not record.get("evidence_refs"):
            err(f"benchmark record {record.get('id')} needs evidence references")
        for ref in _as_list(record.get("evidence_refs")):
            _validate_evidence_ref(ref, root, err, f"benchmark record {record.get('id')}")

    retired = _as_list(data.get("retired_milestones"))
    retired_ids = [item.get("id") for item in retired if isinstance(item, dict)]
    if len(retired_ids) != len(set(retired_ids)) or set(retired_ids).intersection(known_ids):
        err("retired milestone IDs must be unique and cannot also be active")
    for item in retired:
        if not isinstance(item, dict) or not all(
            _is_text(item.get(field)) for field in ("id", "retired_date", "reason")
        ):
            err("each retired milestone needs an id, retired_date, and reason")

    blockers = _as_list(data.get("blockers"))
    for blocker in blockers:
        if not isinstance(blocker, dict) or not all(
            _is_text(blocker.get(k)) for k in ("id", "description", "impact", "next_action")
        ):
            err("each blocker needs id, description, impact, and next_action")
    for priority in _as_list(data.get("priority_order")):
        if priority not in known_ids:
            err(f"priority_order references unknown milestone {priority}")

    backlog = data.get("backlog_audit")
    if not isinstance(backlog, dict):
        err("backlog_audit must be an object")
    else:
        for field in (
            "audit_date",
            "main_commit",
            "open_issue_count",
            "open_pull_request_count",
            "scope_note",
            "notable_pull_requests",
            "dependency_chains",
            "triage",
        ):
            if field not in backlog:
                err(f"backlog_audit missing {field}")
        if backlog.get("audit_date") != data.get("assessment_date"):
            err("backlog_audit date differs from the dataset assessment date")
        if backlog.get("main_commit") != data.get("assessment_main_commit"):
            err("backlog_audit main commit differs from the assessed main commit")
        for field in ("open_issue_count", "open_pull_request_count"):
            if not isinstance(backlog.get(field), int) or backlog[field] < 0:
                err(f"backlog_audit {field} must be a nonnegative integer")
        notable = _as_list(backlog.get("notable_pull_requests"))
        pr_ids = [item.get("number") for item in notable if isinstance(item, dict)]
        if len(pr_ids) != len(set(pr_ids)):
            err("backlog_audit notable PR numbers must be unique")
        if len(notable) > backlog.get("open_pull_request_count", 0):
            err("backlog_audit has more notable PRs than total open PRs")
        pr_by_id = {item.get("number"): item for item in notable if isinstance(item, dict)}
        for item in notable:
            if not isinstance(item, dict) or not all(
                _is_text(item.get(key))
                for key in ("title", "head_branch", "base_branch", "relationship", "status")
            ):
                err(
                    "every notable PR needs title, head_branch, base_branch, relationship, and status"
                )
                continue
            if not isinstance(item.get("number"), int) or item.get("number", 0) <= 0:
                err("notable PR number must be a positive integer")
            if item.get("status") != "open" or item.get("integrated_on_main") is not False:
                err(f"notable PR #{item.get('number')} must remain pending and not integrated")
        for chain in _as_list(backlog.get("dependency_chains")):
            if not isinstance(chain, list) or len(chain) < 2:
                err("backlog dependency chains need at least two PR numbers")
                continue
            for number in chain:
                if number not in pr_by_id:
                    err(f"backlog dependency chain references unlisted PR #{number}")
            for base_number, dependent_number in pairwise(chain):
                base_pr = pr_by_id.get(base_number)
                dependent_pr = pr_by_id.get(dependent_number)
                if (
                    base_pr
                    and dependent_pr
                    and base_pr.get("head_branch") != dependent_pr.get("base_branch")
                ):
                    err(
                        f"backlog chain #{base_number} -> #{dependent_number} does not match PR branch relationship"
                    )

    return errors


def _validate_evidence_ref(ref: Any, root: Path, err: Any, context: str) -> None:
    if not isinstance(ref, dict) or not _is_text(ref.get("path")):
        err(f"{context} evidence references must contain a path")
        return
    try:
        path = _local_path(root, ref)
    except ProgressError as exc:
        err(f"{context}: {exc}")
        return
    if not path.is_file():
        err(f"{context} evidence file does not exist: {ref['path']}")


def _validate_dependency_graph(milestones: dict[str, dict[str, Any]], err: Any) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            err(f"milestone dependency cycle contains {node}")
            return
        if node in visited:
            return
        visiting.add(node)
        for child in milestones[node].get("dependencies", []):
            if child in milestones:
                visit(child)
        visiting.remove(node)
        visited.add(node)

    for milestone_id in milestones:
        visit(milestone_id)


def _workstream_map(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in data["workstreams"]}


def render_readme_block(data: dict[str, Any], scores: dict[str, Any]) -> str:
    streams = _workstream_map(data)
    milestones_by_stream: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in data["milestones"]:
        milestones_by_stream[item["workstream"]].append(item)
    lines = [
        START_MARKER,
        "### Evidence-backed development dashboard",
        "",
        f"**Overall completion:** {_progress_bar(scores['overall'])}",
        f"**Learned Meta-Surgeon:** {_progress_bar(scores['workstreams']['meta_surgeon']['overall'])}",
        f"**Research coverage:** {scores['dimensions']['research']:.1f}% · "
        f"**Optimization effectiveness:** {scores['dimensions']['effectiveness']:.1f}% · "
        f"**Model compatibility:** {scores['compatibility']:.1f}% · "
        f"**Production readiness:** {scores['dimensions']['readiness']:.1f}%",
        "",
        f"Assessment date: **{data['assessment_date']}** against main `{data['assessment_main_commit'][:12]}`. "
        "Only evidence present on the assessed main commit contributes; open PRs are tracked but not counted as integrated. "
        f"Backlog snapshot: **{data['backlog_audit']['open_issue_count']} open issues / "
        f"{data['backlog_audit']['open_pull_request_count']} open PRs**. Negative research can be complete "
        "while effectiveness remains zero. See the [scoring rules](docs/progress/README.md#scoring-method) "
        "and [PR backlog](docs/progress/PR_BACKLOG.md).",
        "",
        "| Workstream | Weight | Overall | Research | Implementation | Effectiveness | Readiness | Details |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    detail_links = {
        "meta_surgeon": "META_SURGEON.md",
        "pruning": "OPTIMIZATION_METHODS.md",
        "quantization": "OPTIMIZATION_METHODS.md",
        "moe": "OPTIMIZATION_METHODS.md",
        "recovery": "OPTIMIZATION_METHODS.md",
        "architecture": "OPTIMIZATION_METHODS.md",
        "search": "OPTIMIZATION_METHODS.md",
        "compatibility": "MODEL_COMPATIBILITY.md",
        "hardware": "BENCHMARKS.md",
        "benchmarking": "BENCHMARKS.md",
        "scientific_research": "RESEARCH_INVENTORY.md",
        "production_readiness": "PRODUCTION_READINESS.md",
    }
    for stream_id, stream in streams.items():
        computed = scores["workstreams"][stream_id]
        dims = computed["dimensions"]
        link = detail_links.get(stream_id, "README.md")
        cells = [f"{dims.get(dim, 0):.0f}%" for dim in DIMENSIONS]
        lines.append(
            f"| {_markdown(stream['title'])} | {stream['weight']}% | {computed['overall']:.0f}% | "
            + " | ".join(cells)
            + f" | [matrix](docs/progress/{link}) |"
        )
    lines.extend(
        [
            "",
            "**Current capability boundary:** existing Hugging Face surgery is physically validated only in bounded model/runtime cells; native GGUF coverage is operation/codec-specific. The learned Meta-Surgeon exists and has been tested, but held-out transfer has not shown a dependable optimization advantage. See the [generated feature matrix](docs/progress/FEATURE_MATRIX.md) and the existing [goal-to-reality audit](docs/research/goal-reality-gap-matrix.md).",
            "",
            "**Capability summary:** validated means bounded physical evidence exists; experimental means the path or study is incomplete; research-only means no integrated implementation claim; unsupported/unknown cells remain fail-closed. Physical surgery and task checks exist for named Hugging Face checkpoints and a narrow GGUF Llama Q4_K_M cell. Wider quantization, MoE, multimodal, unseen-family transfer, broad task quality, and production optimizer readiness remain unproven or unsupported. Failed quality gates remain in the evidence ledger; no performance superiority is claimed.",
            "",
            "**Compatibility:** "
            + ", ".join(
                f"{count} {_label(status).lower()}"
                for status, count in sorted(scores["compatibility_counts"].items())
            )
            + f" representative cells; weighted verified-support coverage is {scores['compatibility']:.1f}%. See the [compatibility matrix](docs/progress/MODEL_COMPATIBILITY.md).",
            "",
            "**Research findings:** the inventory preserves measured positive, negative, inconclusive, infeasible, and unavailable results. No single experiment establishes cross-model superiority; see the [benchmark ledger](docs/progress/BENCHMARKS.md) and [research inventory](docs/progress/RESEARCH_INVENTORY.md).",
            "",
            "**Next priorities:** "
            + "; ".join(
                _markdown(next(m["title"] for m in data["milestones"] if m["id"] == item))
                for item in data["priority_order"][:3]
            )
            + ".",
            "",
            "**Documentation:** [progress overview](docs/progress/README.md) · [feature matrix](docs/progress/FEATURE_MATRIX.md) · [Meta-Surgeon](docs/progress/META_SURGEON.md) · [optimization methods](docs/progress/OPTIMIZATION_METHODS.md) · [model compatibility](docs/progress/MODEL_COMPATIBILITY.md) · [research inventory](docs/progress/RESEARCH_INVENTORY.md) · [benchmarks](docs/progress/BENCHMARKS.md) · [production readiness](docs/progress/PRODUCTION_READINESS.md) · [PR backlog snapshot](docs/progress/PR_BACKLOG.md) · [canonical data](docs/progress/modelsurgeon-progress-v1.json) · [schema](docs/progress/schema-v1.json) · [update workflow](docs/progress/CONTRIBUTING.md) · [roadmap](ROADMAP.md) · [architecture](ARCHITECTURE.md) · [user guides](docs/user-guides/README.md).",
            END_MARKER,
        ]
    )
    return "\n".join(lines)


def render_feature_matrix(data: dict[str, Any], scores: dict[str, Any]) -> str:
    streams = _workstream_map(data)
    lines = [
        "# ModelSurgeon feature and milestone matrix",
        "",
        f"Assessment: {data['assessment_date']} against main `{data['assessment_main_commit']}`. Overall weighted completion is **{scores['overall']:.1f}%**.",
        "",
        "A completed research stage is not an implementation claim. Effectiveness uses physical outcome evidence; a negative or inconclusive experiment can close research while leaving the optimization effectiveness score at zero. An `integrated` implementation must be present on the assessed main commit; open PRs do not count.",
        "",
        "| Workstream | Weight | Overall | Research | Implementation | Effectiveness | Readiness |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stream_id, stream in streams.items():
        result = scores["workstreams"][stream_id]
        dim = result["dimensions"]
        lines.append(
            f"| {_markdown(stream['title'])} | {stream['weight']}% | {result['overall']:.1f}% | "
            + " | ".join(f"{dim.get(key, 0):.1f}%" for key in DIMENSIONS)
            + " |"
        )
    lines.extend(["", "## Milestones", ""])
    for stream_id, stream in streams.items():
        lines.extend([f"### {stream['title']}", "", _markdown(stream["description"]), ""])
        lines.extend(
            [
                "| ID | Objective and scope | Research questions | Research | Outcome | Implementation | Effectiveness | Readiness | Acceptance criteria | Dependencies | Evidence | PRs / commits | Remaining work | Limitations |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for milestone in (m for m in data["milestones"] if m["workstream"] == stream_id):
            lines.append(
                f"| `{milestone['id']}` | **{_markdown(milestone['title'])}** — {_markdown(milestone['scope'])} | "
                f"{_markdown('; '.join(milestone['research_questions']))} | "
                f"{_label(milestone['research_status'])} | {_label(milestone['research_outcome']) if milestone['research_outcome'] else 'Open'} | "
                f"{_label(milestone['implementation_status'])} | {_label(milestone['effectiveness_status'])} | "
                f"{_label(milestone['readiness_status'])} | "
                f"{_markdown('; '.join(milestone['acceptance_criteria']))} | "
                f"{', '.join(f'`{dependency}`' for dependency in milestone['dependencies']) or '—'} | "
                f"{_reference_markdown(milestone['evidence_refs'])} | "
                f"{', '.join(f'#{number}' for number in milestone['related_prs']) or '—'} / "
                f"{', '.join(f'`{sha}`' for sha in milestone['related_commits']) or '—'} | "
                f"{_markdown(milestone['remaining_work'])} | {_markdown('; '.join(milestone['limitations']))} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_meta_surgeon(data: dict[str, Any], scores: dict[str, Any]) -> str:
    rows = [m for m in data["milestones"] if m["workstream"] == "meta_surgeon"]
    lines = [
        "# Learned Meta-Surgeon progress",
        "",
        f"Assessment: {data['assessment_date']} · maturity score: **{scores['workstreams']['meta_surgeon']['overall']:.1f}%**. This is implementation and evidence maturity, not a claim that learned decisions improve optimization.",
        "",
        "The Meta-Surgeon remains the central scientific objective. Each row separates research closure, implementation status, measured effectiveness, and readiness. Predictor existence, falling training loss, or offline AUC alone does not establish optimization value.",
        "",
        "| Capability | Research question | Research | Outcome | Implementation | Effectiveness | Acceptance test / target | Evidence | Remaining work |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in rows:
        lines.append(
            f"| `{item['id']}` — **{_markdown(item['title'])}** | {_markdown('; '.join(item['research_questions']))} | {_label(item['research_status'])} | "
            f"{_label(item['research_outcome']) if item['research_outcome'] else 'Open'} | {_label(item['implementation_status'])} | "
            f"{_label(item['effectiveness_status'])} | {_markdown('; '.join(item['acceptance_criteria']))} | "
            f"{_reference_markdown(item['evidence_refs'])} | {_markdown(item['remaining_work'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            "Do not promote the Meta-Surgeon as an optimization advantage until held-out, model-grouped physical comparisons show a statistically defensible gain over random, magnitude, and no-guidance baselines at equal evaluation budget, while meeting declared quality and resource gates. Record regressions and abstentions as evidence; do not discard them.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_methods(data: dict[str, Any]) -> str:
    method_streams = {"pruning", "quantization", "moe", "recovery", "architecture", "search"}
    lines = [
        "# Optimization method maturity",
        "",
        f"Assessment: {data['assessment_date']}. Statuses describe ModelSurgeon's implementation and evidence, not upstream capability claims.",
        "",
        "| Method area | Research | Conclusion | Implementation | Effectiveness | Evidence | Acceptance criteria / remaining work |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    stream_titles = {item["id"]: item["title"] for item in data["workstreams"]}
    for item in data["milestones"]:
        if item["workstream"] not in method_streams:
            continue
        lines.append(
            f"| **{stream_titles[item['workstream']]}** · {_markdown(item['title'])} (`{item['id']}`) | "
            f"{_label(item['research_status'])} | {_label(item['research_outcome']) if item['research_outcome'] else 'Open'} | "
            f"{_label(item['implementation_status'])} | {_label(item['effectiveness_status'])} | {_reference_markdown(item['evidence_refs'])} | "
            f"{_markdown('; '.join(item['acceptance_criteria']))} Remaining: {_markdown(item['remaining_work'])} |"
        )
    return "\n".join(lines) + "\n"


def render_compatibility(data: dict[str, Any], scores: dict[str, Any]) -> str:
    lines = [
        "# Representative model, format, runtime, and operation compatibility",
        "",
        f"Assessment: {data['assessment_date']} · weighted verified-support coverage: **{scores['compatibility']:.1f}%** across {len(data['compatibility_cells'])} declared representative cells.",
        "",
        "This is a curated, evidence-bounded set, not a Cartesian claim over every model x format x quantization x operation x runtime x recovery method. The table separates theoretical plausibility, implementation location, and validation level. `in_pr` is not main support; only physical or independently reproduced results can be `verified`.",
        "",
        "| Cell | Architecture / scope | Dense/MoE/etc. | Format / quantization | Operation | Backend / runtime | Recovery | Theory | Implementation | Validation | Status | Evidence | Limitations |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in data["compatibility_cells"]:
        lines.append(
            f"| `{cell['id']}` | {_markdown(cell['architecture'])} / {_markdown(cell['model_scope'])} | {_label(cell['model_type'])} | {_markdown(cell['format'])} / {_markdown(cell['quantization'])} | "
            f"{_markdown(cell['operation'])} | {_markdown(cell['hardware_backend'])} / {_markdown(cell['runtime'])} | {_markdown(cell['recovery_method'])} | {_label(cell['theoretical_status'])} | {_label(cell['implementation_level'])} | {_label(cell['validation_level'])} | {_label(cell['status'])} | "
            f"{_reference_markdown(cell['evidence_refs'])} | {_markdown('; '.join(cell['limitations']))} |"
        )
    lines.extend(
        [
            "",
            "## Rules for expanding coverage",
            "",
            "Add a representative cell only when an exact architecture/configuration/operation boundary has a concrete test or experiment plan. Promote it to verified only with physical artifact validation, reload/runtime evidence, and quality/resource measurements appropriate to the operation. Keep adjacent but untested combinations unknown or unsupported; never infer them from a family name.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_research(data: dict[str, Any]) -> str:
    source_by_id = {source["id"]: source for source in data["research_sources"]}
    lines = [
        "# Independent research inventory",
        "",
        f"Assessment: {data['assessment_date']} · {len(source_by_id)} source/project records. Uninvestigated entries remain explicitly unpinned; they are research backlog, not evidence of support.",
        "",
        "| Project / method | Revision | Licence | Research status | Outcome | Source paths inspected | Approach | Relevance / decision | Hypothesis and validation | Findings | Remaining investigation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for source in data["research_sources"]:
        revision = (
            f"[`{source['pinned_revision'][:12]}`]({source['url']}/commit/{source['pinned_revision']})"
            if source.get("pinned_revision")
            else "Not pinned"
        )
        license_text = source.get("license_spdx") or source.get("licence_note", "Not verified")
        source_paths = source.get("source_paths", [])
        source_path_links = ", ".join(
            f"[`{_markdown(path)}`]({source['url']}/blob/{source['pinned_revision']}/{path})"
            for path in source_paths
            if source.get("pinned_revision")
        ) or ("Not inspected" if not source_paths else "Pinned revision unavailable")
        lines.append(
            f"| [{_markdown(source['project'])}]({source['url']})"
            + (f" · [paper]({source['paper_url']})" if source.get("paper_url") else "")
            + f" · {_markdown(source['method'])} | {revision} | {_markdown(license_text)} | "
            f"{_label(source['research_status'])} | {_label(source['research_outcome']) if source['research_outcome'] else 'Open'} | {source_path_links} | {_markdown(source['integration_approach'])} | {_markdown(source['relevance'])} | "
            f"Hypothesis: {_markdown(source['hypothesis'])} Validation: {_markdown(source['requirements'])} | "
            f"{_markdown(source['findings'])} | {_markdown(source['remaining_investigation'])} |"
        )
    lines.extend(
        [
            "",
            "## Research-stage policy",
            "",
            "Prior art review, feasibility, implementation, physical experiments, matched baselines, and reproducible conclusions are separate stages. A negative study can be research-complete, but its method remains ineffective or unsupported. When a source is revisited, pin the exact commit, inspect its licence and actual source path, and append a dated conclusion rather than replacing earlier evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_benchmarks(data: dict[str, Any]) -> str:
    lines = [
        "# Benchmark and experiment evidence ledger",
        "",
        f"Assessment: {data['assessment_date']}. This ledger reports only records with inspectable local evidence references; unavailable and unmerged experiments remain visible but do not count as main-integrated results.",
        "",
        "| Record | Method / baseline | Model | Dataset / split | Outcome | Main-integrated | Finding | Evidence | Limitations |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in data["benchmark_records"]:
        lines.append(
            f"| `{record['id']}` · {_markdown(record['name'])} | {_markdown(record['method'])} | {_markdown(record['model_scope'])} | "
            f"{_markdown(record['dataset'])} / {_markdown(record['split'])} | {_label(record['outcome'])} | "
            f"{'Yes' if record['integrated_on_main'] else 'No — PR ' + ', '.join(map(str, record.get('pull_requests', [])))} | "
            f"{_markdown(record['finding'])} | {_reference_markdown(record['evidence_refs'])} | {_markdown('; '.join(record['limitations']))} |"
        )
    lines.extend(
        [
            "",
            "## Minimum record before closing a physical experiment",
            "",
            "Pin source model and dataset revisions and hashes; preserve independent calibration and test identities; record commit, seed, hardware, runtime, commands, resource ceilings, candidate/baseline metrics, uncertainty, artifact hashes, reload validation, quality gates, and limitations. CI validates the record and generated docs; it does not substitute for the full experiment.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_readiness(data: dict[str, Any], scores: dict[str, Any]) -> str:
    stream = _workstream_map(data)["production_readiness"]
    items = [m for m in data["milestones"] if m["workstream"] == stream["id"]]
    lines = [
        "# Production readiness matrix",
        "",
        f"Assessment: {data['assessment_date']} · overall project readiness dimension: **{scores['dimensions']['readiness']:.1f}%** · production-readiness workstream maturity: **{scores['workstreams']['production_readiness']['overall']:.1f}%**.",
        "",
        "A release boundary, UI workflow, mock, schema, CI pass, or local experiment does not establish production optimizer readiness. The optimizer must prove physical quality-preserving results, resource behavior, artifact safety, and reproducibility within its declared support scope.",
        "",
        "| Area | Implementation | Readiness | Effectiveness | Acceptance criteria | Evidence | Remaining work / limitations |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in items:
        lines.append(
            f"| **{_markdown(item['title'])}** (`{item['id']}`) | {_label(item['implementation_status'])} | {_label(item['readiness_status'])} | "
            f"{_label(item['effectiveness_status'])} | {_markdown('; '.join(item['acceptance_criteria']))} | {_reference_markdown(item['evidence_refs'])} | "
            f"{_markdown(item['remaining_work'])}; {_markdown('; '.join(item['limitations']))} |"
        )
    return "\n".join(lines) + "\n"


def render_backlog(data: dict[str, Any]) -> str:
    audit = data["backlog_audit"]
    lines = [
        "# Open pull-request and issue backlog snapshot",
        "",
        f"GitHub CLI snapshot: **{audit['audit_date']}** · assessed `main` `{audit['main_commit']}` · "
        f"{audit['open_issue_count']} open issues · {audit['open_pull_request_count']} open PRs.",
        "",
        audit["scope_note"],
        "",
        "All open PRs were enumerated; notable optimizer/research and stacked-dependency changes are listed below. This is a dated snapshot, not live status. An open PR or green check is not integrated functionality.",
        "",
        "## Stacked dependencies",
        "",
        "Chains are listed in base-first order. Re-fetch branch bases and review conflicts before integration.",
        "",
        "| Chain | Branch relationship |",
        "| --- | --- |",
    ]
    by_number = {item["number"]: item for item in audit["notable_pull_requests"]}
    for chain in audit["dependency_chains"]:
        links = " → ".join(
            f"[#{number}](https://github.com/xdCloudy/ModelSurgeon/pull/{number})"
            for number in chain
        )
        branches = " → ".join(by_number[number]["head_branch"] for number in chain)
        lines.append(f"| {links} | `{_markdown(branches)}` |")
    lines.extend(
        [
            "",
            "## Notable open optimizer and evidence PRs",
            "",
            "| PR | Base | Branch | Area / relationship | Integrated on main |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for item in sorted(audit["notable_pull_requests"], key=lambda row: row["number"]):
        lines.append(
            f"| [#{item['number']}: {_markdown(item['title'])}](https://github.com/xdCloudy/ModelSurgeon/pull/{item['number']}) | "
            f"`{_markdown(item['base_branch'])}` | `{_markdown(item['head_branch'])}` | "
            f"{_markdown(item['relationship'])} | No |"
        )
    lines.extend(
        [
            "",
            "## Triage",
            "",
            _markdown(audit["triage"]),
            "",
            "Re-fetch and re-audit the live backlog before later review or branch integration. Respect normal CI and approval requirements; this inventory does not authorize merging or bypass review.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_progress_overview(data: dict[str, Any], scores: dict[str, Any]) -> str:
    priorities = {m["id"]: m for m in data["milestones"]}
    lines = [
        "# ModelSurgeon progress and scoring",
        "",
        f"Assessment date: **{data['assessment_date']}** · assessed main commit: `{data['assessment_main_commit']}` · inventory scope: `{data['scope_version']}`.",
        "",
        data["scope_note"],
        "Canonical source: [`modelsurgeon-progress-v1.json`](modelsurgeon-progress-v1.json) · [JSON Schema](schema-v1.json) · [maintenance workflow](CONTRIBUTING.md). The dashboard is generated locally; repository edits remain reviewable in the same PR.",
        "",
        f"## Current scores\n\nOverall weighted completion: **{scores['overall']:.1f}%**. This is the defined, versioned scope score, not a release certification.",
        "",
        "| Dimension | Score | Dimension weight in overall | Meaning |",
        "| --- | ---: | ---: | --- |",
        f"| Research coverage | {scores['dimensions']['research']:.1f}% | {DIMENSION_WEIGHTS['research']}% | Progress through prior art, hypotheses, physical experiments, matched baselines, and reproducible conclusions. Negative conclusions count as completed research. |",
        f"| Implementation maturity | {scores['dimensions']['implementation']:.1f}% | {DIMENSION_WEIGHTS['implementation']}% | Progress through tested and integrated software; integrated means present on assessed main. |",
        f"| Optimization effectiveness | {scores['dimensions']['effectiveness']:.1f}% | {DIMENSION_WEIGHTS['effectiveness']}% | Evidence that methods meet quality/resource gates; negative or unmeasured methods score zero. |",
        f"| Production readiness | {scores['dimensions']['readiness']:.1f}% | {DIMENSION_WEIGHTS['readiness']}% | Verification, integration, regression, deployment, and operating readiness within declared scope. |",
        f"| Representative compatibility coverage | {scores['compatibility']:.1f}% | Separate coverage measure | Weighted share of explicit model/format/operation/backend/runtime/recovery cells with evidence-backed support. It is not added a second time to the overall score. |",
        "",
        "## Scoring method",
        "",
        "Each workstream has a positive integer weight; workstream weights total 100. Each workstream's active milestone weights total 100. A milestone contributes only to dimensions listed in `scored_dimensions`. Within a workstream, milestone-weighted dimension scores are combined using its explicit dimension weights (renormalized over applicable dimensions); the overall completion is then the workstream-weighted mean. Research, implementation, effectiveness, and readiness use the ordered score maps defined in `tools/progress.py`. Research completion applies only to the milestone's explicit research question and scope; broader open questions must be separate. Research outcome is not itself a score: an evidence-backed negative conclusion can have 100% research completion and 0% effectiveness. Compatibility coverage is separately reported; `verified` requires a plausible scope, implementation on assessed main, and physical or independent reproduction evidence.",
        "",
        "Newly discovered scope is added as milestones and a dated scope-history entry. It expands the denominator and may reduce scores. A milestone can be retired only with a permanent record in `retired_milestones`; research conclusions and identifiers are not deleted. No unmerged PR contributes to integrated or main-readiness stages.",
        "",
        "## Workstreams",
        "",
        "| Workstream | Weight | Overall | Research | Implementation | Effectiveness | Readiness |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stream in data["workstreams"]:
        values = scores["workstreams"][stream["id"]]
        dims = values["dimensions"]
        lines.append(
            f"| [{_markdown(stream['title'])}](FEATURE_MATRIX.md#{stream['id'].replace('_', '-')}) | {stream['weight']}% | {values['overall']:.1f}% | "
            + " | ".join(f"{dims.get(dim, 0):.1f}%" for dim in DIMENSIONS)
            + " |"
        )
    lines.extend(["", "## Priorities", ""])
    for index, milestone_id in enumerate(data["priority_order"], start=1):
        item = priorities[milestone_id]
        lines.append(
            f"{index}. **{item['title']}** (`{milestone_id}`): {_markdown(item['remaining_work'])}"
        )
    lines.extend(
        [
            "",
            "## Critical-path dependencies",
            "",
            "A dependent milestone remains bounded by every listed prerequisite; opening a PR does not satisfy a dependency.",
            "",
            "| Milestone | Depends on |",
            "| --- | --- |",
        ]
    )
    for item in data["milestones"]:
        if item["dependencies"]:
            deps = ", ".join(f"`{dep}`" for dep in item["dependencies"])
            lines.append(f"| `{item['id']}` — {_markdown(item['title'])} | {deps} |")
    lines.extend(
        ["", "## Current blockers", "", "| Blocker | Impact | Next action |", "| --- | --- | --- |"]
    )
    for blocker in data["blockers"]:
        lines.append(
            f"| **{_markdown(blocker['id'])}** · {_markdown(blocker['description'])} | "
            f"{_markdown(blocker['impact'])} | {_markdown(blocker['next_action'])} |"
        )
    lines.extend(
        [
            "",
            "## Commands",
            "",
            "```bash",
            "uv run python tools/progress.py validate",
            "uv run python tools/progress.py generate",
            "uv run python tools/progress.py check",
            "```",
            "",
            "`generate` writes derived README/matrices only when content changes. `check` is read-only and fails with file-specific stale-output diagnostics; CI runs `validate` and `check` and never commits generated updates.",
        ]
    )
    return "\n".join(lines) + "\n"


def expected_outputs(data: dict[str, Any], root: Path = ROOT) -> dict[Path, str]:
    scores = calculate_progress(data)
    readme = (root / README_PATH).read_text(encoding="utf-8")
    if readme.count(START_MARKER) != 1 or readme.count(END_MARKER) != 1:
        raise ProgressError("README must contain exactly one generated progress marker pair")
    start = readme.index(START_MARKER)
    end = readme.index(END_MARKER)
    if end < start:
        raise ProgressError("README progress markers are out of order")
    rendered_readme = (
        readme[:start] + render_readme_block(data, scores) + readme[end + len(END_MARKER) :]
    )
    outputs = {
        root / README_PATH: rendered_readme,
        root / "docs/progress/README.md": render_progress_overview(data, scores),
        root / "docs/progress/FEATURE_MATRIX.md": render_feature_matrix(data, scores),
        root / "docs/progress/META_SURGEON.md": render_meta_surgeon(data, scores),
        root / "docs/progress/OPTIMIZATION_METHODS.md": render_methods(data),
        root / "docs/progress/MODEL_COMPATIBILITY.md": render_compatibility(data, scores),
        root / "docs/progress/RESEARCH_INVENTORY.md": render_research(data),
        root / "docs/progress/BENCHMARKS.md": render_benchmarks(data),
        root / "docs/progress/PRODUCTION_READINESS.md": render_readiness(data, scores),
        root / "docs/progress/PR_BACKLOG.md": render_backlog(data),
    }
    generated_paths = {path.resolve() for path in outputs}
    for output_path, content in outputs.items():
        base = output_path.parent
        for target in re.findall(r"\]\(([^)]+)\)", content):
            if target.startswith(("https://", "http://", "mailto:", "#")):
                continue
            local_target = target.split("#", 1)[0]
            if not local_target:
                continue
            resolved = (base / Path(*PurePosixPath(local_target).parts)).resolve()
            if not resolved.is_relative_to(root.resolve()) or (
                not resolved.exists() and resolved not in generated_paths
            ):
                raise ProgressError(
                    f"generated documentation link is missing or escapes repository: "
                    f"{output_path.relative_to(root).as_posix()} -> {target}"
                )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "generate", "check"))
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    data_path = root / DATA_PATH
    try:
        data = json.loads(data_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"progress data: cannot read {data_path}: {exc}", file=sys.stderr)
        return 1
    errors = validate(data, root)
    if errors:
        print("Progress validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    scores = calculate_progress(data)
    if args.command == "validate":
        print(
            f"Progress data valid: {len(data['workstreams'])} workstreams, "
            f"{len(data['milestones'])} milestones, {len(data['compatibility_cells'])} compatibility cells; "
            f"overall {scores['overall']:.1f}%."
        )
        return 0
    try:
        outputs = expected_outputs(data, root)
    except (OSError, ProgressError) as exc:
        print(f"Progress rendering failed: {exc}", file=sys.stderr)
        return 1
    stale = [
        path
        for path, content in outputs.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != content
    ]
    if args.command == "check":
        if stale:
            print("Generated progress documentation is stale:", file=sys.stderr)
            for path in stale:
                print(
                    f"- {path.relative_to(root).as_posix()}: run `uv run python tools/progress.py generate` and commit the result",
                    file=sys.stderr,
                )
            return 1
        print(
            f"Generated progress documentation is current ({len(outputs)} files; {scores['overall']:.1f}% overall)."
        )
        return 0
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8", newline="\n")
            print(f"updated {path.relative_to(root).as_posix()}")
    print(f"Generated progress documentation ({scores['overall']:.1f}% overall).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
