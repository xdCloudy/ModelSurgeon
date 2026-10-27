"""Tests for the canonical project-progress inventory and generated docs."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("modelsurgeon_progress", ROOT / "tools/progress.py")
assert SPEC is not None and SPEC.loader is not None
progress = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = progress
SPEC.loader.exec_module(progress)


def load_data() -> dict[str, object]:
    return json.loads(
        (ROOT / "docs/progress/modelsurgeon-progress-v1.json").read_text(encoding="utf-8")
    )


def test_canonical_inventory_and_schema_are_valid() -> None:
    data = load_data()
    assert progress.validate(data, ROOT) == []
    schema = json.loads((ROOT / "docs/progress/schema-v1.json").read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == data["schema_version"]
    assert set(schema["$defs"]["milestone"]["required"]) <= set(data["milestones"][0])


def test_negative_research_does_not_count_as_effectiveness() -> None:
    data = load_data()
    milestone = next(
        item for item in data["milestones"] if item["id"] == "meta_surgeon.calibration"
    )
    assert progress.score_dimension(milestone, "research") == 100
    assert progress.score_dimension(milestone, "effectiveness") == 0


def test_open_pr_work_is_not_marked_as_integrated() -> None:
    data = load_data()
    moe = next(item for item in data["milestones"] if item["id"] == "moe.surgery")
    assert moe["related_prs"] == [736, 738]
    assert moe["implementation_status"] == "not_implemented"
    assert moe["readiness_status"] == "not_assessed"


def test_validator_catches_duplicate_ids_bad_weights_and_broken_evidence() -> None:
    data = copy.deepcopy(load_data())
    data["milestones"][1]["id"] = data["milestones"][0]["id"]
    data["milestones"][0]["evidence_refs"].append({"path": "docs/progress/no-such-evidence.json"})
    errors = progress.validate(data, ROOT)
    assert any("milestone identifiers must be unique" in error for error in errors)
    assert any("no-such-evidence.json" in error for error in errors)

    data = copy.deepcopy(load_data())
    data["workstreams"][0]["weight"] += 1
    errors = progress.validate(data, ROOT)
    assert any("workstream weights" in error for error in errors)


def test_generated_documents_match_canonical_source_and_check_is_read_only() -> None:
    data = load_data()
    expected = progress.expected_outputs(data, ROOT)
    assert all(
        path.is_file() and path.read_text(encoding="utf-8") == content
        for path, content in expected.items()
    )
    mtimes_before = {path: path.stat().st_mtime_ns for path in expected}
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/progress.py"), "check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "is current" in result.stdout
    assert mtimes_before == {path: path.stat().st_mtime_ns for path in expected}


def test_dependencies_must_be_acyclic() -> None:
    data = copy.deepcopy(load_data())
    data["milestones"][0]["dependencies"] = [data["milestones"][1]["id"]]
    data["milestones"][1]["dependencies"] = [data["milestones"][0]["id"]]
    assert any("dependency cycle" in error for error in progress.validate(data, ROOT))


def test_narrowed_research_conclusions_preserve_prior_evidence() -> None:
    data = load_data()
    milestones = {item["id"]: item for item in data["milestones"]}

    # Stable ids survive subdivision, and the former negative conclusion remains in history.
    assert {"meta_surgeon.decision", "meta_surgeon.transfer", "meta_surgeon.calibration",
            "meta_surgeon.active_learning", "meta_surgeon.ood", "meta_surgeon.continual",
            "meta_surgeon.autonomy"} <= set(milestones)
    assert milestones["pruning.scores"]["research_outcome"] == "inconclusive"
    assert any(
        event["research_outcome"] == "negative"
        for event in milestones["pruning.scores"]["history"]
    )
    assert all(item["research_questions"] for item in data["milestones"])
    assert "candidate-level score evaluations" in milestones["pruning.scores"]["scope"]
    assert "offline replay" in milestones["search.controls"]["scope"]
    assert "Bayesian/evolutionary search" in milestones["search.controls"]["scope"]


def test_compatibility_separates_theory_implementation_and_validation() -> None:
    data = load_data()
    cells = {item["id"]: item for item in data["compatibility_cells"]}
    assert {
        "model_type",
        "hardware_backend",
        "runtime",
        "recovery_method",
        "theoretical_status",
        "implementation_level",
        "validation_level",
    } <= set(cells["hf-llama-mlp"])
    assert cells["gguf-qwen35-moe-hope"]["validation_level"] == "not_tested"
    assert cells["gguf-qwen35-moe-reap-cpu-llamacpp"]["implementation_level"] == "in_pr"

    data["compatibility_cells"][0]["validation_level"] = "synthetic"
    errors = progress.validate(data, ROOT)
    assert any("verified compatibility cell" in error for error in errors)


def test_research_sources_pin_revisions_and_preserve_license_boundaries() -> None:
    data = load_data()
    sources = {item["id"]: item for item in data["research_sources"]}
    for source_id in ("slice-gpt", "prunenet", "hope", "moep", "short-gpt"):
        assert sources[source_id]["pinned_revision"]
        assert sources[source_id]["source_paths"]
        assert (
            sources[source_id]["history"][-1]["pinned_revision"]
            == sources[source_id]["pinned_revision"]
        )

    assert sources["hope"]["license_spdx"] == "Apache-2.0"
    assert sources["moep"]["license_spdx"] == "MIT"
    assert sources["short-gpt"]["licence_status"] == "unknown"
    assert "do not incorporate" in sources["short-gpt"]["licence_note"].lower()
    assert "not independently reproduced" in sources["moep"]["findings"].lower()


def test_research_questions_and_scope_are_required() -> None:
    data = load_data()
    data["milestones"][0]["research_questions"] = []
    errors = progress.validate(data, ROOT)
    assert any("specific research questions" in error for error in errors)


def test_readme_percentages_are_canonical_and_regenerated() -> None:
    data = load_data()
    scores = progress.calculate_progress(data)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{scores['overall']:.1f}%" in readme
    assert "**Overall completion:** `" in readme
    assert "**Model compatibility:**" in readme
