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
    assert {
        "meta_surgeon.decision",
        "meta_surgeon.transfer",
        "meta_surgeon.calibration",
        "meta_surgeon.active_learning",
        "meta_surgeon.ood",
        "meta_surgeon.continual",
        "meta_surgeon.autonomy",
    } <= set(milestones)
    assert milestones["pruning.scores"]["research_outcome"] == "inconclusive"
    assert any(
        event["research_outcome"] == "negative" for event in milestones["pruning.scores"]["history"]
    )
    assert all(item["research_questions"] for item in data["milestones"])
    assert "candidate-level score evaluations" in milestones["pruning.scores"]["scope"]
    assert "offline replay" in milestones["search.controls"]["scope"]
    assert "Bayesian/evolutionary search" in milestones["search.controls"]["scope"]


def test_bounded_shortgpt_negative_result_is_not_effectiveness_or_integration_credit() -> None:
    data = load_data()
    milestone = next(item for item in data["milestones"] if item["id"] == "pruning.depth_shortgpt")
    source = next(item for item in data["research_sources"] if item["id"] == "shortgpt-mit")

    assert milestone["research_status"] == "research_concluded"
    assert milestone["research_outcome"] == "negative"
    assert progress.score_dimension(milestone, "research") == 100
    assert progress.score_dimension(milestone, "effectiveness") == 0
    assert milestone["implementation_status"] == "experimental"
    assert milestone["readiness_status"] == "experimental"
    assert milestone["related_prs"] == [735]
    assert milestone["related_commits"] == ["61451985128b3c72a433300f69aa5e7a471d79c6"]
    assert source["pinned_revision"] == "78d9615fdcae6d90368832bd0a86c49c323549b9"
    assert source["license_spdx"] == "MIT"
    assert "main-branch support" in milestone["scope"]
    assert "quality gate" in " ".join(milestone["limitations"])


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
    assert cells["hf-deepseek-moe-ddsp"]["status"] == "research_only"
    assert cells["hf-deepseek-moe-ddsp"]["implementation_level"] == "not_implemented"
    assert cells["hf-deepseek-moe-ddsp"]["validation_level"] == "not_tested"

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


def test_added_methods_and_meta_surgeon_questions_are_independently_scoped() -> None:
    data = load_data()
    milestones = {item["id"]: item for item in data["milestones"]}
    expected = {
        "meta_surgeon.dataset_generation",
        "meta_surgeon.representation",
        "meta_surgeon.features_static",
        "meta_surgeon.features_activation",
        "meta_surgeon.features_gradient",
        "meta_surgeon.predictor_selection",
        "meta_surgeon.resource_prediction",
        "meta_surgeon.prospective_acquisition",
        "meta_surgeon.prospective_effectiveness",
        "pruning.unstructured_baselines",
        "pruning.structured_baselines",
        "pruning.dependency_methods",
        "pruning.differentiable_masks",
        "pruning.depth_regularization",
        "quantization.gsq_rco",
        "moe.expert_merging",
    }
    assert expected <= set(milestones)
    assert all(len(milestones[item]["research_questions"]) == 1 for item in expected)
    assert all(milestones[item]["acceptance_criteria"] for item in expected)
    assert (
        sum(item["weight"] for item in data["milestones"] if item["workstream"] == "meta_surgeon")
        == 100
    )
    assert (
        sum(item["weight"] for item in data["milestones"] if item["workstream"] == "pruning") == 100
    )
    assert (
        sum(item["weight"] for item in data["milestones"] if item["workstream"] == "quantization")
        == 100
    )
    assert sum(item["weight"] for item in data["milestones"] if item["workstream"] == "moe") == 100

    sources = {item["id"]: item for item in data["research_sources"]}
    assert (
        sources["deterministic-differentiable-pruning"]["pinned_revision"]
        == "a373238bccb8b16feb00e9e17a9109cb382bc6ce"
    )
    assert sources["deterministic-differentiable-pruning"]["licence_status"] == "unknown"
    assert (
        sources["trsp-layer-regularization"]["pinned_revision"]
        == "9fb34351da7342f43ef060e5d92804f45f46591d"
    )
    assert sources["trsp-layer-regularization"]["licence_status"] == "unknown"
    assert (
        milestones["meta_surgeon.prospective_effectiveness"]["effectiveness_status"]
        == "not_assessed"
    )
    q3 = milestones["meta_surgeon.features_gradient"]
    assert q3["research_outcome"] == "positive"
    assert q3["effectiveness_status"] == "measured_inconclusive"
    assert {
        "meta_surgeon.data",
        "meta_surgeon.features_static",
        "meta_surgeon.features_activation",
        "meta_surgeon.features_gradient",
        "search.controls",
        "benchmarking.statistics",
    } == set(milestones["meta_surgeon.decision"]["dependencies"])
    retired = next(
        item for item in data["retired_milestones"] if item["id"] == "meta_surgeon.features"
    )
    assert retired["archived_record"]["research_outcome"] == "inconclusive"
    overview = progress.render_progress_overview(data, progress.calculate_progress(data))
    assert "Retired scope identifiers" in overview
    assert "meta_surgeon.features_gradient" in overview
    assert progress.validate(data, ROOT) == []


def test_backlog_records_merged_progress_foundation_without_counting_it_open() -> None:
    data = load_data()
    audit = data["backlog_audit"]
    assert audit["open_pull_request_count"] == 61
    foundation = next(item for item in audit["notable_pull_requests"] if item["number"] == 774)
    assert foundation["status"] == "merged"
    assert foundation["integrated_on_main"] is True
    assert foundation["merge_commit"] == audit["main_commit"]
    progress_parent = next(item for item in audit["notable_pull_requests"] if item["number"] == 776)
    structural_study = next(
        item for item in audit["notable_pull_requests"] if item["number"] == 777
    )
    assert progress_parent["status"] == "open"
    assert progress_parent["integrated_on_main"] is False
    assert structural_study["status"] == "open"
    assert structural_study["base_branch"] == progress_parent["head_branch"]
    assert structural_study["integrated_on_main"] is False
    reap_evidence = next(item for item in audit["notable_pull_requests"] if item["number"] == 778)
    assert reap_evidence["status"] == "open"
    assert reap_evidence["base_branch"] == structural_study["head_branch"]
    assert reap_evidence["integrated_on_main"] is False
    gradient_depth = next(item for item in audit["notable_pull_requests"] if item["number"] == 779)
    activation_baseline = next(
        item for item in audit["notable_pull_requests"] if item["number"] == 735
    )
    assert gradient_depth["status"] == "open"
    assert gradient_depth["base_branch"] == activation_baseline["head_branch"]
    assert gradient_depth["integrated_on_main"] is False
    kfac_physical = next(item for item in audit["notable_pull_requests"] if item["number"] == 780)
    assert kfac_physical["status"] == "open"
    assert kfac_physical["base_branch"] == "codex/kfac-structured-sensitivity"
    assert kfac_physical["integrated_on_main"] is False
    assert [775, 780] in audit["dependency_chains"]
    assert "| Yes |" in progress.render_backlog(data)
    assert progress.validate(data, ROOT) == []

    malformed = copy.deepcopy(data)
    next(
        item
        for item in malformed["backlog_audit"]["notable_pull_requests"]
        if item["number"] == 774
    ).pop("merge_commit")
    errors = progress.validate(malformed, ROOT)
    assert any("merged PR #774 needs a full merge commit" in error for error in errors)


def test_readme_percentages_are_canonical_and_regenerated() -> None:
    data = load_data()
    scores = progress.calculate_progress(data)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{scores['overall']:.1f}%" in readme
    assert "**Overall completion:** `" in readme
    assert "**Model compatibility:**" in readme


def test_reap_profile_stability_is_bounded_and_not_quality_effectiveness() -> None:
    data = load_data()
    milestone = next(
        item for item in data["milestones"] if item["id"] == "moe.reap_profile_stability_qwen35"
    )

    assert milestone["research_status"] == "research_concluded"
    assert milestone["research_outcome"] == "positive"
    assert milestone["effectiveness_status"] == "measured_inconclusive"
    assert milestone["related_prs"] == [738]
    assert "three calibration profiles" in " ".join(milestone["limitations"])
    assert "No candidate artifact" in " ".join(milestone["limitations"])
