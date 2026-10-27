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
    milestone = next(item for item in data["milestones"] if item["id"] == "meta_surgeon.decision")
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
