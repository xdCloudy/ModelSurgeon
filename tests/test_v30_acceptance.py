"""Executable contract checks for the bounded v3.0 acceptance gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.audit_v30_acceptance import V30AcceptanceError, audit_release

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs" / "research" / "v3.0-acceptance-matrix-v1.json"
EVIDENCE = ROOT / "docs" / "research" / "v3.0-acceptance-evidence-v1.json"


def test_checked_in_v30_matrix_and_evidence_are_complete() -> None:
    audit_release(ROOT)


def test_evidence_retains_negative_cells_and_optimizer_boundary() -> None:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    expected = {item["id"]: item["expected_outcome"] for item in matrix["cells"]}
    observed = {item["id"]: item["observed_outcome"] for item in evidence["cells"]}

    assert expected["optimizer_capability"] == "unsupported"
    assert expected["cuda_and_hosted"] == "unsupported"
    assert observed["optimizer_capability"] == "unsupported"
    assert observed["cuda_and_hosted"] == "unsupported"
    assert evidence["failures"] == []
    assert evidence["measurement_worktree_clean"] is True


def test_audit_fails_closed_if_ux_is_promoted_to_optimizer_correctness(tmp_path: Path) -> None:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    matrix["authority_boundary"]["ux_completion_is_optimizer_correctness"] = True
    mutated = tmp_path / "matrix.json"
    mutated.write_text(json.dumps(matrix), encoding="utf-8")

    with pytest.raises(V30AcceptanceError, match="UX/optimizer"):
        audit_release(ROOT, matrix=mutated)


def test_audit_fails_closed_if_a_negative_cell_is_dropped(tmp_path: Path) -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    evidence["cells"] = [item for item in evidence["cells"] if item["id"] != "optimizer_capability"]
    mutated = tmp_path / "evidence.json"
    mutated.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(V30AcceptanceError, match="does not cover every matrix cell"):
        audit_release(ROOT, evidence=mutated)
