"""Integrity and claim-link tests for the v1.0 scientific release report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "docs" / "research" / "v1.0-scientific-results.json"
REPORT = ROOT / "docs" / "research" / "v1.0-scientific-results.md"


def test_scientific_manifest_hashes_and_claim_links_are_complete() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report = REPORT.read_text(encoding="utf-8")
    artifacts = {item["run_id"]: item for item in manifest["artifacts"]}

    assert manifest["record_type"] == "v1.0_scientific_results_manifest"
    assert manifest["limitation_categories"] == [
        "prediction",
        "masking",
        "physical-surgery",
        "quantization",
        "hardware",
        "reproducibility",
    ]
    for run_id, artifact in artifacts.items():
        assert run_id in report
        evidence = ROOT / "docs" / "research" / artifact["artifact"]
        evidence_record = ROOT / "docs" / "research" / artifact["evidence"]
        assert evidence.exists()
        assert evidence_record.exists()
        if artifact["sha256"] is not None:
            digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
            assert digest == artifact["sha256"]

    for claim in manifest["claims"]:
        assert claim["run_ids"]
        assert all(run_id in artifacts for run_id in claim["run_ids"])
        assert claim["claim"]

    for heading in (
        "### Prediction",
        "### Masking",
        "### Physical surgery",
        "### Quantization",
        "### Hardware and reproducibility",
    ):
        assert heading in report
