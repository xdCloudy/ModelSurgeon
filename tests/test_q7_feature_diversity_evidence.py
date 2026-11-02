"""Integrity checks for the committed Q7 feature-diversity replay evidence."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_q7_feature_diversity_report_declares_its_git_normalized_json_digest() -> None:
    report = ROOT / "docs" / "research" / "v0.8-q7-multi-model-active-learning-v2.json"
    evidence = ROOT / "docs" / "research" / "v0.8-q7-feature-diversity-evidence.md"

    match = re.search(
        r"Git-normalized \(LF\) SHA-256 `([0-9a-f]{64})`",
        evidence.read_text(encoding="utf-8"),
    )

    assert match is not None
    normalized_bytes = report.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(normalized_bytes).hexdigest() == match.group(1)
