"""Tests for the v1.0 release audit contract."""

from pathlib import Path

from tools.audit_v1_release import audit_release

ROOT = Path(__file__).resolve().parents[1]


def test_v1_release_contract_is_complete() -> None:
    audit_release(ROOT, version="v1.0.0")


def test_v1_release_audit_lists_the_declared_boundaries() -> None:
    text = (ROOT / "docs" / "release" / "v1.0-release-audit.md").read_text(encoding="utf-8")
    for phrase in (
        "Hugging Face dense Llama/SmolLM2",
        "Native GGUF Q4_K_M Llama",
        "Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, and Q8_0",
        "Missing external models, datasets, or tools",
    ):
        assert phrase in text
