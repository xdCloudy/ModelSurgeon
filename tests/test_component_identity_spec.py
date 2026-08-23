"""Contract checks for the normative component identity specification."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "docs" / "design" / "component-identity-spec.md"
VECTORS_PATH = ROOT / "tests" / "fixtures" / "component_id_conformance.json"

REQUIRED_KINDS = {
    "attention_head",
    "embedding",
    "kv_head",
    "mlp_channel",
    "moe_expert",
    "moe_router",
    "normalization",
    "projection",
    "residual_path",
    "transformer_layer",
}

REQUIRED_SECTIONS = {
    "## Text grammar",
    "## Segment types and semantic kinds",
    "## Component sets",
    "## Post-surgery remapping",
    "## Invalid and ambiguous inputs",
    "## Conformance vectors",
}


def load_vectors() -> dict[str, object]:
    return json.loads(VECTORS_PATH.read_text(encoding="utf-8"))


def test_specification_has_required_normative_sections() -> None:
    specification = SPEC_PATH.read_text(encoding="utf-8")

    assert all(section in specification for section in REQUIRED_SECTIONS)


def test_conformance_vectors_are_unique_and_cover_required_kinds() -> None:
    vectors = load_vectors()
    valid = vectors["valid"]
    invalid = vectors["invalid"]
    assert isinstance(valid, list)
    assert isinstance(invalid, list)

    canonical_ids = [entry["canonical"] for entry in valid]
    invalid_ids = [entry["text"] for entry in invalid]
    kinds = {entry["kind"] for entry in valid}

    assert len(canonical_ids) == len(set(canonical_ids))
    assert len(invalid_ids) == len(set(invalid_ids))
    assert kinds >= REQUIRED_KINDS
    assert set(canonical_ids).isdisjoint(invalid_ids)


def test_invalid_vectors_have_actionable_reasons() -> None:
    vectors = load_vectors()
    invalid = vectors["invalid"]
    assert isinstance(invalid, list)

    assert all(entry["reason"].strip() for entry in invalid)
