from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelsurgeon.evaluation import (
    DEFAULT_QUANTIZED_BASELINE_PROTOCOL,
    QuantizationArm,
    QuantizedArtifact,
    QuantizedBaselineError,
    QuantizedOutcome,
    render_quantized_baseline_protocol,
)


def test_default_protocol_covers_matched_profiles_and_negative_cells() -> None:
    protocol = DEFAULT_QUANTIZED_BASELINE_PROTOCOL

    assert len(protocol.models) == 2
    assert {model.family for model in protocol.models} == {"llama", "qwen"}
    assert len(protocol.profiles) == 6
    assert len(protocol.cells) == 72
    assert len(protocol.compatibility) == 24
    assert {cell.outcome for cell in protocol.cells} == {QuantizedOutcome.UNSUPPORTED}
    assert sum(
        cell.arm is QuantizationArm.PRUNING_PLUS_QUANTIZATION
        for cell in protocol.cells
    ) == 36


def test_protocol_identity_and_rendering_are_deterministic() -> None:
    first = DEFAULT_QUANTIZED_BASELINE_PROTOCOL
    second = DEFAULT_QUANTIZED_BASELINE_PROTOCOL

    assert first.protocol_id == second.protocol_id
    assert render_quantized_baseline_protocol(first) == render_quantized_baseline_protocol(second)
    payload = json.loads(render_quantized_baseline_protocol())
    assert payload["protocol_id"] == first.protocol_id
    evidence_path = (
        Path(__file__).parents[1]
        / "docs"
        / "research"
        / "v1.1-quantized-baseline-evidence-v1.json"
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["protocol_id"] == first.protocol_id


def test_measured_artifact_requires_reload_and_generation() -> None:
    with pytest.raises(QuantizedBaselineError, match="generation"):
        QuantizedArtifact(100, 80, "a" * 64, True, False)


def test_measured_quantized_cell_requires_a_reloadable_artifact() -> None:
    protocol = DEFAULT_QUANTIZED_BASELINE_PROTOCOL
    template = protocol.cells[0]
    with pytest.raises(QuantizedBaselineError, match="reloadable"):
        type(template)(
            template.model,
            template.arm,
            template.profile,
            template.target_fraction,
            template.seed,
            template.corpus,
            template.budget,
            QuantizedOutcome.MEASURED,
            QuantizedArtifact(100, 80, "a" * 64, False, False),
            template.metrics,
            template.loss,
        )
