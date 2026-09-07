"""Automatic capability and candidate-space planner tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.evaluation.architecture_compatibility import (
    ArchitectureOperation,
    ArchitectureProfile,
)
from modelsurgeon.search.auto_space import (
    AutoCandidateSpaceRequest,
    AutoHardwareProfile,
    AutoSpaceError,
    AutoSpaceOutcome,
    ExclusionStatus,
    ModelCapabilityInput,
    build_auto_candidate_space,
)
from modelsurgeon.search.candidate_space import ArchitectureChoice, AxisDomain
from modelsurgeon.search.deployable_state import ArchitectureAxis


def _request(**overrides: object) -> AutoCandidateSpaceRequest:
    model = ModelCapabilityInput(
        "tiny/model",
        "model-revision",
        "state_" + "a" * 64,
        ModelFamily.LLAMA,
        ArchitectureProfile.HF_DENSE,
        1_000,
        100,
        "runtime-revision",
    )
    hardware = AutoHardwareProfile("cpu-small", "hardware-revision", 8_000, None, 10_000)
    values: dict[str, object] = {
        "model": model,
        "hardware": hardware,
        "objective_revision": "objective-revision",
        "tool_revision": "tool-revision",
        "seed": 7,
        "domains": (
            AxisDomain(
                ArchitectureAxis.DEPTH,
                (ArchitectureChoice(1, 10, 100), ArchitectureChoice(2, 20, 200)),
            ),
            AxisDomain(
                ArchitectureAxis.SPARSITY,
                (ArchitectureChoice(0, 10, 100), ArchitectureChoice(1, 5, 50)),
            ),
        ),
        "objective_metrics": ("quality", "latency"),
    }
    values.update(overrides)
    return AutoCandidateSpaceRequest(**values)  # type: ignore[arg-type]


def test_supported_plan_is_deterministic_and_bounded() -> None:
    first = build_auto_candidate_space(_request())
    second = build_auto_candidate_space(_request())
    assert first.outcome is AutoSpaceOutcome.SUPPORTED
    assert first.space is not None
    assert first.plan_id == second.plan_id
    assert first.candidate_upper_bound == 4
    assert first.evaluation_upper_bound == 4
    assert first.artifact_upper_bound_bytes == 200
    assert first.space.generate().space_id == first.space.space_id


def test_unknown_architecture_or_runtime_refuses_total_space() -> None:
    unknown_architecture = build_auto_candidate_space(
        _request(model=replace(_request().model, family=None))
    )
    assert unknown_architecture.outcome is AutoSpaceOutcome.UNKNOWN
    assert unknown_architecture.space is None
    assert unknown_architecture.exclusions[0].status is ExclusionStatus.UNKNOWN

    unknown_runtime = build_auto_candidate_space(
        _request(model=replace(_request().model, runtime_revision=None))
    )
    assert unknown_runtime.outcome is AutoSpaceOutcome.UNKNOWN
    assert any(item.key == "runtime" for item in unknown_runtime.exclusions)


def test_unsupported_plugins_and_codecs_are_explicit() -> None:
    plugin_plan = build_auto_candidate_space(
        _request(required_plugins=("missing-plugin",), available_plugins=())
    )
    assert plugin_plan.outcome is AutoSpaceOutcome.UNSUPPORTED
    assert plugin_plan.space is None
    assert plugin_plan.exclusions[0].key == "plugin:missing-plugin"

    codec_plan = build_auto_candidate_space(
        _request(
            model=replace(
                _request().model,
                profile=ArchitectureProfile.GGUF_F16,
            ),
            required_operations=(ArchitectureOperation.MLP_MASK,),
        )
    )
    assert codec_plan.outcome is AutoSpaceOutcome.UNSUPPORTED
    assert any(item.key == "operation:MLP mask" for item in codec_plan.exclusions)


def test_incomplete_coupling_and_resource_overflow_fail_closed() -> None:
    with pytest.raises(AutoSpaceError, match="candidate cap"):
        replace(_request(), max_candidates=100_001)
    coupling_plan = build_auto_candidate_space(
        _request(coupling_axes=((ArchitectureAxis.DEPTH, ArchitectureAxis.QUANTIZATION),))
    )
    assert coupling_plan.outcome is AutoSpaceOutcome.FAILED
    assert coupling_plan.exclusions[0].status is ExclusionStatus.FAILED

    overflow_plan = build_auto_candidate_space(
        _request(
            hardware=AutoHardwareProfile("tiny-disk", "rev", 8_000, None, 50),
        )
    )
    assert overflow_plan.outcome is AutoSpaceOutcome.FAILED
    assert any(item.key == "hardware:disk" for item in overflow_plan.exclusions)
