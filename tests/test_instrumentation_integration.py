"""Golden bounded calibration-to-feature integration coverage."""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import pytest

from modelsurgeon.config import MemoryMode
from modelsurgeon.experiments.memory import (
    MemoryPlanningError,
    OperationMemoryEstimates,
    ResourceCapacity,
    ResourceCeilings,
    ResourceEstimate,
    plan_memory_mode,
)
from modelsurgeon.features import (
    ActivationBatch,
    ActivationSummaryCollector,
    FeatureSampleContext,
    PrecisionProvenance,
    PrecisionSource,
    extract_weight_statistics,
)
from modelsurgeon.graph import ComponentId
from modelsurgeon.instrumentation import ActivationHookManager


class TinyTensor:
    """Minimal deterministic tensor surface consumed by weight statistics."""

    def __init__(self, values: tuple[float, ...], shape: tuple[int, ...]) -> None:
        self._values = values
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"

    def detach(self) -> TinyTensor:
        return self

    def numel(self) -> int:
        return len(self._values)

    def cpu(self) -> TinyTensor:
        return self

    def double(self) -> TinyTensor:
        return self

    def reshape(self, *shape: int) -> TinyTensor:
        del shape
        return self

    def tolist(self) -> list[float]:
        return list(self._values)


@dataclass
class TinyHandle:
    module: TinyModule
    hook: object
    removed: bool = False

    def remove(self) -> None:
        self.removed = True
        self.module.hooks.remove(self.hook)


class TinyModule:
    def __init__(self) -> None:
        self.hooks: list[object] = []
        self.handles: list[TinyHandle] = []

    def register_forward_hook(self, hook: object) -> TinyHandle:
        self.hooks.append(hook)
        handle = TinyHandle(self, hook)
        self.handles.append(handle)
        return handle

    def emit(self, output: object) -> None:
        for hook in tuple(self.hooks):
            hook(self, (), output)  # type: ignore[operator]


class ActivationPayload:
    def __init__(
        self,
        values: tuple[tuple[tuple[float, ...], ...], ...],
    ) -> None:
        self.values = values


class TinyCalibrationModel:
    """One deterministic layer plus one parameter tensor for integration testing."""

    def __init__(self) -> None:
        self.layer = TinyModule()
        self.weight = TinyTensor((1.0, 2.0, 3.0, 4.0), (2, 2))

    def forward_calibration(self) -> None:
        self.layer.emit(
            ActivationPayload(
                (
                    (
                        (1.0, 2.0),
                        (3.0, 4.0),
                    ),
                )
            )
        )


def _memory_estimates() -> OperationMemoryEstimates:
    return OperationMemoryEstimates(
        full=ResourceEstimate(1024, 0, 0),
        tensor=ResourceEstimate(512, 0, 0),
        streaming=ResourceEstimate(128, 0, 0),
    )


def test_tiny_calibration_emits_golden_static_and_activation_features() -> None:
    plan = plan_memory_mode(
        MemoryMode.AUTO,
        _memory_estimates(),
        ResourceCapacity(4096, 0, 4096),
        ResourceCeilings(max_ram_bytes=256, max_vram_bytes=0, max_scratch_bytes=0),
    )
    assert plan.mode is MemoryMode.STREAMING
    assert plan.peak.peak_ram_bytes == 128
    assert plan.effective_capacity.ram_bytes == 256

    model = TinyCalibrationModel()
    weight_id = ComponentId.parse("model.layers.0.mlp.up_proj.weight")
    activation_id = ComponentId.parse("model.layers.0.mlp")
    precision = PrecisionProvenance(
        PrecisionSource.HIGH_PRECISION,
        "float32",
        "float64",
    )
    static_records = extract_weight_statistics(weight_id, model.weight).feature_records()
    activation_collector = ActivationSummaryCollector(activation_id, precision)
    manager = ActivationHookManager({activation_id: model.layer}, (activation_id,))

    payload_reference: weakref.ReferenceType[ActivationPayload]
    with manager:
        model.forward_calibration()
        payload = manager.captures[0].value
        assert isinstance(payload, ActivationPayload)
        payload_reference = weakref.ref(payload)
        activation_collector.update(
            ActivationBatch(
                ("sample-0",),
                payload.values,
                ((True, True),),
            )
        )
        del payload
        assert payload_reference() is not None

    assert manager.active is False
    assert manager.captures == ()
    assert model.layer.hooks == []
    assert all(handle.removed for handle in model.layer.handles)
    assert payload_reference() is None

    context = FeatureSampleContext(
        "fixture/tiny-calibration",
        "fixture-rev-1",
        "validation",
        ("sample-0",),
        "prep-v1",
        "fixture/tokenizer",
        "tokenizer-rev-1",
    )
    activation_records = activation_collector.records(context)
    records = (*static_records, *activation_records)
    by_key = {(str(record.component_id), record.name): record for record in records}

    assert len(static_records) == 12
    assert len(activation_records) == 8
    assert len(records) == 20
    assert by_key[(str(weight_id), "weight_mean")].to_record() == {
        "schema_version": 1,
        "component_id": str(weight_id),
        "name": "weight_mean",
        "kind": "scalar",
        "value": 2.5,
        "dtype": "float64",
        "extractor": "weight_statistics",
        "extractor_version": "1",
        "precision": {
            "source": "high_precision",
            "storage_dtype": "float32",
            "compute_dtype": "float64",
            "quantization": None,
            "codec_version": None,
            "error": None,
        },
        "sample_context": None,
        "metadata": {
            "element_count": 4,
            "shape": "2x2",
            "source_device": "cpu",
        },
    }
    assert by_key[(str(activation_id), "activation_mean")].to_record() == {
        "schema_version": 1,
        "component_id": str(activation_id),
        "name": "activation_mean",
        "kind": "scalar",
        "value": 2.5,
        "dtype": "float64",
        "extractor": "activation_summary",
        "extractor_version": "1",
        "precision": {
            "source": "high_precision",
            "storage_dtype": "float32",
            "compute_dtype": "float64",
            "quantization": None,
            "codec_version": None,
            "error": None,
        },
        "sample_context": {
            "dataset": "fixture/tiny-calibration",
            "revision": "fixture-rev-1",
            "split": "validation",
            "sample_ids": ["sample-0"],
            "preprocessing_version": "prep-v1",
            "tokenizer": "fixture/tokenizer",
            "tokenizer_revision": "tokenizer-rev-1",
        },
        "metadata": {
            "aggregation_axes": "batch,token,feature",
            "feature_width": 2,
            "observation_count": 4,
        },
    }


def test_tiny_calibration_memory_budget_fails_before_execution() -> None:
    with pytest.raises(MemoryPlanningError, match="no permitted memory mode"):
        plan_memory_mode(
            MemoryMode.AUTO,
            _memory_estimates(),
            ResourceCapacity(4096, 0, 4096),
            ResourceCeilings(max_ram_bytes=64, max_vram_bytes=0, max_scratch_bytes=0),
        )
