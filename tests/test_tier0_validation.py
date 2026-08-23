"""Tests for Tier 0 load, graph, shape, and bounded-forward validation."""

from __future__ import annotations

import pytest

from modelsurgeon.evaluation.tier0 import (
    Tier0Stage,
    Tier0ValidationConfig,
    run_tier0_validation,
)


class TinyCPUBackend:
    device = "cpu"

    def __init__(self, fail_stage: Tier0Stage | None = None) -> None:
        self.fail_stage = fail_stage
        self.forward_budget: int | None = None
        self.calls: list[Tier0Stage] = []

    def _fail(self, stage: Tier0Stage) -> None:
        self.calls.append(stage)
        if self.fail_stage is stage:
            raise RuntimeError(f"{stage.value} failed")

    def load(self) -> object:
        self._fail(Tier0Stage.LOAD)
        return {"weights": ((1.0, 2.0), (3.0, 4.0))}

    def validate_graph(self, model: object) -> None:
        assert model is not None
        self._fail(Tier0Stage.GRAPH)

    def validate_shapes(self, model: object) -> None:
        assert model is not None
        self._fail(Tier0Stage.SHAPES)

    def forward(self, model: object, max_tokens: int) -> object:
        assert model is not None
        self._fail(Tier0Stage.FORWARD)
        self.forward_budget = max_tokens
        return ((0.1, 0.2),)


def test_cpu_tiny_fixture_passes_all_stages_with_forward_budget() -> None:
    backend = TinyCPUBackend()
    result = run_tier0_validation(backend, Tier0ValidationConfig(max_forward_tokens=7))

    assert result.passed
    assert result.completed_stages == tuple(Tier0Stage)
    assert result.failure_stage is None
    assert result.device == "cpu"
    assert backend.forward_budget == 7
    assert backend.calls == list(Tier0Stage)
    assert result.to_record()["completed_stages"] == [stage.value for stage in Tier0Stage]


@pytest.mark.parametrize("stage", list(Tier0Stage))
def test_first_failure_is_classified_and_later_stages_do_not_run(stage: Tier0Stage) -> None:
    backend = TinyCPUBackend(stage)
    result = run_tier0_validation(backend)

    assert not result.passed
    assert result.failure_stage is stage
    assert result.failure_type == "RuntimeError"
    assert result.failure_message == f"{stage.value} failed"
    assert backend.calls[-1] is stage
    assert backend.calls == list(Tier0Stage)[: list(Tier0Stage).index(stage) + 1]


def test_load_returning_none_is_classified_as_load_failure() -> None:
    class EmptyLoader(TinyCPUBackend):
        def load(self) -> object:
            self.calls.append(Tier0Stage.LOAD)
            return None

    result = run_tier0_validation(EmptyLoader())
    assert not result.passed
    assert result.failure_stage is Tier0Stage.LOAD
    assert result.failure_type == "ValueError"
    assert "no model" in (result.failure_message or "")


def test_forward_budget_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        Tier0ValidationConfig(max_forward_tokens=0)
