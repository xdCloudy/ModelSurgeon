"""Tests for single-mutation experiment CLI orchestration and rollback safety."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import pytest
from typer.testing import CliRunner

import modelsurgeon.cli.app as app_module
from modelsurgeon.cli.experiment import (
    ExperimentCommandError,
    ResolvedExperiment,
    parse_mutation_request_json,
    run_single_mutation_experiment,
    write_experiment_result,
)
from modelsurgeon.evaluation.tiered import (
    EscalationAction,
    EvaluationTier,
    TierDecision,
    TieredEvaluationReport,
)
from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.contracts import (
    MutationDelta,
    MutationKind,
    MutationPlan,
    MutationRequest,
    MutationTransaction,
    TransactionState,
)
from modelsurgeon.surgery.serialization import (
    MutationOutcome,
    MutationOutcomeStatus,
    MutationProvenance,
)
from modelsurgeon.surgery.transaction import InMemoryMutationTransaction


class SnapshotTarget:
    def __init__(self) -> None:
        self.restores = 0

    def snapshot(self) -> object:
        return b"original"

    def restore(self, snapshot: object) -> None:
        assert snapshot == b"original"
        self.restores += 1


def _request() -> MutationRequest:
    return MutationRequest(
        MutationKind.MASK,
        (ComponentId.parse("model.layers.0.self_attn"),),
        (("head_index", 1),),
    )


def _plan(request: MutationRequest | None = None) -> MutationPlan:
    resolved = request or _request()
    return MutationPlan(resolved, resolved.targets, (), MutationDelta())


def _evaluation() -> TieredEvaluationReport:
    return TieredEvaluationReport(
        (
            TierDecision(
                EvaluationTier.TIER0,
                True,
                True,
                EscalationAction.COMPLETE,
                None,
                (),
            ),
            TierDecision(
                EvaluationTier.TIER1,
                False,
                None,
                EscalationAction.SKIP,
                "tier not configured",
                (),
            ),
            TierDecision(
                EvaluationTier.TIER2,
                False,
                None,
                EscalationAction.SKIP,
                "tier not configured",
                (),
            ),
            TierDecision(
                EvaluationTier.TIER3,
                False,
                None,
                EscalationAction.SKIP,
                "tier not configured",
                (),
            ),
        ),
        True,
        EvaluationTier.TIER0,
    )


class FakeRuntime:
    def __init__(self, *, interrupt: bool = False) -> None:
        self.interrupt = interrupt
        self.owner = object()
        self.target = SnapshotTarget()
        self.calls: list[str] = []
        self.last_transaction: InMemoryMutationTransaction | None = None

    def resolve(self, request: MutationRequest) -> ResolvedExperiment:
        self.calls.append("resolve")
        return ResolvedExperiment(
            _plan(request),
            MutationProvenance("model-sha", "tool-sha", "/private/model"),
        )

    def transaction(
        self,
        plan: MutationPlan,
    ) -> AbstractContextManager[MutationTransaction]:
        del plan
        self.calls.append("transaction")
        transaction = InMemoryMutationTransaction(
            self.owner,
            {"target": self.target},
            ("target",),
        )
        self.last_transaction = transaction
        return transaction

    def mutation_scope(
        self,
        plan: MutationPlan,
        transaction: MutationTransaction,
    ) -> AbstractContextManager[object]:
        del plan, transaction
        self.calls.append("mutation_scope")
        return nullcontext(object())

    def evaluate(self, plan: MutationPlan) -> TieredEvaluationReport:
        del plan
        self.calls.append("evaluate")
        if self.interrupt:
            raise KeyboardInterrupt
        return _evaluation()

    def rolled_back_outcome(
        self,
        plan: MutationPlan,
        evaluation: TieredEvaluationReport,
    ) -> MutationOutcome:
        del plan, evaluation
        self.calls.append("outcome")
        return MutationOutcome(
            MutationOutcomeStatus.ROLLED_BACK,
            MutationDelta(),
            (),
            "experiment mutation reverted after evaluation",
        )


def _request_payload() -> str:
    return json.dumps(_request().to_record(), sort_keys=True)


def test_parse_mutation_request_canonicalizes_targets_and_parameters() -> None:
    payload = json.dumps(
        {
            "schema_version": 1,
            "kind": "mask",
            "targets": ["model.layers.2", "model.layers.0"],
            "parameters": {"z": 2, "a": 1},
        }
    )
    request = parse_mutation_request_json(payload)
    assert tuple(map(str, request.targets)) == ("model.layers.0", "model.layers.2")
    assert request.parameters == (("a", 1), ("z", 2))

    with pytest.raises(ExperimentCommandError, match="finite"):
        parse_mutation_request_json(
            '{"schema_version":1,"kind":"mask","targets":["model.layers.0"],'
            '"parameters":{"bad":NaN}}'
        )


def test_dry_run_resolves_and_serializes_plan_without_opening_transaction() -> None:
    runtime = FakeRuntime()
    result = run_single_mutation_experiment(_request(), runtime, dry_run=True)

    assert result.dry_run
    assert result.evaluation is None
    assert result.run_record.outcome is None
    assert runtime.calls == ["resolve"]
    record = result.to_record()
    assert record["run"]["mutation_id"] == _request().mutation_id  # type: ignore[index]


def test_experiment_evaluates_inside_transaction_and_rolls_back_afterward() -> None:
    runtime = FakeRuntime()
    result = run_single_mutation_experiment(_request(), runtime)

    assert runtime.calls == [
        "resolve",
        "transaction",
        "mutation_scope",
        "evaluate",
        "outcome",
    ]
    assert runtime.last_transaction is not None
    assert runtime.last_transaction.state is TransactionState.ROLLED_BACK
    assert runtime.target.restores == 1
    assert result.evaluation is not None and result.evaluation.accepted
    assert result.run_record.outcome is not None
    assert result.run_record.outcome.status is MutationOutcomeStatus.ROLLED_BACK


def test_keyboard_interrupt_rolls_back_before_propagating() -> None:
    runtime = FakeRuntime(interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        run_single_mutation_experiment(_request(), runtime)

    assert runtime.last_transaction is not None
    assert runtime.last_transaction.state is TransactionState.ROLLED_BACK
    assert runtime.target.restores == 1
    assert "outcome" not in runtime.calls


def test_cli_dry_run_prints_resolved_structured_plan(tmp_path: Path, monkeypatch) -> None:
    request_path = tmp_path / "mutation.json"
    request_path.write_text(_request_payload(), encoding="utf-8")
    runtime = FakeRuntime()
    monkeypatch.setattr(app_module, "load_experiment_runtime", lambda spec: runtime)

    result = CliRunner().invoke(
        app_module.app,
        [
            "experiment",
            str(request_path),
            "--runtime",
            "fake:factory",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["run"]["plan"]["request"]["kind"] == "mask"
    assert payload["run"]["provenance"]["input_path"] == "<redacted-local-path>"
    assert runtime.calls == ["resolve"]


def test_cli_interrupt_returns_130_after_transaction_rollback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = tmp_path / "mutation.json"
    request_path.write_text(_request_payload(), encoding="utf-8")
    runtime = FakeRuntime(interrupt=True)
    monkeypatch.setattr(app_module, "load_experiment_runtime", lambda spec: runtime)

    result = CliRunner().invoke(
        app_module.app,
        ["experiment", str(request_path), "--runtime", "fake:factory"],
    )

    assert result.exit_code == 130
    assert "transaction rolled back" in result.stderr
    assert runtime.last_transaction is not None
    assert runtime.last_transaction.state is TransactionState.ROLLED_BACK
    assert runtime.target.restores == 1


def test_result_save_is_non_overwriting(tmp_path: Path) -> None:
    result = run_single_mutation_experiment(_request(), FakeRuntime(), dry_run=True)
    destination = tmp_path / "result.json"
    write_experiment_result(destination, result)
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["dry_run"] is True

    with pytest.raises(ExperimentCommandError, match="already exists"):
        write_experiment_result(destination, result)
