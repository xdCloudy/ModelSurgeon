"""Deterministic, approval-visible strategy selection with a safe rule fallback."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum

SELECTOR_SCHEMA_VERSION = 1


class StrategySelectorError(ValueError):
    """Raised when strategy-selection inputs are ambiguous or unsafe."""


class StrategyOutcome(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class EvidenceLevel(StrEnum):
    VERIFIED = "verified"
    EXPERIMENTAL = "experimental"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategySelectorError(f"{label} is required")
    return value


def _finite(value: float, label: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise StrategySelectorError(f"{label} must be finite and non-negative")


def _positive(value: int, label: str) -> None:
    if isinstance(value, bool) or value <= 0:
        raise StrategySelectorError(f"{label} must be positive")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class StrategyOption:
    """One compatible strategy choice with retained capability and cost evidence."""

    name: str
    kind: str
    compatible: bool
    evidence: EvidenceLevel
    evaluation_cost: float
    uncertainty: float
    strong_baseline: bool = False
    requires_approval: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        _text(self.name, "strategy name")
        _text(self.kind, "strategy kind")
        _finite(self.evaluation_cost, "evaluation cost")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise StrategySelectorError("strategy uncertainty must be between zero and one")
        if not self.reason.strip():
            raise StrategySelectorError("strategy options require a capability reason")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "compatible": self.compatible,
            "evidence": self.evidence.value,
            "evaluation_cost": self.evaluation_cost,
            "uncertainty": self.uncertainty,
            "strong_baseline": self.strong_baseline,
            "requires_approval": self.requires_approval,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class StrategySelectionRequest:
    objective_contract_id: str
    candidate_space_id: str
    tool_revision: str
    seed: int
    candidate_count: int
    evaluation_budget: int
    cost_budget: float
    required_baselines: tuple[str, ...]
    options: tuple[StrategyOption, ...]
    override: str | None = None
    override_approved: bool = False

    def __post_init__(self) -> None:
        _text(self.objective_contract_id, "objective contract ID")
        _text(self.candidate_space_id, "candidate-space ID")
        _text(self.tool_revision, "tool revision")
        if isinstance(self.seed, bool) or not 0 <= self.seed < 1 << 64:
            raise StrategySelectorError("selector seed must be an unsigned 64-bit integer")
        _positive(self.candidate_count, "candidate count")
        _positive(self.evaluation_budget, "evaluation budget")
        _finite(self.cost_budget, "cost budget")
        if not self.required_baselines:
            raise StrategySelectorError("at least one strong baseline is required")
        if len(set(self.required_baselines)) != len(self.required_baselines):
            raise StrategySelectorError("required baselines must be unique")
        if not self.options:
            raise StrategySelectorError("strategy selection requires options")
        names = tuple(option.name for option in self.options)
        if len(names) != len(set(names)):
            raise StrategySelectorError("strategy option names must be unique")

    def to_record(self) -> dict[str, object]:
        return {
            "objective_contract_id": self.objective_contract_id,
            "candidate_space_id": self.candidate_space_id,
            "tool_revision": self.tool_revision,
            "seed": self.seed,
            "candidate_count": self.candidate_count,
            "evaluation_budget": self.evaluation_budget,
            "cost_budget": self.cost_budget,
            "required_baselines": list(self.required_baselines),
            "options": [item.to_record() for item in self.options],
            "override": self.override,
            "override_approved": self.override_approved,
        }


@dataclass(frozen=True, slots=True)
class StrategySelection:
    outcome: StrategyOutcome
    selected: tuple[str, ...]
    baseline_names: tuple[str, ...]
    surgeon: str | None
    acquisition: str | None
    search_policy: str | None
    evaluation_budget: int
    estimated_cost: float
    alternatives: tuple[tuple[str, str], ...]
    reasons: tuple[str, ...]
    decision_id: str

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "strategy_selection",
            "schema_version": SELECTOR_SCHEMA_VERSION,
            "decision_id": self.decision_id,
            "outcome": self.outcome.value,
            "selected": list(self.selected),
            "baseline_names": list(self.baseline_names),
            "surgeon": self.surgeon,
            "acquisition": self.acquisition,
            "search_policy": self.search_policy,
            "evaluation_budget": self.evaluation_budget,
            "estimated_cost": self.estimated_cost,
            "alternatives": [
                {"name": name, "reason": reason} for name, reason in self.alternatives
            ],
            "reasons": list(self.reasons),
        }


def _decision_id(request: StrategySelectionRequest, record: dict[str, object]) -> str:
    payload = {
        "schema_version": SELECTOR_SCHEMA_VERSION,
        "request": request.to_record(),
        "result": record,
    }
    return f"strategy_decision_{hashlib.sha256(_canonical(payload).encode()).hexdigest()}"


def _option_score(option: StrategyOption) -> tuple[float, float, str]:
    evidence_penalty = {
        EvidenceLevel.VERIFIED: 0.0,
        EvidenceLevel.EXPERIMENTAL: 0.25,
        EvidenceLevel.UNKNOWN: 1.0,
        EvidenceLevel.UNSUPPORTED: 2.0,
    }[option.evidence]
    return (
        option.evaluation_cost + option.uncertainty + evidence_penalty,
        option.uncertainty,
        option.name,
    )


def select_strategy(request: StrategySelectionRequest) -> StrategySelection:
    """Select compatible strategies without learning from unvalidated evidence."""

    by_name = {option.name: option for option in request.options}
    alternatives: list[tuple[str, str]] = []
    required: list[StrategyOption] = []
    for baseline in request.required_baselines:
        option = by_name.get(baseline)
        if option is None:
            return _refusal(
                request,
                StrategyOutcome.UNSUPPORTED,
                f"required strong baseline is unavailable: {baseline}",
            )
        if not option.strong_baseline:
            return _refusal(
                request,
                StrategyOutcome.FAILED,
                f"required baseline is not marked strong: {baseline}",
            )
        if not option.compatible or option.evidence is EvidenceLevel.UNSUPPORTED:
            return _refusal(
                request,
                StrategyOutcome.UNSUPPORTED,
                f"required baseline is incompatible: {baseline}",
            )
        required.append(option)
    override_option = None if request.override is None else by_name.get(request.override)
    if request.override is not None and override_option is None:
        return _refusal(request, StrategyOutcome.UNSUPPORTED, "override strategy is unavailable")
    if override_option is not None:
        if not request.override_approved:
            return _refusal(request, StrategyOutcome.FAILED, "override requires explicit approval")
        if not override_option.compatible or override_option.evidence is EvidenceLevel.UNSUPPORTED:
            return _refusal(
                request,
                StrategyOutcome.UNSUPPORTED,
                "override strategy is incompatible",
            )
    chosen: list[StrategyOption] = []
    reasons: list[str] = []
    for kind in ("surgeon", "acquisition", "search_policy"):
        candidates = [
            option
            for option in request.options
            if option.kind == kind
            and option.compatible
            and option.evidence is not EvidenceLevel.UNSUPPORTED
        ]
        if not candidates:
            return _refusal(
                request,
                StrategyOutcome.UNKNOWN,
                f"no compatible {kind} strategy evidence",
            )
        if override_option is not None and override_option.kind == kind:
            selected_option = override_option
            reasons.append("explicit override was approved")
        else:
            selected_option = min(candidates, key=_option_score)
            reasons.append(
                "rule-based selection retained because held-out meta-evidence is not validated"
            )
        chosen.append(selected_option)
        reasons.append(
            f"selected {selected_option.name} by deterministic cost/uncertainty ordering"
        )
    total_cost = math.fsum(item.evaluation_cost for item in (*required, *chosen))
    if total_cost > request.cost_budget:
        return _refusal(
            request, StrategyOutcome.FAILED, "required baselines and strategy exceed cost budget"
        )
    evaluation_budget = min(request.evaluation_budget, request.candidate_count)
    selected = tuple(item.name for item in (*required, *chosen))
    for option in request.options:
        if option.name not in selected:
            alternatives.append(
                (
                    option.name,
                    "incompatible"
                    if not option.compatible
                    else "not selected by deterministic ordering",
                )
            )
    result: dict[str, object] = {
        "outcome": StrategyOutcome.SUPPORTED.value,
        "selected": list(selected),
        "evaluation_budget": evaluation_budget,
        "estimated_cost": total_cost,
    }
    decision_id = _decision_id(request, result)
    return StrategySelection(
        StrategyOutcome.SUPPORTED,
        selected,
        tuple(item.name for item in required),
        next((item.name for item in chosen if item.kind == "surgeon"), None),
        next((item.name for item in chosen if item.kind == "acquisition"), None),
        next((item.name for item in chosen if item.kind == "search_policy"), None),
        evaluation_budget,
        total_cost,
        tuple(sorted(alternatives)),
        tuple(reasons),
        decision_id,
    )


def _refusal(
    request: StrategySelectionRequest,
    outcome: StrategyOutcome,
    reason: str,
) -> StrategySelection:
    result: dict[str, object] = {"outcome": outcome.value, "reason": reason}
    return StrategySelection(
        outcome,
        (),
        (),
        None,
        None,
        None,
        0,
        0.0,
        (),
        (reason,),
        _decision_id(request, result),
    )


__all__ = [
    "SELECTOR_SCHEMA_VERSION",
    "EvidenceLevel",
    "StrategyOption",
    "StrategyOutcome",
    "StrategySelection",
    "StrategySelectionRequest",
    "StrategySelectorError",
    "select_strategy",
]
