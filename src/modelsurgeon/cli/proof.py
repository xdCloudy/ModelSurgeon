"""End-to-end orchestration for producing leakage-safe First Surgeon proof datasets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Protocol, runtime_checkable

import typer

from modelsurgeon.cli.experiment import (
    ExperimentCommandError,
    SingleMutationExperimentResult,
    SingleMutationExperimentRuntime,
    load_experiment_runtime,
    run_single_mutation_experiment,
)
from modelsurgeon.datasets.builder import (
    ExperimentFeatureJoin,
    MutationExampleBuildReport,
    build_mutation_examples,
)
from modelsurgeon.datasets.grouped_splits import (
    GroupedSplitConfig,
    GroupedSplitManifest,
    GroupedSplitMode,
    SplitPartition,
    SplitRatios,
    create_grouped_split,
)
from modelsurgeon.datasets.leakage import LeakageAuditReport, audit_dataset_leakage
from modelsurgeon.experiments.candidates import (
    CandidateEnumerationReport,
    CandidateEnumeratorConfig,
    CandidateFilter,
    CandidateScope,
    MutationCandidate,
    enumerate_mutation_candidates,
)
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.experiments.schema import ExperimentRecord, MutationExampleRecord
from modelsurgeon.features.cache import FeaturePartition
from modelsurgeon.graph import ComponentGraph


class FirstSurgeonProofError(ValueError):
    """Raised when a proof campaign cannot produce a trainable leakage-safe dataset."""


@runtime_checkable
class FirstSurgeonProofRuntime(SingleMutationExperimentRuntime, Protocol):
    """Model-specific hooks required beyond the generic single-experiment runtime."""

    @property
    def component_graph(self) -> ComponentGraph: ...

    @property
    def run_id(self) -> str: ...

    def pre_mutation_feature_partitions(
        self,
        candidate: MutationCandidate,
    ) -> tuple[FeaturePartition, ...]: ...

    def experiment_record(
        self,
        candidate: MutationCandidate,
        result: SingleMutationExperimentResult,
    ) -> ExperimentRecord: ...


@dataclass(frozen=True, slots=True)
class FirstSurgeonProofConfig:
    seed: int = 0
    split_seed: int = 1
    max_candidates: int | None = None
    scopes: tuple[CandidateScope, ...] = tuple(CandidateScope)
    ratios: SplitRatios = field(default_factory=SplitRatios)

    def __post_init__(self) -> None:
        for label, value in (("seed", self.seed), ("split_seed", self.split_seed)):
            if isinstance(value, bool) or value < 0 or value >= 1 << 64:
                raise FirstSurgeonProofError(f"{label} must be an unsigned 64-bit integer")
        if self.max_candidates is not None and (
            isinstance(self.max_candidates, bool) or self.max_candidates <= 0
        ):
            raise FirstSurgeonProofError("max_candidates must be positive when set")
        if not self.scopes or len(self.scopes) != len(set(self.scopes)):
            raise FirstSurgeonProofError("proof scopes must be non-empty and unique")


@dataclass(frozen=True, slots=True)
class FirstSurgeonProofResult:
    enumeration: CandidateEnumerationReport
    build: MutationExampleBuildReport
    split: GroupedSplitManifest
    leakage: LeakageAuditReport

    @property
    def examples(self) -> tuple[MutationExampleRecord, ...]:
        return self.build.examples

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "first_surgeon_proof_campaign",
            "enumeration": self.enumeration.to_record(),
            "dataset_build": self.build.to_record(),
            "split": self.split.to_record(),
            "leakage": self.leakage.to_record(),
        }


def load_first_surgeon_proof_runtime(specification: str) -> FirstSurgeonProofRuntime:
    """Load a proof runtime while preserving the existing runtime factory contract."""

    runtime = load_experiment_runtime(specification)
    if not isinstance(runtime, FirstSurgeonProofRuntime):
        raise FirstSurgeonProofError(
            "proof runtime must also expose component_graph, run_id, "
            "pre_mutation_feature_partitions(), and experiment_record()"
        )
    return runtime


def _validate_join(
    runtime: FirstSurgeonProofRuntime,
    candidate: MutationCandidate,
    result: SingleMutationExperimentResult,
    features: tuple[FeaturePartition, ...],
) -> ExperimentFeatureJoin:
    if not features:
        raise FirstSurgeonProofError(
            f"candidate {candidate.candidate_id} produced no pre-mutation feature partitions"
        )
    record = runtime.experiment_record(candidate, result)
    if record.mutation.plan.request != candidate.request:
        raise FirstSurgeonProofError(
            f"candidate {candidate.candidate_id} experiment record changed the mutation request"
        )
    if record.mutation.mutation_id != candidate.mutation_id:
        raise FirstSurgeonProofError(
            f"candidate {candidate.candidate_id} experiment record changed mutation identity"
        )
    return ExperimentFeatureJoin(record, features)


def run_first_surgeon_proof(
    runtime: FirstSurgeonProofRuntime,
    config: FirstSurgeonProofConfig,
) -> FirstSurgeonProofResult:
    """Execute mask candidates and produce a held-out-component training dataset."""

    enumeration = enumerate_mutation_candidates(
        runtime.component_graph,
        runtime.run_id,
        CandidateEnumeratorConfig(
            seed=config.seed,
            filters=CandidateFilter(scopes=config.scopes),
            max_candidates=config.max_candidates,
        ),
    )
    if not enumeration.candidates:
        raise FirstSurgeonProofError("proof candidate enumeration produced no candidates")

    joins: list[ExperimentFeatureJoin] = []
    for candidate in enumeration.candidates:
        features = runtime.pre_mutation_feature_partitions(candidate)
        result = run_single_mutation_experiment(candidate.request, runtime)
        joins.append(_validate_join(runtime, candidate, result, features))

    build = build_mutation_examples(joins)
    if not build.examples:
        raise FirstSurgeonProofError(
            "proof campaign produced no trainable examples after dataset policy exclusions"
        )
    split = create_grouped_split(
        build.examples,
        GroupedSplitConfig(
            GroupedSplitMode.COMPONENT,
            config.split_seed,
            config.ratios,
        ),
    )
    empty = tuple(
        partition
        for partition, count in split.example_counts.items()
        if count == 0
    )
    if empty:
        names = ", ".join(partition.value for partition in empty)
        raise FirstSurgeonProofError(
            "held-out-component split left empty partitions "
            f"({names}); collect more independent component groups"
        )
    leakage = audit_dataset_leakage(build.examples, split)
    leakage.require_clean()
    return FirstSurgeonProofResult(enumeration, build, split, leakage)


def _write_new(path: Path, payload: str) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise FirstSurgeonProofError(f"proof output already exists: {path}") from error
    except OSError as error:
        raise FirstSurgeonProofError(f"cannot write proof output {path}: {error}") from error


def write_first_surgeon_proof(
    output: Path,
    result: FirstSurgeonProofResult,
) -> dict[str, Path]:
    """Persist canonical dataset, split, leakage, and campaign records without overwriting."""

    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FirstSurgeonProofError(f"proof output directory already exists: {output}") from error
    except OSError as error:
        raise FirstSurgeonProofError(f"cannot create proof output directory: {error}") from error

    examples_path = output / "examples.jsonl"
    split_path = output / "split.json"
    leakage_path = output / "leakage.json"
    campaign_path = output / "campaign.json"
    _write_new(
        examples_path,
        "".join(
            canonical_identity_json(example.to_record()) + "\n"
            for example in result.examples
        ),
    )
    _write_new(split_path, canonical_identity_json(result.split.to_record()) + "\n")
    _write_new(leakage_path, canonical_identity_json(result.leakage.to_record()) + "\n")
    _write_new(campaign_path, canonical_identity_json(result.to_record()) + "\n")
    return {
        "examples": examples_path,
        "split": split_path,
        "leakage": leakage_path,
        "campaign": campaign_path,
    }


def _scope_tuple(values: Sequence[CandidateScope] | None) -> tuple[CandidateScope, ...]:
    if not values:
        return tuple(CandidateScope)
    return tuple(dict.fromkeys(values))


def first_surgeon_proof_command(
    runtime: Annotated[
        str,
        typer.Option(
            "--runtime",
            help="FirstSurgeonProofRuntime factory as module:factory",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="New directory for examples/split/audit records"),
    ],
    seed: Annotated[int, typer.Option("--seed", min=0)] = 0,
    split_seed: Annotated[int, typer.Option("--split-seed", min=0)] = 1,
    max_candidates: Annotated[
        int | None,
        typer.Option("--max-candidates", min=1),
    ] = None,
    scope: Annotated[
        list[CandidateScope] | None,
        typer.Option("--scope", help="Candidate scope; repeat to restrict the proof campaign"),
    ] = None,
    train_ratio: Annotated[float, typer.Option("--train-ratio", min=0.01, max=0.98)] = 0.8,
    validation_ratio: Annotated[
        float,
        typer.Option("--validation-ratio", min=0.01, max=0.98),
    ] = 0.1,
    test_ratio: Annotated[float, typer.Option("--test-ratio", min=0.01, max=0.98)] = 0.1,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the campaign record as JSON"),
    ] = False,
) -> None:
    """Generate the leakage-safe dataset needed for the v0.5 First Surgeon proof."""

    try:
        proof_runtime = load_first_surgeon_proof_runtime(runtime)
        config = FirstSurgeonProofConfig(
            seed=seed,
            split_seed=split_seed,
            max_candidates=max_candidates,
            scopes=_scope_tuple(scope),
            ratios=SplitRatios(train_ratio, validation_ratio, test_ratio),
        )
        result = run_first_surgeon_proof(proof_runtime, config)
        paths = write_first_surgeon_proof(output, result)
        if output_json:
            typer.echo(canonical_identity_json(result.to_record()))
            return
        counts = result.split.example_counts
        typer.echo(
            "first-surgeon proof dataset ready: "
            f"examples={len(result.examples)} "
            f"train={counts[SplitPartition.TRAIN]} "
            f"validation={counts[SplitPartition.VALIDATION]} "
            f"test={counts[SplitPartition.TEST]}"
        )
        typer.echo(f"examples: {paths['examples']}")
        typer.echo(f"split: {paths['split']}")
        typer.echo(f"leakage audit: {paths['leakage']}")
        typer.echo(
            "next: modelsurgeon train-surgeon "
            f"{paths['examples']} --split {paths['split']} "
            f"--registry {output / 'registry'} --target perplexity "
            "--baseline lightgbm-regressor"
        )
    except KeyboardInterrupt:
        typer.echo("first-surgeon proof interrupted; active mutation rolled back", err=True)
        raise typer.Exit(130) from None
    except (ExperimentCommandError, OSError, RuntimeError, ValueError) as error:
        typer.echo(f"first-surgeon-proof error: {error}", err=True)
        raise typer.Exit(2) from error
