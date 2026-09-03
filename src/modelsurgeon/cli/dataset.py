"""Campaign-to-dataset command with validated, leakage-safe JSONL output."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Protocol, runtime_checkable

import typer

from modelsurgeon.datasets.builder import ExperimentFeatureJoin, build_mutation_examples
from modelsurgeon.datasets.grouped_splits import (
    GroupedSplitConfig,
    GroupedSplitMode,
    SplitPartition,
    create_grouped_split,
)
from modelsurgeon.datasets.validation import validate_mutation_dataset
from modelsurgeon.experiments.campaign import CampaignProgress
from modelsurgeon.experiments.identity import canonical_identity_json

DATASET_COMMAND_SCHEMA_VERSION = 1


class DatasetCommandError(RuntimeError):
    """Raised when a campaign cannot produce a valid persisted dataset."""


@dataclass(frozen=True, slots=True)
class DatasetCampaignResult:
    progress: CampaignProgress
    joins: tuple[ExperimentFeatureJoin, ...]


@runtime_checkable
class DatasetRuntime(Protocol):
    """Trusted runtime that starts or resumes one persisted mutation campaign."""

    def run(self, *, resume: bool) -> DatasetCampaignResult: ...


def load_dataset_runtime(specification: str) -> DatasetRuntime:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise DatasetCommandError("runtime must use module:factory syntax")
    try:
        runtime = getattr(importlib.import_module(module_name), attribute)()
    except Exception as error:
        raise DatasetCommandError(f"cannot load dataset runtime {specification!r}") from error
    if not isinstance(runtime, DatasetRuntime):
        raise DatasetCommandError("dataset runtime does not implement run(resume=...)")
    return runtime


def generate_dataset(
    runtime: DatasetRuntime,
    *,
    output: Path,
    resume: bool,
    split_mode: GroupedSplitMode,
    split_seed: int,
) -> dict[str, object]:
    """Run a campaign, build examples, and expose validated split JSONL files."""

    if output.exists() or output.is_symlink():
        raise DatasetCommandError(f"output already exists: {output}")
    execution = runtime.run(resume=resume)
    built = build_mutation_examples(execution.joins)
    validation = validate_mutation_dataset(built.examples)
    if not validation.valid:
        raise DatasetCommandError("built dataset failed validation")
    split = create_grouped_split(
        built.examples,
        GroupedSplitConfig(split_mode, split_seed),
    )
    by_id = {example.example_id: example for example in built.examples}
    output.mkdir(parents=True)
    for partition in SplitPartition:
        examples = [by_id[item] for item in split.example_ids(partition)]
        (output / f"{partition.value}.jsonl").write_text(
            "".join(canonical_identity_json(item.to_record()) + "\n" for item in examples),
            encoding="utf-8",
            newline="\n",
        )
    record = {
        "schema_version": DATASET_COMMAND_SCHEMA_VERSION,
        "record_type": "generated_dataset",
        "campaign_progress": execution.progress.to_record(),
        "resumed": resume,
        "build": built.to_record(),
        "validation": validation.to_record(),
        "split": split.to_record(),
        "partitions": {partition.value: f"{partition.value}.jsonl" for partition in SplitPartition},
    }
    (output / "manifest.json").write_text(
        canonical_identity_json(record) + "\n", encoding="utf-8", newline="\n"
    )
    return record


def generate_dataset_command(
    runtime: Annotated[
        str, typer.Option("--runtime", help="Trusted campaign runtime as module:factory")
    ],
    output: Annotated[Path, typer.Option("--output", help="New directory for split JSONL dataset")],
    resume: Annotated[
        bool, typer.Option("--resume", help="Resume the campaign before building")
    ] = False,
    split_mode: Annotated[
        GroupedSplitMode, typer.Option("--split-mode", help="Leakage grouping policy")
    ] = GroupedSplitMode.COMPONENT,
    split_seed: Annotated[int, typer.Option("--split-seed", help="Deterministic split seed")] = 0,
) -> None:
    """Start/resume a campaign and write validated leakage-safe dataset partitions."""

    try:
        record = generate_dataset(
            load_dataset_runtime(runtime),
            output=output,
            resume=resume,
            split_mode=split_mode,
            split_seed=split_seed,
        )
        typer.echo(canonical_identity_json(record))
    except (DatasetCommandError, OSError, ValueError) as error:
        typer.echo(f"generate-dataset error: {error}", err=True)
        raise typer.Exit(2) from error
