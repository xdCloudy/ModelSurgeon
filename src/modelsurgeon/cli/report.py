"""Evidence-backed reports for persisted runs and candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from modelsurgeon.experiments.store import ExperimentMetadataStore, StoredCandidate
from modelsurgeon.explain import (
    GeneratedReport,
    ReportFailure,
    ReportInput,
    ReportLink,
    ReportRedaction,
    generate_report,
)

_DEFAULT_REDACTION = ReportRedaction()


class ReportCommandError(RuntimeError):
    """Raised when a report subject cannot be resolved safely."""


@dataclass(frozen=True, slots=True)
class ResolvedReportSubject:
    kind: str
    subject_id: str
    run_id: str
    candidates: tuple[StoredCandidate, ...]


def _resolve_subject(store: ExperimentMetadataStore, subject_id: str) -> ResolvedReportSubject:
    if subject_id.startswith("run_"):
        run = store.get_run(subject_id)
        if run is None:
            raise ReportCommandError(f"run not found: {subject_id}")
        candidates = store.list_run_candidates(subject_id)
        if not candidates:
            raise ReportCommandError(
                f"run is incomplete: {subject_id} has no persisted candidates"
            )
        return ResolvedReportSubject("run", subject_id, subject_id, candidates)
    if subject_id.startswith("candidate_"):
        candidate = store.get_candidate(subject_id)
        if candidate is None:
            raise ReportCommandError(f"candidate not found: {subject_id}")
        return ResolvedReportSubject("run", subject_id, candidate.run_id, (candidate,))
    if subject_id.startswith(("campaign_", "search_")):
        kind = subject_id.split("_", 1)[0]
        raise ReportCommandError(
            f"{kind} reports are not persisted by this metadata schema; "
            "provide its canonical run_ ID instead"
        )
    raise ReportCommandError(
        "ID must use the canonical run_, candidate_, campaign_, or search_ prefix"
    )


def generate_persisted_report(
    subject_id: str,
    *,
    metadata_path: str | Path,
    redaction: ReportRedaction = _DEFAULT_REDACTION,
) -> GeneratedReport:
    """Resolve stored evidence into one deterministic report without mutating it."""

    metadata_file = Path(metadata_path)
    if not metadata_file.is_file() or metadata_file.is_symlink():
        raise ReportCommandError("metadata path must be an existing regular SQLite file")
    with ExperimentMetadataStore(metadata_file) as store:
        subject = _resolve_subject(store, subject_id)
        run = store.get_run(subject.run_id)
        if run is None:  # pragma: no cover - protected by _resolve_subject
            raise ReportCommandError(f"run not found: {subject.run_id}")
        source = store.get_input(run.input_id)
        if source is None:
            raise ReportCommandError(
                f"run is incomplete: missing input metadata for {subject.run_id}"
            )

        metrics: list[tuple[str, float | None]] = []
        failures: list[ReportFailure] = []
        lineage: list[dict[str, object]] = []
        links: list[ReportLink] = [
            ReportLink("run", "run", run.run_id),
            ReportLink("experiment", "experiment", run.experiment_id),
            ReportLink("input", "input", run.input_id),
        ]
        for candidate in subject.candidates:
            links.append(ReportLink("candidate", "candidate", candidate.candidate_id))
            lineage.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "mutation_id": candidate.mutation_id,
                    "candidate_order": candidate.candidate_order,
                    "affected_components": list(candidate.affected_components),
                }
            )
            for metric in store.list_metrics(candidate.candidate_id):
                metrics.append(
                    (f"{candidate.candidate_id}:{metric.phase.value}:{metric.name}", metric.value)
                )
                if metric.state != "measured":
                    failures.append(
                        ReportFailure(
                            "metric",
                            metric.state,
                            metric.reason
                            or f"{metric.phase.value}:{metric.name} is {metric.state}",
                            metric.state in {"unknown", "skipped"},
                        )
                    )
            for state in store.list_states(candidate.candidate_id):
                if state.state in {"failed", "interrupted", "recoverable_oom"}:
                    failures.append(
                        ReportFailure(
                            "candidate",
                            state.state,
                            state.detail or f"candidate entered {state.state}",
                            state.state != "failed",
                        )
                    )
            for artifact in store.list_artifact_references(candidate.candidate_id):
                links.append(ReportLink(artifact.role, "artifact", artifact.digest))

    report_input = ReportInput(
        subject.kind,
        subject.subject_id,
        None,
        {
            "config_digest": source.config_digest,
            "model": {
                "identifier": source.model_identifier,
                "revision": source.model_revision,
                "family": source.model_family,
                "format": source.model_format,
                "parameter_count": source.model_parameter_count,
                "quantization": source.model_quantization,
            },
            "dataset": {
                "identifier": source.dataset_identifier,
                "revision": source.dataset_revision,
                "split": source.dataset_split,
                "manifest_id": source.dataset_manifest_id,
                "tokenizer": source.tokenizer,
                "tokenizer_revision": source.tokenizer_revision,
            },
            "versions": run.versions,
            "seeds": run.seeds,
            "mutation": run.mutation,
            "outcome": run.outcome,
            "quantization_control": run.quantization_control,
        },
        tuple(lineage),
        tuple(sorted(metrics)),
        (),
        tuple(sorted(failures, key=lambda item: (item.stage, item.code, item.message))),
        run.hardware,
        tuple(sorted(links, key=lambda item: item.immutable_id)),
        redaction,
    )
    return generate_report(report_input)


def _write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(contents)
    except FileExistsError as error:
        raise ReportCommandError(f"output already exists: {path}") from error


def report_command(
    subject_id: Annotated[
        str, typer.Argument(help="Canonical run, campaign, search, or candidate ID")
    ],
    metadata: Annotated[
        Path, typer.Option("--metadata", help="Experiment metadata SQLite database")
    ],
    output: Annotated[
        Path | None, typer.Option("--output", help="Write offline HTML report")
    ] = None,
    json_output: Annotated[
        Path | None, typer.Option("--json-output", help="Write canonical JSON report")
    ] = None,
) -> None:
    """Create deterministic JSON and offline HTML reports from persisted evidence."""

    try:
        report = generate_persisted_report(subject_id, metadata_path=metadata)
        if output is not None:
            _write_text(output, report.html_text)
        if json_output is not None:
            _write_text(json_output, report.json_text)
        typer.echo(report.json_text, nl=False)
    except (OSError, ReportCommandError, ValueError) as error:
        typer.echo(f"report error: {error}", err=True)
        raise typer.Exit(2) from error
