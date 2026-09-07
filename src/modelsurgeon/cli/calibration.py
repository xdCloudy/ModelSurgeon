"""Plan, execute, and safely cache one Hugging Face calibration manifest."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from modelsurgeon.datasets.calibration import (
    SELECTION_ALGORITHM,
    CalibrationContract,
    CalibrationSample,
    DatasetIdentity,
    DatasetTrust,
    PreprocessingIdentity,
    SelectionConfig,
    TokenizerIdentity,
)
from modelsurgeon.datasets.huggingface import (
    CalibrationManifest,
    HuggingFaceCalibrationRequest,
    stream_huggingface_calibration,
)
from modelsurgeon.experiments.identity import canonical_identity_json

CALIBRATION_COMMAND_SCHEMA_VERSION = 1


class CalibrationCommandError(RuntimeError):
    """Raised when a calibration plan or cache cannot satisfy its contract."""


MetadataPrimitive = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class CalibrationPlan:
    request: HuggingFaceCalibrationRequest


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CalibrationCommandError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CalibrationCommandError(f"{label} must be non-empty text")
    return value


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CalibrationCommandError(f"{label} must be a positive integer")
    return value


def _non_negative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CalibrationCommandError(f"{label} must be a non-negative integer")
    return value


def _configuration_digest(value: object, label: str) -> str:
    configuration = _object(value, label)
    try:
        payload = canonical_identity_json(configuration)
    except ValueError as error:
        raise CalibrationCommandError(f"{label} is not canonical JSON") from error
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _metadata(value: object) -> tuple[tuple[str, MetadataPrimitive], ...]:
    if value is None:
        return ()
    record = _object(value, "dataset metadata")
    result: list[tuple[str, MetadataPrimitive]] = []
    for key, item in record.items():
        if not key or (
            not isinstance(item, (str, int, float, bool)) and item is not None
        ):
            raise CalibrationCommandError(
                "dataset metadata values must be JSON scalar values"
            )
        result.append((key, item))
    return tuple(result)


def _strict_fields(record: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(record) != expected:
        raise CalibrationCommandError(f"{label} has missing or unknown fields")


def _input_ids(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise CalibrationCommandError(f"{label} are invalid")
    return value


def _sample_identity(record: dict[str, object], label: str) -> CalibrationSample:
    _strict_fields(record, {"sample_id", "content_sha256", "metadata"}, label)
    try:
        sample = CalibrationSample(
            _text(record["sample_id"], f"{label} ID"),
            _text(record["content_sha256"], f"{label} content digest"),
            _metadata(record["metadata"]),
        )
    except ValueError as error:
        raise CalibrationCommandError(f"{label} has an invalid identity") from error
    if sample.to_record() != record:
        raise CalibrationCommandError(f"{label} is not canonical")
    return sample


def load_calibration_plan(path: Path) -> CalibrationPlan:
    """Load a strict calibration plan with all mutable identities declared."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationCommandError(f"cannot read calibration config: {error}") from error
    record = _object(raw, "calibration config")
    _strict_fields(
        record,
        {
            "schema_version",
            "dataset",
            "preprocessing",
            "tokenizer",
            "selection",
            "text_field",
            "batch_size",
            "max_tokens",
        },
        "calibration config",
    )
    if record["schema_version"] != CALIBRATION_COMMAND_SCHEMA_VERSION:
        raise CalibrationCommandError("unsupported calibration-command schema version")

    dataset = _object(record["dataset"], "dataset")
    _strict_fields(
        dataset,
        {"dataset", "revision", "split", "license", "trust", "trust_reason", "metadata"},
        "dataset",
    )
    license_value = dataset["license"]
    if license_value is not None and not isinstance(license_value, str):
        raise CalibrationCommandError("dataset license must be text or null")
    try:
        trust = DatasetTrust(_text(dataset["trust"], "dataset trust"))
        dataset_identity = DatasetIdentity(
            _text(dataset["dataset"], "dataset identifier"),
            _text(dataset["revision"], "dataset revision"),
            _text(dataset["split"], "dataset split"),
            license_value,
            trust,
            _text(dataset["trust_reason"], "dataset trust reason"),
            _metadata(dataset["metadata"]),
        )
    except ValueError as error:
        raise CalibrationCommandError(f"invalid dataset identity: {error}") from error

    preprocessing = _object(record["preprocessing"], "preprocessing")
    _strict_fields(preprocessing, {"name", "version", "configuration"}, "preprocessing")
    preprocessing_identity = PreprocessingIdentity(
        _text(preprocessing["name"], "preprocessing name"),
        _text(preprocessing["version"], "preprocessing version"),
        _configuration_digest(preprocessing["configuration"], "preprocessing configuration"),
    )

    tokenizer = _object(record["tokenizer"], "tokenizer")
    _strict_fields(tokenizer, {"tokenizer", "revision", "configuration"}, "tokenizer")
    tokenizer_identity = TokenizerIdentity(
        _text(tokenizer["tokenizer"], "tokenizer identifier"),
        _text(tokenizer["revision"], "tokenizer revision"),
        _configuration_digest(tokenizer["configuration"], "tokenizer configuration"),
    )

    selection = _object(record["selection"], "selection")
    _strict_fields(selection, {"seed", "sample_count", "algorithm"}, "selection")
    algorithm = _text(selection["algorithm"], "selection algorithm")
    if algorithm != SELECTION_ALGORITHM:
        raise CalibrationCommandError(f"unsupported selection algorithm {algorithm!r}")
    try:
        selection_config = SelectionConfig(
            _non_negative_int(selection["seed"], "selection seed"),
            _positive_int(selection["sample_count"], "selection sample count"),
            algorithm,
        )
        contract = CalibrationContract(
            dataset_identity,
            preprocessing_identity,
            tokenizer_identity,
            selection_config,
        )
        request = HuggingFaceCalibrationRequest(
            contract,
            text_field=_text(record["text_field"], "text field"),
            batch_size=_positive_int(record["batch_size"], "batch size"),
            max_tokens=_positive_int(record["max_tokens"], "max tokens"),
        )
    except ValueError as error:
        raise CalibrationCommandError(f"invalid calibration contract: {error}") from error
    return CalibrationPlan(request)


def _cache_identity(manifest_record: Mapping[str, object]) -> str:
    payload = canonical_identity_json(manifest_record).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _summary(
    *, state: str, path: Path, plan: CalibrationPlan, cache_identity: str | None,
    sample_count: int, token_count: int | None,
) -> dict[str, object]:
    request = plan.request
    return {
        "record_type": "calibration_run",
        "schema_version": CALIBRATION_COMMAND_SCHEMA_VERSION,
        "state": state,
        "cache": str(path),
        "cache_identity": cache_identity,
        "dataset": request.contract.dataset.dataset,
        "dataset_revision": request.contract.dataset.revision,
        "tokenizer": request.contract.tokenizer.tokenizer,
        "tokenizer_revision": request.contract.tokenizer.revision,
        "sample_count": sample_count,
        "requested_sample_count": request.contract.selection.sample_count,
        "token_count": token_count,
    }


def _validate_cache(record: object, plan: CalibrationPlan, path: Path) -> dict[str, object]:
    cached = _object(record, "calibration cache")
    _strict_fields(
        cached,
        {
            "record_type",
            "command_schema_version",
            "schema_version",
            "dataset",
            "preprocessing",
            "tokenizer",
            "selection",
            "samples",
            "tokenized_samples",
            "cache_identity",
            "sample_count",
            "token_count",
        },
        "calibration cache",
    )
    if cached["record_type"] != "calibration_manifest":
        raise CalibrationCommandError(f"cache has an invalid record type: {path}")
    if cached["command_schema_version"] != CALIBRATION_COMMAND_SCHEMA_VERSION:
        raise CalibrationCommandError(f"cache has an unsupported command schema: {path}")
    expected_contract = plan.request.contract.to_record(())
    for key in ("schema_version", "dataset", "preprocessing", "tokenizer", "selection"):
        if cached[key] != expected_contract[key]:
            raise CalibrationCommandError(f"cache contract does not match plan: {path}")
    samples = cached["samples"]
    tokenized = cached["tokenized_samples"]
    if not isinstance(samples, list) or not isinstance(tokenized, list):
        raise CalibrationCommandError(f"cache samples must be arrays: {path}")
    if len(samples) != len(tokenized) or not samples:
        raise CalibrationCommandError(f"cache must contain matching non-empty samples: {path}")
    if len(samples) != plan.request.contract.selection.sample_count:
        raise CalibrationCommandError(f"cache sample count does not match plan: {path}")
    sample_ids: set[str] = set()
    for identity, encoded in zip(samples, tokenized, strict=True):
        identity_record = _object(identity, "calibration cache sample")
        encoded_record = _object(encoded, "tokenized calibration cache sample")
        identity_sample = _sample_identity(identity_record, "cache sample")
        if identity_sample.sample_id in sample_ids:
            raise CalibrationCommandError(f"cache sample IDs are not unique: {path}")
        sample_ids.add(identity_sample.sample_id)
        _strict_fields(
            encoded_record,
            {"sample_id", "content_sha256", "metadata", "input_ids"},
            "tokenized cache sample",
        )
        _sample_identity(
            {key: encoded_record[key] for key in ("sample_id", "content_sha256", "metadata")},
            "tokenized cache sample",
        )
        if {key: encoded_record[key] for key in identity_record} != identity_record:
            raise CalibrationCommandError(f"cache sample identities do not match: {path}")
        _input_ids(encoded_record["input_ids"], f"cache token IDs at {path}")
    sample_count = cached["sample_count"]
    if not isinstance(sample_count, int) or isinstance(sample_count, bool):
        raise CalibrationCommandError(f"cache sample count is invalid: {path}")
    if sample_count != len(tokenized):
        raise CalibrationCommandError(f"cache sample count is invalid: {path}")
    token_count = sum(
        len(_input_ids(_object(item, "tokenized sample")["input_ids"], "token IDs"))
        for item in tokenized
    )
    stored_token_count = cached["token_count"]
    if not isinstance(stored_token_count, int) or isinstance(stored_token_count, bool):
        raise CalibrationCommandError(f"cache token count is invalid: {path}")
    if stored_token_count != token_count:
        raise CalibrationCommandError(f"cache token count is invalid: {path}")
    cache_identity = cached["cache_identity"]
    if not isinstance(cache_identity, str):
        raise CalibrationCommandError(f"cache identity is invalid: {path}")
    manifest_record = {key: cached[key] for key in expected_contract}
    manifest_record["samples"] = samples
    manifest_record["tokenized_samples"] = tokenized
    if cache_identity != _cache_identity(manifest_record):
        raise CalibrationCommandError(f"cache identity is invalid: {path}")
    return _summary(
        state="reused",
        path=path,
        plan=plan,
        cache_identity=cache_identity,
        sample_count=len(tokenized),
        token_count=token_count,
    )


def _publish_cache(path: Path, manifest: CalibrationManifest) -> dict[str, object]:
    base = manifest.to_record()
    tokenized = base["tokenized_samples"]
    if not isinstance(tokenized, list) or not tokenized:
        raise CalibrationCommandError("calibration produced no tokenized samples")
    token_count = sum(
        len(_input_ids(_object(item, "tokenized sample")["input_ids"], "token IDs"))
        for item in tokenized
    )
    record = {
        **base,
        "record_type": "calibration_manifest",
        "command_schema_version": CALIBRATION_COMMAND_SCHEMA_VERSION,
        "cache_identity": _cache_identity(base),
        "sample_count": len(tokenized),
        "token_count": token_count,
    }
    payload = canonical_identity_json(record) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return record


def run_calibration(
    plan: CalibrationPlan,
    *,
    cache_path: Path,
    dry_run: bool = False,
    refresh: bool = False,
) -> dict[str, object]:
    """Execute one calibration plan without publishing partial cache state."""

    if dry_run:
        return _summary(
            state="dry_run",
            path=cache_path,
            plan=plan,
            cache_identity=None,
            sample_count=0,
            token_count=None,
        )
    if cache_path.exists() and not refresh:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CalibrationCommandError(f"cannot read calibration cache: {error}") from error
        return _validate_cache(cached, plan, cache_path)

    cache_existed = cache_path.exists()
    manifest = stream_huggingface_calibration(plan.request)
    if len(manifest.samples) != plan.request.contract.selection.sample_count:
        raise CalibrationCommandError("calibration adapter returned the wrong sample count")
    expected = plan.request.contract.to_record(
        tuple(sample.identity for sample in manifest.samples)
    )
    if manifest.contract_record != expected:
        raise CalibrationCommandError("calibration adapter returned a contract mismatch")
    record = _publish_cache(cache_path, manifest)
    cache_identity = record["cache_identity"]
    sample_count = record["sample_count"]
    token_count = record["token_count"]
    if (
        not isinstance(cache_identity, str)
        or not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or not isinstance(token_count, int)
        or isinstance(token_count, bool)
    ):
        raise CalibrationCommandError("calibration cache publication returned invalid metadata")
    return _summary(
        state="refreshed" if cache_existed else "created",
        path=cache_path,
        plan=plan,
        cache_identity=cache_identity,
        sample_count=sample_count,
        token_count=token_count,
    )


def calibrate_command(
    config: Annotated[Path, typer.Argument(help="Strict JSON calibration plan")],
    cache: Annotated[
        Path, typer.Option("--cache", help="Atomic calibration manifest cache")
    ],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Validate and report without dataset access")
    ] = False,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Rebuild an existing cache atomically")
    ] = False,
) -> None:
    """Stream a pinned dataset, tokenize bounded samples, and cache the manifest."""

    try:
        result = run_calibration(
            load_calibration_plan(config),
            cache_path=cache,
            dry_run=dry_run,
            refresh=refresh,
        )
        typer.echo(canonical_identity_json(result))
    except KeyboardInterrupt:
        typer.echo("calibrate interrupted; existing cache preserved", err=True)
        raise typer.Exit(130) from None
    except (CalibrationCommandError, OSError, ValueError, RuntimeError) as error:
        typer.echo(f"calibrate error: {error}", err=True)
        raise typer.Exit(2) from error
