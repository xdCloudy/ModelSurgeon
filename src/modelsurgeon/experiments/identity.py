"""Canonical deterministic experiment, run, and candidate identities."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

from modelsurgeon.experiments.schema import DatasetTarget, ModelTarget, SeedContext

IDENTITY_SCHEMA_VERSION = 1


class ExperimentIdentityError(ValueError):
    """Raised when immutable identity inputs cannot be canonicalized safely."""


@dataclass(frozen=True, slots=True)
class PathAlias:
    """Map one machine-local root to a stable semantic alias before hashing."""

    name: str
    local_root: str

    def __post_init__(self) -> None:
        if not self.name or not self.local_root:
            raise ExperimentIdentityError("path aliases require a name and local root")
        if any(character in self.name for character in "/\\{}"):
            raise ExperimentIdentityError("path alias names cannot contain path delimiters or braces")


@dataclass(frozen=True, slots=True)
class ExperimentIdentitySpec:
    model: ModelTarget
    dataset: DatasetTarget
    resolved_config: Mapping[str, object]
    seeds: SeedContext
    tool_revision: str
    evaluator_version: str
    feature_schema_version: int
    mutation_record_schema_version: int
    path_aliases: tuple[PathAlias, ...] = ()

    def __post_init__(self) -> None:
        if not self.tool_revision or not self.evaluator_version:
            raise ExperimentIdentityError("tool and evaluator revisions are required")
        if self.feature_schema_version <= 0 or self.mutation_record_schema_version <= 0:
            raise ExperimentIdentityError("schema versions must be positive")
        names = tuple(alias.name for alias in self.path_aliases)
        if len(names) != len(set(names)):
            raise ExperimentIdentityError("path alias names must be unique")


@dataclass(frozen=True, slots=True)
class DerivedExperimentIdentity:
    experiment_id: str
    config_digest: str
    canonical_payload: str


@dataclass(frozen=True, slots=True)
class DerivedRunIdentity:
    run_id: str
    experiment_id: str
    logical_run_key: str


@dataclass(frozen=True, slots=True)
class DerivedCandidateIdentity:
    candidate_id: str
    run_id: str
    mutation_id: str


def _normalize_slashes(value: str) -> str:
    normalized = value.replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    if len(normalized) > 1:
        normalized = normalized.rstrip("/")
    return normalized


def _path_match(value: str, alias: PathAlias) -> str | None:
    normalized_value = _normalize_slashes(value)
    normalized_root = _normalize_slashes(alias.local_root)
    windows_style = len(normalized_root) >= 2 and normalized_root[1] == ":"
    compared_value = normalized_value.casefold() if windows_style else normalized_value
    compared_root = normalized_root.casefold() if windows_style else normalized_root
    if compared_value == compared_root:
        return f"@{alias.name}"
    prefix = f"{compared_root}/"
    if compared_value.startswith(prefix):
        suffix = normalized_value[len(normalized_root) :]
        return f"@{alias.name}{suffix}"
    return None


def _canonical_string(value: str, aliases: tuple[PathAlias, ...]) -> str:
    ranked = sorted(aliases, key=lambda item: (-len(_normalize_slashes(item.local_root)), item.name))
    for alias in ranked:
        replacement = _path_match(value, alias)
        if replacement is not None:
            return replacement
    return value


def canonicalize_identity_value(value: object, aliases: tuple[PathAlias, ...] = ()) -> object:
    """Convert JSON-compatible immutable inputs to a canonical alias-normalized tree."""

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExperimentIdentityError("identity inputs cannot contain non-finite floats")
        return value
    if isinstance(value, str):
        return _canonical_string(value, aliases)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) and key for key in value):
            raise ExperimentIdentityError("identity mappings require non-empty string keys")
        return {
            key: canonicalize_identity_value(value[key], aliases)
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [canonicalize_identity_value(item, aliases) for item in value]
    raise ExperimentIdentityError(
        f"identity input type {type(value).__name__} is not JSON-compatible"
    )


def canonical_identity_json(value: object, aliases: tuple[PathAlias, ...] = ()) -> str:
    canonical = canonicalize_identity_value(value, aliases)
    return json.dumps(
        canonical,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(namespace: str, payload: object) -> str:
    encoded = canonical_identity_json(payload).encode("utf-8")
    return f"{namespace}_{hashlib.sha256(encoded).hexdigest()}"


def derive_experiment_identity(spec: ExperimentIdentitySpec) -> DerivedExperimentIdentity:
    canonical_config = canonicalize_identity_value(spec.resolved_config, spec.path_aliases)
    config_json = canonical_identity_json(canonical_config)
    config_digest = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    payload = {
        "identity_schema_version": IDENTITY_SCHEMA_VERSION,
        "model": canonicalize_identity_value(spec.model.to_record(), spec.path_aliases),
        "dataset": canonicalize_identity_value(spec.dataset.to_record(), spec.path_aliases),
        "config": canonical_config,
        "config_digest": config_digest,
        "seeds": spec.seeds.to_record(),
        "tool_revision": spec.tool_revision,
        "evaluator_version": spec.evaluator_version,
        "feature_schema_version": spec.feature_schema_version,
        "mutation_record_schema_version": spec.mutation_record_schema_version,
    }
    canonical_payload = canonical_identity_json(payload)
    experiment_id = f"exp_{hashlib.sha256(canonical_payload.encode('utf-8')).hexdigest()}"
    return DerivedExperimentIdentity(experiment_id, config_digest, canonical_payload)


def derive_run_identity(experiment_id: str, logical_run_key: str = "default") -> DerivedRunIdentity:
    if not experiment_id.startswith("exp_") or not logical_run_key:
        raise ExperimentIdentityError("run identity requires an experiment ID and logical run key")
    run_id = _digest(
        "run",
        {
            "identity_schema_version": IDENTITY_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "logical_run_key": logical_run_key,
        },
    )
    return DerivedRunIdentity(run_id, experiment_id, logical_run_key)


def derive_candidate_identity(run_id: str, mutation_id: str) -> DerivedCandidateIdentity:
    if not run_id.startswith("run_") or not mutation_id:
        raise ExperimentIdentityError("candidate identity requires a run ID and mutation ID")
    candidate_id = _digest(
        "cand",
        {
            "identity_schema_version": IDENTITY_SCHEMA_VERSION,
            "run_id": run_id,
            "mutation_id": mutation_id,
        },
    )
    return DerivedCandidateIdentity(candidate_id, run_id, mutation_id)
