"""Immutable surgeon model bundles, schema checks, and evaluation cards."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

from modelsurgeon.experiments.artifacts import (
    ArtifactDigest,
    ContentAddressedArtifactStore,
    StoredArtifact,
)
from modelsurgeon.experiments.identity import canonical_identity_json

from .matrix import SurgeonPreprocessor
from .models import SerializableModel, model_from_json, model_to_json
from .targets import TargetSchema

SURGEON_BUNDLE_SCHEMA_VERSION: Final[int] = 1


class SurgeonRegistryError(ValueError):
    """Raised when an immutable surgeon bundle is incomplete or incompatible."""


@dataclass(frozen=True, slots=True)
class TrainingModelIdentity:
    identifier: str
    revision: str
    quantization: str | None = None

    def __post_init__(self) -> None:
        if not self.identifier or not self.revision:
            raise SurgeonRegistryError("training model identity and revision are required")
        if self.quantization is not None and not self.quantization:
            raise SurgeonRegistryError("training model quantization cannot be blank")

    def to_record(self) -> dict[str, str | None]:
        return {
            "identifier": self.identifier,
            "revision": self.revision,
            "quantization": self.quantization,
        }


@dataclass(frozen=True, slots=True)
class SurgeonEvaluationCard:
    model_kind: str
    target_name: str
    training_models: tuple[TrainingModelIdentity, ...]
    metrics: tuple[tuple[str, float | None], ...]
    split_manifest: Mapping[str, object]
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.model_kind or not self.target_name or not self.training_models:
            raise SurgeonRegistryError(
                "evaluation cards require model kind, target, and training model identities"
            )
        metric_names = tuple(name for name, _ in self.metrics)
        if metric_names != tuple(sorted(set(metric_names))):
            raise SurgeonRegistryError("evaluation card metric names must be unique and canonical")

    def to_record(self) -> dict[str, object]:
        return {
            "model_kind": self.model_kind,
            "target_name": self.target_name,
            "training_models": [item.to_record() for item in self.training_models],
            "metrics": dict(self.metrics),
            "split_manifest": dict(self.split_manifest),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class SurgeonBundle:
    model: SerializableModel
    preprocessor: SurgeonPreprocessor
    target_schema: Mapping[str, object]
    card: SurgeonEvaluationCard
    artifact: StoredArtifact


def _model_kind(model: SerializableModel) -> str:
    record = model.to_record()
    kind = record.get("kind")
    if not isinstance(kind, str) or not kind:
        raise SurgeonRegistryError("surgeon model record has no kind")
    return kind


def _metric_pairs(metrics: Mapping[str, float | None]) -> tuple[tuple[str, float | None], ...]:
    values: list[tuple[str, float | None]] = []
    for name, value in metrics.items():
        if not name:
            raise SurgeonRegistryError("evaluation metric names cannot be blank")
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise SurgeonRegistryError("evaluation metrics must be numeric or null")
        values.append((name, None if value is None else float(value)))
    return tuple(sorted(values))


class SurgeonModelRegistry:
    """Persist one self-contained surgeon version as a content-addressed immutable artifact."""

    def __init__(self, root: str) -> None:
        self.artifacts = ContentAddressedArtifactStore(root)

    def publish(
        self,
        model: SerializableModel,
        preprocessor: SurgeonPreprocessor,
        target_schema: TargetSchema,
        *,
        training_models: Sequence[TrainingModelIdentity],
        metrics: Mapping[str, float | None],
        split_manifest: Mapping[str, object],
        provenance: Mapping[str, object],
    ) -> StoredArtifact:
        if tuple(model.feature_names) != preprocessor.output_feature_names:
            raise SurgeonRegistryError(
                "model feature names do not match the preprocessing output schema"
            )
        valid_targets = {
            "safe_mutation",
            *(item.name for item in target_schema.metrics),
        }
        if model.target_name not in valid_targets:
            raise SurgeonRegistryError(
                f"model target {model.target_name!r} is absent from target schema"
            )
        card = SurgeonEvaluationCard(
            _model_kind(model),
            model.target_name,
            tuple(training_models),
            _metric_pairs(metrics),
            dict(split_manifest),
            dict(provenance),
        )
        payload = {
            "schema_version": SURGEON_BUNDLE_SCHEMA_VERSION,
            "model": json.loads(model_to_json(model)),
            "preprocessor": preprocessor.to_record(),
            "target_schema": target_schema.to_record(),
            "card": card.to_record(),
        }
        return self.artifacts.put_bytes(
            canonical_identity_json(payload).encode("utf-8")
        )

    def load(
        self,
        digest: ArtifactDigest | str,
        *,
        expected_feature_schema_version: int | None = None,
        expected_feature_names: Sequence[str] | None = None,
        expected_target_schema: TargetSchema | None = None,
    ) -> SurgeonBundle:
        artifact = self.artifacts.get(digest)
        try:
            raw = json.loads(artifact.data_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SurgeonRegistryError("surgeon bundle is unreadable or corrupt") from error
        if not isinstance(raw, dict) or raw.get("schema_version") != SURGEON_BUNDLE_SCHEMA_VERSION:
            raise SurgeonRegistryError("surgeon bundle schema version is incompatible")
        model_raw = raw.get("model")
        preprocessor_raw = raw.get("preprocessor")
        target_raw = raw.get("target_schema")
        card_raw = raw.get("card")
        if (
            not isinstance(model_raw, dict)
            or not isinstance(preprocessor_raw, Mapping)
            or not isinstance(target_raw, Mapping)
            or not isinstance(card_raw, Mapping)
        ):
            raise SurgeonRegistryError("surgeon bundle sections are missing or malformed")

        preprocessor = SurgeonPreprocessor.from_record(preprocessor_raw)
        model = model_from_json(canonical_identity_json(model_raw))
        if tuple(model.feature_names) != preprocessor.output_feature_names:
            raise SurgeonRegistryError(
                "persisted model/preprocessing feature schemas are internally incompatible"
            )
        if (
            expected_feature_schema_version is not None
            and preprocessor.source_feature_schema_version != expected_feature_schema_version
        ):
            raise SurgeonRegistryError(
                "inference feature schema version is incompatible with trained preprocessing"
            )
        if (
            expected_feature_names is not None
            and tuple(expected_feature_names) != preprocessor.output_feature_names
        ):
            raise SurgeonRegistryError(
                "inference feature names are incompatible with trained preprocessing"
            )
        if (
            expected_target_schema is not None
            and dict(target_raw) != expected_target_schema.to_record()
        ):
            raise SurgeonRegistryError(
                "inference target schema is incompatible with trained surgeon"
            )

        training_models_raw = card_raw.get("training_models")
        metrics_raw = card_raw.get("metrics")
        split_raw = card_raw.get("split_manifest")
        provenance_raw = card_raw.get("provenance")
        model_kind = card_raw.get("model_kind")
        target_name = card_raw.get("target_name")
        if (
            not isinstance(training_models_raw, list)
            or not isinstance(metrics_raw, Mapping)
            or not isinstance(split_raw, Mapping)
            or not isinstance(provenance_raw, Mapping)
            or not isinstance(model_kind, str)
            or not isinstance(target_name, str)
        ):
            raise SurgeonRegistryError("surgeon evaluation card is malformed")
        training_models: list[TrainingModelIdentity] = []
        for item in training_models_raw:
            if not isinstance(item, Mapping):
                raise SurgeonRegistryError("training model card entries must be objects")
            identifier = item.get("identifier")
            revision = item.get("revision")
            quantization = item.get("quantization")
            if (
                not isinstance(identifier, str)
                or not isinstance(revision, str)
                or (quantization is not None and not isinstance(quantization, str))
            ):
                raise SurgeonRegistryError("training model card identity is malformed")
            training_models.append(
                TrainingModelIdentity(identifier, revision, quantization)
            )
        parsed_metrics: dict[str, float | None] = {}
        for name, value in metrics_raw.items():
            if not isinstance(name, str):
                raise SurgeonRegistryError("evaluation metric names must be strings")
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise SurgeonRegistryError("evaluation metric values must be numeric or null")
            parsed_metrics[name] = None if value is None else float(value)

        card = SurgeonEvaluationCard(
            model_kind,
            target_name,
            tuple(training_models),
            _metric_pairs(parsed_metrics),
            dict(split_raw),
            dict(provenance_raw),
        )
        return SurgeonBundle(
            model,
            preprocessor,
            cast(Mapping[str, object], target_raw),
            card,
            artifact,
        )
