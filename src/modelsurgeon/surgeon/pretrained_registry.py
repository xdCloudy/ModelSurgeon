"""Signed, offline-resolvable registry records for pretrained meta-surgeons."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, cast

from modelsurgeon.experiments.artifacts import (
    ArtifactDigest,
    ContentAddressedArtifactStore,
    StoredArtifact,
)

from .registry import SurgeonBundle, SurgeonModelRegistry, TrainingModelIdentity

PRETRAINED_REGISTRY_SCHEMA_VERSION: Final[int] = 1
PRETRAINED_REGISTRY_PROTOCOL_REVISION: Final[str] = "pretrained-registry-v1"


class PretrainedRegistryError(ValueError):
    """Raised when a pretrained card or signed registry identity is unsafe."""


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PretrainedRegistryError(f"{label} is required")
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode()).hexdigest()


def _artifact_digest(value: str, label: str) -> str:
    try:
        return str(ArtifactDigest.parse(value))
    except (TypeError, ValueError) as error:
        raise PretrainedRegistryError(f"{label} must be a SHA-256 artifact digest") from error


@dataclass(frozen=True, slots=True)
class SignedContentIdentity:
    content_digest: str
    key_id: str
    signature: str
    algorithm: str = "hmac-sha256"

    def __post_init__(self) -> None:
        _artifact_digest(self.content_digest, "content digest")
        _text(self.key_id, "signature key ID")
        if self.algorithm != "hmac-sha256":
            raise PretrainedRegistryError("unsupported registry signature algorithm")
        if len(self.signature) != 64 or any(
            character not in "0123456789abcdef" for character in self.signature
        ):
            raise PretrainedRegistryError("registry signature must be lowercase hexadecimal")

    def to_record(self) -> dict[str, str]:
        return {
            "content_digest": self.content_digest,
            "key_id": self.key_id,
            "signature": self.signature,
            "algorithm": self.algorithm,
        }


@dataclass(frozen=True, slots=True)
class PretrainedSurgeonCard:
    """Complete metadata required before a persisted bundle may be deserialized."""

    bundle_digest: str
    model_id: str
    model_revision: str
    training_models: tuple[TrainingModelIdentity, ...]
    source_evidence_digests: tuple[str, ...]
    feature_schema_version: int
    target_schema_version: int
    state_schema_version: int
    supported_mutations: tuple[str, ...]
    supported_codecs: tuple[str, ...]
    supported_hardware: tuple[str, ...]
    calibration_policy: Mapping[str, object]
    compatibility: Mapping[str, object]
    license: str
    metrics: tuple[tuple[str, float | None], ...]
    limitations: tuple[str, ...]
    parent_bundle_digest: str | None = None
    adaptation_request_id: str | None = None
    signed_identity: SignedContentIdentity | None = None
    schema_version: int = PRETRAINED_REGISTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PRETRAINED_REGISTRY_SCHEMA_VERSION:
            raise PretrainedRegistryError("unsupported pretrained card schema")
        for label, value in (
            ("bundle digest", self.bundle_digest),
            ("model ID", self.model_id),
            ("model revision", self.model_revision),
            ("license", self.license),
        ):
            _text(value, label)
        _artifact_digest(self.bundle_digest, "bundle digest")
        if not self.training_models or not self.source_evidence_digests:
            raise PretrainedRegistryError("pretrained cards require source models and evidence")
        for digest in self.source_evidence_digests:
            _artifact_digest(digest, "source evidence digest")
        for label, version in (
            ("feature schema version", self.feature_schema_version),
            ("target schema version", self.target_schema_version),
            ("state schema version", self.state_schema_version),
        ):
            if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
                raise PretrainedRegistryError(f"{label} must be positive")
        if not self.calibration_policy or not self.compatibility:
            raise PretrainedRegistryError(
                "pretrained cards require calibration and compatibility metadata"
            )
        for label, values in (
            ("supported mutation", self.supported_mutations),
            ("supported codec", self.supported_codecs),
            ("supported hardware", self.supported_hardware),
            ("limitation", self.limitations),
        ):
            if values != tuple(sorted(set(values))) or any(not value.strip() for value in values):
                raise PretrainedRegistryError(f"{label} declarations must be unique and canonical")
        metric_names = tuple(name for name, _ in self.metrics)
        if metric_names != tuple(sorted(set(metric_names))) or any(
            not name for name in metric_names
        ):
            raise PretrainedRegistryError("pretrained card metrics must be unique and canonical")
        for name, metric_value in self.metrics:
            if metric_value is not None and (
                isinstance(metric_value, bool) or not isinstance(metric_value, (int, float))
            ):
                raise PretrainedRegistryError(f"metric {name!r} must be numeric or null")
        if (self.parent_bundle_digest is None) != (self.adaptation_request_id is None):
            raise PretrainedRegistryError("adapted cards require parent bundle and request lineage")
        if self.parent_bundle_digest is not None:
            _artifact_digest(self.parent_bundle_digest, "parent bundle digest")
            if self.parent_bundle_digest == self.bundle_digest:
                raise PretrainedRegistryError("adapted child cannot overwrite its parent bundle")

    def content_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_revision": PRETRAINED_REGISTRY_PROTOCOL_REVISION,
            "bundle_digest": self.bundle_digest,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "training_models": [item.to_record() for item in self.training_models],
            "source_evidence_digests": list(self.source_evidence_digests),
            "feature_schema_version": self.feature_schema_version,
            "target_schema_version": self.target_schema_version,
            "state_schema_version": self.state_schema_version,
            "supported_mutations": list(self.supported_mutations),
            "supported_codecs": list(self.supported_codecs),
            "supported_hardware": list(self.supported_hardware),
            "calibration_policy": dict(self.calibration_policy),
            "compatibility": dict(self.compatibility),
            "license": self.license,
            "metrics": dict(self.metrics),
            "limitations": list(self.limitations),
            "parent_bundle_digest": self.parent_bundle_digest,
            "adaptation_request_id": self.adaptation_request_id,
        }

    def sign(self, key_id: str, secret: bytes) -> PretrainedSurgeonCard:
        _text(key_id, "signature key ID")
        if not secret:
            raise PretrainedRegistryError("registry signing secret cannot be empty")
        payload = _canonical(self.content_record()).encode()
        content_digest = _digest(self.content_record())
        signature = hmac.new(secret, payload, hashlib.sha256).hexdigest()
        return replace(
            self,
            signed_identity=SignedContentIdentity(content_digest, key_id, signature),
        )

    def verify(self, secret: bytes) -> None:
        identity = self.signed_identity
        if identity is None:
            raise PretrainedRegistryError("pretrained card is unsigned")
        payload = _canonical(self.content_record()).encode()
        expected_digest = _digest(self.content_record())
        expected_signature = hmac.new(secret, payload, hashlib.sha256).hexdigest()
        if identity.content_digest != expected_digest or not hmac.compare_digest(
            identity.signature, expected_signature
        ):
            raise PretrainedRegistryError(
                "pretrained card signature or content identity is invalid"
            )

    def to_record(self) -> dict[str, object]:
        record = self.content_record()
        record["signed_identity"] = (
            None if self.signed_identity is None else self.signed_identity.to_record()
        )
        return record

    @classmethod
    def from_record(cls, raw: Mapping[str, object]) -> PretrainedSurgeonCard:
        if raw.get("schema_version") != PRETRAINED_REGISTRY_SCHEMA_VERSION:
            raise PretrainedRegistryError("pretrained card schema version is incompatible")
        training_raw = raw.get("training_models")
        metrics_raw = raw.get("metrics")
        signed_raw = raw.get("signed_identity")
        if not isinstance(training_raw, list) or not isinstance(metrics_raw, Mapping):
            raise PretrainedRegistryError("pretrained card model or metric metadata is malformed")
        training: list[TrainingModelIdentity] = []
        for item in training_raw:
            if not isinstance(item, Mapping):
                raise PretrainedRegistryError("pretrained training identity is malformed")
            identifier, revision, quantization = (
                item.get("identifier"),
                item.get("revision"),
                item.get("quantization"),
            )
            if not isinstance(identifier, str) or not isinstance(revision, str):
                raise PretrainedRegistryError("pretrained training identity is incomplete")
            if quantization is not None and not isinstance(quantization, str):
                raise PretrainedRegistryError("pretrained quantization identity is malformed")
            training.append(TrainingModelIdentity(identifier, revision, quantization))
        metrics: list[tuple[str, float | None]] = []
        for name, value in metrics_raw.items():
            if not isinstance(name, str) or (
                value is not None
                and (isinstance(value, bool) or not isinstance(value, (int, float)))
            ):
                raise PretrainedRegistryError("pretrained card metric is malformed")
            metrics.append((name, None if value is None else float(value)))
        signed: SignedContentIdentity | None = None
        if signed_raw is not None:
            if not isinstance(signed_raw, Mapping):
                raise PretrainedRegistryError("pretrained card signature is malformed")
            signed = SignedContentIdentity(
                cast(str, signed_raw.get("content_digest")),
                cast(str, signed_raw.get("key_id")),
                cast(str, signed_raw.get("signature")),
                cast(str, signed_raw.get("algorithm", "hmac-sha256")),
            )

        def _strings(name: str) -> tuple[str, ...]:
            value = raw.get(name)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise PretrainedRegistryError(f"pretrained card {name} is malformed")
            return tuple(cast(list[str], value))

        def _mapping(name: str) -> Mapping[str, object]:
            value = raw.get(name)
            if not isinstance(value, Mapping):
                raise PretrainedRegistryError(f"pretrained card {name} is malformed")
            return dict(value)

        return cls(
            cast(str, raw.get("bundle_digest")),
            cast(str, raw.get("model_id")),
            cast(str, raw.get("model_revision")),
            tuple(training),
            _strings("source_evidence_digests"),
            cast(int, raw.get("feature_schema_version")),
            cast(int, raw.get("target_schema_version")),
            cast(int, raw.get("state_schema_version")),
            _strings("supported_mutations"),
            _strings("supported_codecs"),
            _strings("supported_hardware"),
            _mapping("calibration_policy"),
            _mapping("compatibility"),
            cast(str, raw.get("license")),
            tuple(metrics),
            _strings("limitations"),
            cast(str | None, raw.get("parent_bundle_digest")),
            cast(str | None, raw.get("adaptation_request_id")),
            signed,
        )


@dataclass(frozen=True, slots=True)
class PublishedPretrainedCard:
    card: PretrainedSurgeonCard
    artifact: StoredArtifact


class PretrainedSurgeonRegistry:
    """Resolve signed local cards before loading any model bytes."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().absolute().resolve(strict=False)
        self.bundles = SurgeonModelRegistry(str(self.root))
        self.cards = ContentAddressedArtifactStore(self.root / "cards")

    def publish(
        self,
        card: PretrainedSurgeonCard,
        *,
        key_id: str,
        secret: bytes,
    ) -> PublishedPretrainedCard:
        signed = card.sign(key_id, secret)
        signed.verify(secret)
        try:
            self.bundles.artifacts.get(signed.bundle_digest)
        except Exception as error:
            raise PretrainedRegistryError(
                "pretrained card references an unavailable bundle"
            ) from error
        payload = _canonical(signed.to_record()).encode()
        artifact = self.cards.put_bytes(payload)
        return PublishedPretrainedCard(signed, artifact)

    def verify(self, card_digest: ArtifactDigest | str, *, secret: bytes) -> PretrainedSurgeonCard:
        artifact = self.cards.get(card_digest)
        try:
            raw = json.loads(artifact.data_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PretrainedRegistryError("pretrained card is unreadable or corrupt") from error
        if not isinstance(raw, Mapping):
            raise PretrainedRegistryError("pretrained card must be an object")
        card = PretrainedSurgeonCard.from_record(raw)
        card.verify(secret)
        try:
            self.bundles.artifacts.get(card.bundle_digest)
        except Exception as error:
            raise PretrainedRegistryError(
                "pretrained bundle is unavailable or corrupted"
            ) from error
        return card

    def resolve(
        self,
        card_digest: ArtifactDigest | str,
        *,
        secret: bytes,
        expected_feature_schema_version: int | None = None,
        expected_target_schema_version: int | None = None,
    ) -> tuple[PretrainedSurgeonCard, SurgeonBundle]:
        """Verify the card and compatibility metadata before model deserialization."""

        card = self.verify(card_digest, secret=secret)
        if (
            expected_feature_schema_version is not None
            and card.feature_schema_version != expected_feature_schema_version
        ):
            raise PretrainedRegistryError("pretrained card feature schema is incompatible")
        if (
            expected_target_schema_version is not None
            and card.target_schema_version != expected_target_schema_version
        ):
            raise PretrainedRegistryError("pretrained card target schema is incompatible")
        bundle = self.bundles.load(card.bundle_digest)
        if bundle.artifact.metadata.digest != ArtifactDigest.parse(card.bundle_digest):
            raise PretrainedRegistryError("resolved bundle identity does not match card")
        return card, bundle

    def list_cards(self) -> tuple[str, ...]:
        if not self.cards.algorithm_root.exists():
            return ()
        digests: list[str] = []
        for path in self.cards.algorithm_root.glob("*/*/data"):
            if path.is_file():
                digests.append(f"sha256:{path.parent.parent.name}{path.parent.name}")
        return tuple(sorted(digests))
