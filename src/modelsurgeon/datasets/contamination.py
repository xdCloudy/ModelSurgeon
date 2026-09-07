"""Bounded, provenance-aware corpus contamination and ancestry audits."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

CONTAMINATION_AUDIT_SCHEMA_VERSION: Final[int] = 1
CONTAMINATION_PROTOCOL_REVISION: Final[str] = "contamination-audit-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ContaminationAuditError(ValueError):
    """Raised when an audit input or bounded resource declaration is invalid."""


class ContaminationStatus(StrEnum):
    CLEAN = "clean"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


class FindingKind(StrEnum):
    EXACT_OVERLAP = "exact_overlap"
    NEAR_OVERLAP = "near_overlap"
    ANCESTRY_OVERLAP = "ancestry_overlap"
    UNAVAILABLE_SOURCE = "unavailable_source"
    RESOURCE_LIMIT = "resource_limit"
    LICENSE_LIMITATION = "license_limitation"


class FindingAction(StrEnum):
    BLOCK = "block_claim"
    MOVE = "move_sample_and_new_protocol"
    LIMITATION = "retain_audit_limitation"


def _require_text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContaminationAuditError(f"{label} is required")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ContaminationPolicy:
    """Deterministic matching thresholds and pairwise resource ceilings."""

    ngram_size: int = 5
    near_jaccard_threshold: float = 0.8
    max_items: int = 100_000
    max_pair_comparisons: int = 1_000_000
    license_safe_only: bool = True

    def __post_init__(self) -> None:
        if self.ngram_size < 1 or not 0.0 < self.near_jaccard_threshold <= 1.0:
            raise ContaminationAuditError("invalid contamination matching threshold")
        if self.max_items <= 0 or self.max_pair_comparisons <= 0:
            raise ContaminationAuditError("contamination resource limits must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            "ngram_size": self.ngram_size,
            "near_jaccard_threshold": self.near_jaccard_threshold,
            "max_items": self.max_items,
            "max_pair_comparisons": self.max_pair_comparisons,
            "license_safe_only": self.license_safe_only,
        }


@dataclass(frozen=True, slots=True)
class CorpusSource:
    """Pinned source identity and auditability metadata."""

    corpus_id: str
    revision: str
    split: str
    license: str
    kind: str
    available: bool = True
    ancestry_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for label, value in (
            ("corpus_id", self.corpus_id),
            ("revision", self.revision),
            ("split", self.split),
            ("license", self.license),
            ("kind", self.kind),
        ):
            _require_text(value, label)
        duplicate_ancestry = len(set(self.ancestry_ids)) != len(self.ancestry_ids)
        if self.corpus_id in self.ancestry_ids or duplicate_ancestry:
            raise ContaminationAuditError("source ancestry IDs must be unique and exclude itself")

    def to_record(self) -> dict[str, object]:
        return {
            "corpus_id": self.corpus_id,
            "revision": self.revision,
            "split": self.split,
            "license": self.license,
            "kind": self.kind,
            "available": self.available,
            "ancestry_ids": list(self.ancestry_ids),
        }


@dataclass(frozen=True, slots=True)
class CorpusItem:
    """One text or generated output represented by safe derived features."""

    item_id: str
    corpus_id: str
    text: str
    generated: bool = False
    ancestry_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.item_id, "item_id")
        _require_text(self.corpus_id, "item corpus_id")
        if not self.text:
            raise ContaminationAuditError("corpus item text cannot be empty")
        if len(set(self.ancestry_ids)) != len(self.ancestry_ids):
            raise ContaminationAuditError("item ancestry IDs must be unique")

    @property
    def text_digest(self) -> str:
        return _sha256(self.text)

    def shingles(self, size: int) -> frozenset[str]:
        normalized = " ".join(self.text.split())
        tokens = normalized.split(" ")
        if len(tokens) <= size:
            return frozenset((normalized,))
        return frozenset(
            " ".join(tokens[index : index + size])
            for index in range(len(tokens) - size + 1)
        )

    def to_record(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "corpus_id": self.corpus_id,
            "text_sha256": self.text_digest,
            "text_bytes": len(self.text.encode("utf-8")),
            "generated": self.generated,
            "ancestry_ids": list(self.ancestry_ids),
        }


@dataclass(frozen=True, slots=True)
class ContaminationFinding:
    kind: FindingKind
    action: FindingAction
    left_item_id: str | None
    right_item_id: str | None
    corpus_ids: tuple[str, ...]
    score: float | None
    detail: str

    def __post_init__(self) -> None:
        if not self.corpus_ids or self.corpus_ids != tuple(sorted(set(self.corpus_ids))):
            raise ContaminationAuditError("finding corpus IDs must be non-empty and sorted")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ContaminationAuditError("finding score must be within [0, 1]")
        _require_text(self.detail, "finding detail")

    @property
    def finding_id(self) -> str:
        payload = {
            "kind": self.kind.value,
            "action": self.action.value,
            "left": self.left_item_id,
            "right": self.right_item_id,
            "corpus_ids": self.corpus_ids,
            "score": self.score,
            "detail": self.detail,
        }
        return "finding_" + hashlib.sha256(_canonical(payload).encode()).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            "finding_id": self.finding_id,
            "kind": self.kind.value,
            "action": self.action.value,
            "left_item_id": self.left_item_id,
            "right_item_id": self.right_item_id,
            "corpus_ids": list(self.corpus_ids),
            "score": self.score,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class ContaminationAuditReport:
    sources: tuple[CorpusSource, ...]
    policy: ContaminationPolicy
    findings: tuple[ContaminationFinding, ...]
    audited_item_count: int
    compared_pair_count: int
    status: ContaminationStatus
    protocol_revision: str = CONTAMINATION_PROTOCOL_REVISION

    def __post_init__(self) -> None:
        if self.audited_item_count < 0 or self.compared_pair_count < 0:
            raise ContaminationAuditError("audit counts cannot be negative")
        if self.status is ContaminationStatus.CLEAN and self.findings:
            raise ContaminationAuditError("clean audits cannot contain findings")

    @property
    def report_id(self) -> str:
        return (
            "contamination_"
            + hashlib.sha256(_canonical(self.to_record(False)).encode()).hexdigest()
        )

    def require_publishable(self) -> None:
        if self.status is not ContaminationStatus.CLEAN:
            raise ContaminationAuditError(f"contamination audit is {self.status.value}")

    def remediated_protocol_id(self, original_protocol_id: str) -> str:
        _require_text(original_protocol_id, "original protocol ID")
        finding_ids = [
            item.finding_id for item in self.findings if item.action is FindingAction.MOVE
        ]
        return (
            "protocol_"
            + hashlib.sha256(
                _canonical(
                    {"original": original_protocol_id, "moved": finding_ids}
                ).encode()
            ).hexdigest()
        )

    def to_record(self, include_id: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": CONTAMINATION_AUDIT_SCHEMA_VERSION,
            "protocol_revision": self.protocol_revision,
            "sources": [source.to_record() for source in self.sources],
            "policy": self.policy.to_record(),
            "findings": [finding.to_record() for finding in self.findings],
            "audited_item_count": self.audited_item_count,
            "compared_pair_count": self.compared_pair_count,
            "status": self.status.value,
        }
        if include_id:
            record["report_id"] = self.report_id
        return record


def _near_score(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def audit_corpora(
    sources: tuple[CorpusSource, ...],
    items: tuple[CorpusItem, ...],
    policy: ContaminationPolicy | None = None,
) -> ContaminationAuditReport:
    """Audit exact, bounded near, ancestry, license, and availability overlap."""
    selected = policy or ContaminationPolicy()
    if len(items) > selected.max_items:
        finding = ContaminationFinding(
            FindingKind.RESOURCE_LIMIT,
            FindingAction.LIMITATION,
            None,
            None,
            ("__audit__",),
            None,
            "item count exceeds audit limit",
        )
        return ContaminationAuditReport(
            sources,
            selected,
            (finding,),
            0,
            0,
            ContaminationStatus.UNKNOWN,
        )
    source_by_id = {source.corpus_id: source for source in sources}
    declared_item_sources = {item.corpus_id for item in items}
    if len(source_by_id) != len(sources) or declared_item_sources - set(source_by_id):
        raise ContaminationAuditError(
            "source IDs must be unique and every item source must be declared"
        )
    if any(item.corpus_id not in source_by_id for item in items):
        raise ContaminationAuditError("every item must reference a declared source")
    findings: list[ContaminationFinding] = []
    for source in sources:
        if not source.available:
            findings.append(
                ContaminationFinding(
                    FindingKind.UNAVAILABLE_SOURCE,
                    FindingAction.LIMITATION,
                    None,
                    None,
                    (source.corpus_id,),
                    None,
                    "source data was unavailable; cleanliness cannot be claimed",
                )
            )
        if selected.license_safe_only and source.license.casefold() in {
            "unknown",
            "pending",
            "restricted",
        }:
            findings.append(
                ContaminationFinding(
                    FindingKind.LICENSE_LIMITATION,
                    FindingAction.LIMITATION,
                    None,
                    None,
                    (source.corpus_id,),
                    None,
                        "source license was not auditable under the safe-only policy",
                )
            )
    exact: dict[str, list[CorpusItem]] = defaultdict(list)
    for item in items:
        exact[item.text_digest].append(item)
    for same in exact.values():
        for index, left in enumerate(same):
            for right in same[index + 1 :]:
                if left.corpus_id == right.corpus_id:
                    continue
                ids = tuple(sorted((left.corpus_id, right.corpus_id)))
                findings.append(
                    ContaminationFinding(
                        FindingKind.EXACT_OVERLAP,
                        FindingAction.BLOCK,
                        left.item_id,
                        right.item_id,
                        ids,
                        1.0,
                        "identical normalized text digest across corpus boundaries",
                    )
                )
    pair_count = 0
    shingles = {item.item_id: item.shingles(selected.ngram_size) for item in items}
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            if left.corpus_id == right.corpus_id:
                continue
            pair_count += 1
            if pair_count > selected.max_pair_comparisons:
                findings.append(
                    ContaminationFinding(
                        FindingKind.RESOURCE_LIMIT,
                        FindingAction.LIMITATION,
                        None,
                        None,
                        tuple(sorted((left.corpus_id, right.corpus_id))),
                        None,
                        "near-duplicate comparison budget exhausted",
                    )
                )
                break
            score = _near_score(shingles[left.item_id], shingles[right.item_id])
            if score >= selected.near_jaccard_threshold and left.text_digest != right.text_digest:
                findings.append(
                    ContaminationFinding(
                        FindingKind.NEAR_OVERLAP,
                        FindingAction.MOVE,
                        left.item_id,
                        right.item_id,
                        tuple(sorted((left.corpus_id, right.corpus_id))),
                        score,
                        "bounded token-shingle Jaccard overlap exceeds threshold",
                    )
                )
        if pair_count > selected.max_pair_comparisons:
            break
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            shared = set(left.ancestry_ids) & set(right.ancestry_ids)
            if left.corpus_id != right.corpus_id and shared:
                findings.append(
                    ContaminationFinding(
                        FindingKind.ANCESTRY_OVERLAP,
                        FindingAction.BLOCK,
                        left.item_id,
                        right.item_id,
                        tuple(sorted((left.corpus_id, right.corpus_id))),
                        None,
                        f"shared model ancestry: {','.join(sorted(shared))}",
                    )
                )
    findings.sort(key=lambda item: item.finding_id)
    blocking = any(item.action in {FindingAction.BLOCK, FindingAction.MOVE} for item in findings)
    limited = any(
        item.kind
        in {
            FindingKind.UNAVAILABLE_SOURCE,
            FindingKind.LICENSE_LIMITATION,
            FindingKind.RESOURCE_LIMIT,
        }
        for item in findings
    )
    status = (
        ContaminationStatus.BLOCKED
        if blocking
        else ContaminationStatus.UNKNOWN
        if limited
        else ContaminationStatus.CLEAN
    )
    return ContaminationAuditReport(
        sources, selected, tuple(findings), len(items), pair_count, status
    )
