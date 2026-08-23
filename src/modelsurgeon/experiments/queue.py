"""Expiring candidate work leases with heartbeat and stale-worker protection."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass

from modelsurgeon.experiments.state_machine import CandidateState
from modelsurgeon.experiments.store import ExperimentMetadataStore, ExperimentStoreError


class WorkLeaseError(RuntimeError):
    """Raised when a worker attempts to mutate a lease it no longer owns."""


@dataclass(frozen=True, slots=True)
class WorkLease:
    candidate_id: str
    attempt_id: str
    worker_id: str
    lease_token: str
    generation: int
    acquired_at_ns: int
    heartbeat_at_ns: int
    expires_at_ns: int
    completed_at_ns: int | None

    @property
    def completed(self) -> bool:
        return self.completed_at_ns is not None

    def expired(self, now_ns: int) -> bool:
        return not self.completed and self.expires_at_ns <= now_ns


_TERMINAL_STATES = frozenset(
    {
        CandidateState.SUCCEEDED.value,
        CandidateState.REJECTED.value,
        CandidateState.FAILED.value,
    }
)


def _validate_identity(name: str, value: str) -> None:
    if not value:
        raise WorkLeaseError(f"{name} must be non-empty")


def _validate_time(now_ns: int) -> None:
    if isinstance(now_ns, bool) or now_ns < 0:
        raise WorkLeaseError("lease timestamps must be non-negative integer nanoseconds")


def _row_to_lease(row: object) -> WorkLease:
    # sqlite3.Row is intentionally accessed structurally to keep this module store-local.
    item = row  # mypy narrows values through indexed access below.
    return WorkLease(
        str(item["candidate_id"]),  # type: ignore[index]
        str(item["attempt_id"]),  # type: ignore[index]
        str(item["worker_id"]),  # type: ignore[index]
        str(item["lease_token"]),  # type: ignore[index]
        int(item["generation"]),  # type: ignore[index]
        int(item["acquired_at_ns"]),  # type: ignore[index]
        int(item["heartbeat_at_ns"]),  # type: ignore[index]
        int(item["expires_at_ns"]),  # type: ignore[index]
        None
        if item["completed_at_ns"] is None  # type: ignore[index]
        else int(item["completed_at_ns"]),  # type: ignore[index]
    )


class ExperimentWorkQueue:
    """Lease candidate work atomically through the experiment metadata database."""

    def __init__(
        self,
        store: ExperimentMetadataStore,
        *,
        lease_duration_ns: int,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        if isinstance(lease_duration_ns, bool) or lease_duration_ns <= 0:
            raise WorkLeaseError("lease duration must be positive integer nanoseconds")
        self.store = store
        self.lease_duration_ns = lease_duration_ns
        self._token_factory = token_factory or (lambda: uuid.uuid4().hex)

    def _new_token(self) -> str:
        token = self._token_factory()
        if not token:
            raise WorkLeaseError("lease token factory returned an empty token")
        return token

    def claim(
        self,
        candidate_id: str,
        *,
        attempt_id: str,
        worker_id: str,
        now_ns: int,
    ) -> WorkLease | None:
        """Claim unowned/expired work; active, completed, or terminal work returns ``None``."""

        for name, value in (
            ("candidate_id", candidate_id),
            ("attempt_id", attempt_id),
            ("worker_id", worker_id),
        ):
            _validate_identity(name, value)
        _validate_time(now_ns)
        token = self._new_token()
        expires_at_ns = now_ns + self.lease_duration_ns
        try:
            with self.store._write() as connection:
                candidate = connection.execute(
                    "SELECT 1 FROM experiment_candidates WHERE candidate_id = ?",
                    (candidate_id,),
                ).fetchone()
                if candidate is None:
                    raise WorkLeaseError(f"unknown experiment candidate {candidate_id}")
                state_row = connection.execute(
                    """
                    SELECT state FROM experiment_state_events
                    WHERE candidate_id = ? ORDER BY sequence DESC LIMIT 1
                    """,
                    (candidate_id,),
                ).fetchone()
                if state_row is not None and str(state_row["state"]) in _TERMINAL_STATES:
                    return None
                row = connection.execute(
                    "SELECT * FROM experiment_work_leases WHERE candidate_id = ?",
                    (candidate_id,),
                ).fetchone()
                if row is not None:
                    existing = _row_to_lease(row)
                    if existing.completed or existing.expires_at_ns > now_ns:
                        return None
                    generation = existing.generation + 1
                    connection.execute(
                        """
                        UPDATE experiment_work_leases
                        SET attempt_id = ?, worker_id = ?, lease_token = ?, generation = ?,
                            acquired_at_ns = ?, heartbeat_at_ns = ?, expires_at_ns = ?,
                            completed_at_ns = NULL
                        WHERE candidate_id = ?
                        """,
                        (
                            attempt_id,
                            worker_id,
                            token,
                            generation,
                            now_ns,
                            now_ns,
                            expires_at_ns,
                            candidate_id,
                        ),
                    )
                else:
                    generation = 1
                    connection.execute(
                        """
                        INSERT INTO experiment_work_leases(
                            candidate_id, attempt_id, worker_id, lease_token, generation,
                            acquired_at_ns, heartbeat_at_ns, expires_at_ns, completed_at_ns
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                        """,
                        (
                            candidate_id,
                            attempt_id,
                            worker_id,
                            token,
                            generation,
                            now_ns,
                            now_ns,
                            expires_at_ns,
                        ),
                    )
                return WorkLease(
                    candidate_id,
                    attempt_id,
                    worker_id,
                    token,
                    generation,
                    now_ns,
                    now_ns,
                    expires_at_ns,
                    None,
                )
        except ExperimentStoreError as error:
            raise WorkLeaseError(str(error)) from error

    def heartbeat(self, lease_token: str, *, now_ns: int) -> WorkLease:
        """Extend only a live current lease; expired or replaced workers fail closed."""

        _validate_identity("lease_token", lease_token)
        _validate_time(now_ns)
        expires_at_ns = now_ns + self.lease_duration_ns
        try:
            with self.store._write() as connection:
                row = connection.execute(
                    "SELECT * FROM experiment_work_leases WHERE lease_token = ?",
                    (lease_token,),
                ).fetchone()
                if row is None:
                    raise WorkLeaseError("lease is not current")
                lease = _row_to_lease(row)
                if lease.completed:
                    raise WorkLeaseError("completed leases cannot heartbeat")
                if lease.expires_at_ns <= now_ns:
                    raise WorkLeaseError("lease expired before heartbeat")
                connection.execute(
                    """
                    UPDATE experiment_work_leases
                    SET heartbeat_at_ns = ?, expires_at_ns = ?
                    WHERE candidate_id = ? AND lease_token = ?
                    """,
                    (now_ns, expires_at_ns, lease.candidate_id, lease_token),
                )
                return WorkLease(
                    lease.candidate_id,
                    lease.attempt_id,
                    lease.worker_id,
                    lease.lease_token,
                    lease.generation,
                    lease.acquired_at_ns,
                    now_ns,
                    expires_at_ns,
                    None,
                )
        except ExperimentStoreError as error:
            raise WorkLeaseError(str(error)) from error

    def complete(self, lease_token: str, *, now_ns: int) -> WorkLease:
        """Complete the current lease once; repeating the same completion is idempotent."""

        _validate_identity("lease_token", lease_token)
        _validate_time(now_ns)
        try:
            with self.store._write() as connection:
                row = connection.execute(
                    "SELECT * FROM experiment_work_leases WHERE lease_token = ?",
                    (lease_token,),
                ).fetchone()
                if row is None:
                    raise WorkLeaseError("lease is not current")
                lease = _row_to_lease(row)
                if lease.completed:
                    return lease
                if lease.expires_at_ns <= now_ns:
                    raise WorkLeaseError("expired lease cannot complete")
                connection.execute(
                    """
                    UPDATE experiment_work_leases
                    SET completed_at_ns = ?
                    WHERE candidate_id = ? AND lease_token = ? AND completed_at_ns IS NULL
                    """,
                    (now_ns, lease.candidate_id, lease_token),
                )
                return WorkLease(
                    lease.candidate_id,
                    lease.attempt_id,
                    lease.worker_id,
                    lease.lease_token,
                    lease.generation,
                    lease.acquired_at_ns,
                    lease.heartbeat_at_ns,
                    lease.expires_at_ns,
                    now_ns,
                )
        except ExperimentStoreError as error:
            raise WorkLeaseError(str(error)) from error

    def current(self, candidate_id: str) -> WorkLease | None:
        _validate_identity("candidate_id", candidate_id)
        try:
            with self.store.reader() as connection:
                row = connection.execute(
                    "SELECT * FROM experiment_work_leases WHERE candidate_id = ?",
                    (candidate_id,),
                ).fetchone()
        except ExperimentStoreError as error:
            raise WorkLeaseError(str(error)) from error
        return None if row is None else _row_to_lease(row)
