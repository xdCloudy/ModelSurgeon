"""Non-overwriting checkpoint staging and atomic destination publication."""

from __future__ import annotations

import os
import shutil
import uuid
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Self


class CheckpointDestinationError(RuntimeError):
    """Raised when checkpoint source/destination safety cannot be guaranteed."""


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().absolute().resolve(strict=False)


def _same_existing(left: Path, right: Path) -> bool:
    if not left.exists() or not right.exists():
        return False
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _remove_staging(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


class AtomicCheckpointDestination(AbstractContextManager["AtomicCheckpointDestination"]):
    """Own one sibling staging path and publish it without overwriting a destination."""

    def __init__(self, source: str | Path, destination: str | Path) -> None:
        self.source = _resolved(source)
        self.destination = _resolved(destination)
        if not self.source.exists():
            raise CheckpointDestinationError("source checkpoint does not exist")
        self._validate_destination_identity()
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.staging_path = self.destination.parent / (
            f".{self.destination.name}.{uuid.uuid4().hex}.staging"
        )
        self._entered = False
        self._published = False

    @property
    def published(self) -> bool:
        return self._published

    def _validate_destination_identity(self) -> None:
        if self.source == self.destination or _same_existing(self.source, self.destination):
            raise CheckpointDestinationError(
                "source and destination resolve to the same checkpoint"
            )
        if self.destination in self.source.parents:
            raise CheckpointDestinationError("destination cannot be an ancestor of the source")
        if self.source.is_dir() and self.source in self.destination.parents:
            raise CheckpointDestinationError("destination cannot be inside the source checkpoint")
        if self.destination.exists() or self.destination.is_symlink():
            raise CheckpointDestinationError("checkpoint destination already exists")

    def __enter__(self) -> Self:
        if self._entered:
            raise CheckpointDestinationError("checkpoint destination context cannot be reused")
        self._validate_destination_identity()
        _remove_staging(self.staging_path)
        self._entered = True
        return self

    def publish(self) -> Path:
        if not self._entered:
            raise CheckpointDestinationError("checkpoint staging context is not active")
        if self._published:
            raise CheckpointDestinationError("checkpoint destination was already published")
        self._validate_destination_identity()
        if not self.staging_path.exists():
            raise CheckpointDestinationError("checkpoint staging output does not exist")

        if self.staging_path.is_file():
            # Windows' CRT requires a writable descriptor for ``fsync``.
            # The file already exists and r+b preserves its contents.
            with self.staging_path.open("r+b") as stream:
                os.fsync(stream.fileno())
            try:
                os.link(self.staging_path, self.destination)
            except FileExistsError as error:
                raise CheckpointDestinationError(
                    "checkpoint destination appeared during publication"
                ) from error
            self.staging_path.unlink()
        elif self.staging_path.is_dir():
            if self.destination.exists() or self.destination.is_symlink():
                raise CheckpointDestinationError(
                    "checkpoint destination appeared during publication"
                )
            os.rename(self.staging_path, self.destination)
        else:
            raise CheckpointDestinationError("checkpoint staging output has unsupported type")

        self._published = True
        return self.destination

    def close(self) -> None:
        if not self._published:
            _remove_staging(self.staging_path)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()
