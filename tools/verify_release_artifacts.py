"""Verify release archives, metadata, and the release tag before publication."""

from __future__ import annotations

import argparse
import email
import re
import tarfile
import tomllib
import zipfile
from collections.abc import Iterable
from pathlib import Path

_PACKAGE_NAME = "modelsurgeon"
_VERSION_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)+(?:[.-][0-9A-Za-z.-]+)?$")


class ReleaseArtifactError(ValueError):
    """Raised when a release artifact is unsafe or inconsistent."""


def _metadata(raw: bytes) -> tuple[str, str, str]:
    message = email.message_from_bytes(raw)
    name = message.get("Name")
    version = message.get("Version")
    requires_python = message.get("Requires-Python")
    if not all(isinstance(value, str) for value in (name, version, requires_python)):
        raise ReleaseArtifactError("package metadata is missing Name, Version, or Requires-Python")
    return name, version, requires_python


def _validate_metadata(raw: bytes, expected_version: str, source: Path) -> None:
    name, version, requires_python = _metadata(raw)
    if name != _PACKAGE_NAME:
        raise ReleaseArtifactError(f"{source.name} has unexpected package name {name!r}")
    if version != expected_version:
        raise ReleaseArtifactError(
            f"{source.name} has version {version!r}, expected {expected_version!r}"
        )
    if not requires_python.startswith(">=3.12"):
        raise ReleaseArtifactError(
            f"{source.name} must require Python 3.12 or newer, got {requires_python!r}"
        )


def _project_version(project: Path) -> str:
    try:
        with project.open("rb") as handle:
            value = tomllib.load(handle)["project"]["version"]
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as error:
        raise ReleaseArtifactError("project metadata has no valid version") from error
    if not isinstance(value, str) or not _VERSION_PATTERN.fullmatch(value):
        raise ReleaseArtifactError(f"invalid project version {value!r}")
    return value


def _validate_archive_paths(names: Iterable[str], source: Path) -> None:
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ReleaseArtifactError(f"{source.name} contains an unsafe archive path {name!r}")


def _verify_wheel(path: Path, expected_version: str) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            _validate_archive_paths(names, path)
            metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(metadata_names) != 1:
                raise ReleaseArtifactError(f"{path.name} must contain one dist-info METADATA file")
            if "modelsurgeon/__init__.py" not in names:
                raise ReleaseArtifactError(f"{path.name} does not contain the package module")
            _validate_metadata(archive.read(metadata_names[0]), expected_version, path)
    except (OSError, zipfile.BadZipFile) as error:
        raise ReleaseArtifactError(f"could not read wheel {path.name}") from error


def _verify_sdist(path: Path, expected_version: str) -> None:
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = archive.getmembers()
            _validate_archive_paths((member.name for member in members), path)
            metadata_members = [member for member in members if member.name.endswith("/PKG-INFO")]
            if len(metadata_members) != 1:
                raise ReleaseArtifactError(f"{path.name} must contain one PKG-INFO file")
            metadata_file = archive.extractfile(metadata_members[0])
            if metadata_file is None:
                raise ReleaseArtifactError(f"{path.name} has an unreadable PKG-INFO file")
            _validate_metadata(metadata_file.read(), expected_version, path)
    except (OSError, tarfile.TarError) as error:
        raise ReleaseArtifactError(f"could not read source archive {path.name}") from error


def verify_release_artifacts(
    dist: Path,
    *,
    project: Path = Path("pyproject.toml"),
    tag: str | None = None,
) -> tuple[Path, Path]:
    """Verify exactly one versioned wheel and sdist against project metadata."""

    if not dist.is_dir():
        raise ReleaseArtifactError(f"distribution directory does not exist: {dist}")
    expected_version = _project_version(project)
    if tag is not None and tag != f"v{expected_version}":
        raise ReleaseArtifactError(
            f"release tag {tag!r} does not match project version v{expected_version}"
        )
    wheels = sorted(dist.glob(f"{_PACKAGE_NAME}-{expected_version}-*.whl"))
    sdists = sorted(dist.glob(f"{_PACKAGE_NAME}-{expected_version}.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ReleaseArtifactError(
            f"expected exactly one wheel and sdist for {expected_version}; "
            f"found {len(wheels)} wheel(s), {len(sdists)} sdist(s)"
        )
    _verify_wheel(wheels[0], expected_version)
    _verify_sdist(sdists[0], expected_version)
    return wheels[0], sdists[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--tag")
    args = parser.parse_args()
    wheel, sdist = verify_release_artifacts(args.dist, project=args.project, tag=args.tag)
    print(f"verified {wheel.name}")
    print(f"verified {sdist.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
