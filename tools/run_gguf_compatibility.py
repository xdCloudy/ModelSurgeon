"""Run the real GGUF compatibility matrix against a pinned llama.cpp executable."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.evaluation.gguf_compatibility import (
    GGUFCompatibilityCell,
    GGUFCompatibilityKey,
    GGUFCompatibilityOperation,
    GGUFCompatibilityStatus,
    GGUFStorageProfile,
    build_complete_matrix,
    load_forward_evidence,
)
from modelsurgeon.evaluation.llama_cpp import LlamaCppValidationConfig, validate_generated_gguf

_FAMILIES = tuple(ModelFamily)
_PROFILES = tuple(GGUFStorageProfile)
_OPERATIONS = tuple(GGUFCompatibilityOperation)


@dataclass(frozen=True, slots=True)
class Fixture:
    family: ModelFamily
    profile: GGUFStorageProfile
    path: Path
    source_identifier: str
    source_revision: str


def _fixture(value: str) -> Fixture:
    parts = value.split("=", 4)
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "fixture must be FAMILY=PROFILE=PATH=SOURCE_IDENTIFIER=SOURCE_REVISION"
        )
    family, profile, path, identifier, revision = parts
    try:
        return Fixture(
            ModelFamily(family),
            GGUFStorageProfile(profile),
            Path(path),
            identifier,
            revision,
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cli", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fixture", action="append", default=[], type=_fixture)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    fixtures: dict[tuple[ModelFamily, GGUFStorageProfile], Fixture] = {}
    for fixture in arguments.fixture:
        key = (fixture.family, fixture.profile)
        if key in fixtures:
            label = f"{fixture.family.value}/{fixture.profile.value}"
            raise ValueError(f"duplicate fixture for {label}")
        fixtures[key] = fixture

    cells: list[GGUFCompatibilityCell] = []
    config = LlamaCppValidationConfig(
        executable=arguments.llama_cli,
        timeout_seconds=arguments.timeout_seconds,
    )
    for family in _FAMILIES:
        for profile in _PROFILES:
            fixture = fixtures.get((family, profile))
            key = GGUFCompatibilityKey(
                family,
                profile,
                GGUFCompatibilityOperation.LOAD_FORWARD,
            )
            if fixture is None:
                cells.append(
                    GGUFCompatibilityCell(
                        key,
                        GGUFCompatibilityStatus.UNSUPPORTED,
                        "no pinned generated fixture and load/forward evidence",
                    )
                )
            else:
                report = validate_generated_gguf(fixture.path, config=config)
                evidence = load_forward_evidence(
                    report,
                    family=family,
                    storage_profile=profile,
                    source_identifier=fixture.source_identifier,
                    source_revision=fixture.source_revision,
                )
                status = (
                    GGUFCompatibilityStatus.RUNTIME_VERIFIED
                    if evidence.successful
                    else GGUFCompatibilityStatus.FAILED
                )
                cells.append(
                    GGUFCompatibilityCell(
                        key,
                        status,
                        "pinned llama.cpp loaded the artifact and completed a forward pass"
                        if evidence.successful
                        else "pinned llama.cpp load/forward validation failed",
                        evidence,
                    )
                )

            for operation in _OPERATIONS:
                if operation is GGUFCompatibilityOperation.LOAD_FORWARD:
                    continue
                implemented = not (
                    operation
                    in {
                        GGUFCompatibilityOperation.MLP_REMOVAL,
                        GGUFCompatibilityOperation.ATTENTION_REMOVAL,
                    }
                    and family in {ModelFamily.MISTRAL, ModelFamily.GEMMA}
                )
                cells.append(
                    GGUFCompatibilityCell(
                        GGUFCompatibilityKey(family, profile, operation),
                        GGUFCompatibilityStatus.STRUCTURAL_ONLY
                        if implemented
                        else GGUFCompatibilityStatus.UNSUPPORTED,
                        (
                            "focused native parser/planner/executor tests exist; no external "
                            "post-surgery load/forward fixture is claimed"
                            if implemented
                            else "native MLP and attention removal intentionally support only "
                            "Llama and dense Qwen"
                        ),
                    )
                )

    matrix = build_complete_matrix(
        families=_FAMILIES,
        storage_profiles=_PROFILES,
        operations=_OPERATIONS,
        cells=cells,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(matrix.to_record(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    matrix.require_no_failures()


if __name__ == "__main__":
    main()
