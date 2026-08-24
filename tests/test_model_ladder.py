from __future__ import annotations

import json

import pytest

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.evaluation.model_ladder import (
    PERMISSIVE_MODEL_LADDER,
    ModelDatasetProtocol,
    ModelEvaluationLadder,
    ModelHardwareMode,
    ModelLadderError,
    ModelLadderTarget,
)


def test_ladder_is_complete_permissive_immutable_and_multi_family() -> None:
    ladder = PERMISSIVE_MODEL_LADDER

    assert [target.rung for target in ladder.targets] == [
        "100M",
        "300M",
        "500M",
        "1B",
        "1.5B",
        "3B",
        "7B",
    ]
    assert {target.family for target in ladder.targets} == {
        ModelFamily.LLAMA,
        ModelFamily.QWEN,
        ModelFamily.MISTRAL,
    }
    assert all(target.license == "apache-2.0" for target in ladder.targets)
    assert all(len(target.revision) == 40 for target in ladder.targets)
    assert all(target.public and not target.gated for target in ladder.targets)
    assert len({(target.identifier, target.revision) for target in ladder.targets}) == 7


def test_every_target_declares_purpose_hardware_and_dataset_compatibility() -> None:
    for target in PERMISSIVE_MODEL_LADDER.targets:
        assert target.purpose
        assert target.hardware_modes
        assert target.datasets == (
            ModelDatasetProtocol.LOCAL_SMOKE,
            ModelDatasetProtocol.WIKITEXT_PERPLEXITY,
        )
    assert ModelHardwareMode.CPU_SMOKE in PERMISSIVE_MODEL_LADDER.targets[0].hardware_modes
    assert (
        ModelHardwareMode.GPU_12_GB_QUANTIZED
        in PERMISSIVE_MODEL_LADDER.targets[-1].hardware_modes
    )


def test_ladder_record_is_canonical_and_never_claims_committed_weights() -> None:
    record = PERMISSIVE_MODEL_LADDER.to_record()
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))

    assert record["weights_committed_to_git"] is False
    assert len(PERMISSIVE_MODEL_LADDER.ladder_id) == 64
    assert PERMISSIVE_MODEL_LADDER.ladder_id == (
        "dcb64f73295bcba8af2999e3e2bd5cdc189f48a2b5ce99962d22c3f92f94edb0"
    )
    assert json.loads(canonical) == record


def test_nonpermissive_gated_and_badly_sized_targets_fail_closed() -> None:
    baseline = PERMISSIVE_MODEL_LADDER.targets[0]
    with pytest.raises(ModelLadderError, match="not permissive"):
        ModelLadderTarget(
            baseline.rung,
            baseline.target_parameters,
            baseline.identifier,
            baseline.revision,
            baseline.family,
            baseline.actual_parameters,
            "other",
            baseline.last_modified,
            baseline.purpose,
            baseline.hardware_modes,
            baseline.datasets,
        )
    with pytest.raises(ModelLadderError, match="public and ungated"):
        ModelLadderTarget(
            baseline.rung,
            baseline.target_parameters,
            baseline.identifier,
            baseline.revision,
            baseline.family,
            baseline.actual_parameters,
            baseline.license,
            baseline.last_modified,
            baseline.purpose,
            baseline.hardware_modes,
            baseline.datasets,
            gated=True,
        )
    with pytest.raises(ModelLadderError, match="outside"):
        ModelLadderTarget(
            baseline.rung,
            baseline.target_parameters,
            baseline.identifier,
            baseline.revision,
            baseline.family,
            1_000_000,
            baseline.license,
            baseline.last_modified,
            baseline.purpose,
            baseline.hardware_modes,
            baseline.datasets,
        )


def test_ladder_rejects_missing_rungs_and_insufficient_family_coverage() -> None:
    with pytest.raises(ModelLadderError, match="ordered rungs"):
        ModelEvaluationLadder(
            PERMISSIVE_MODEL_LADDER.targets[:-1],
            PERMISSIVE_MODEL_LADDER.metadata_verified_on,
        )
    llama_only = tuple(
        ModelLadderTarget(
            target.rung,
            target.target_parameters,
            f"example/model-{index}",
            f"{index + 1:040x}",
            ModelFamily.LLAMA,
            target.target_parameters,
            "apache-2.0",
            target.last_modified,
            target.purpose,
            target.hardware_modes,
            target.datasets,
        )
        for index, target in enumerate(PERMISSIVE_MODEL_LADDER.targets)
    )
    with pytest.raises(ModelLadderError, match="three architecture families"):
        ModelEvaluationLadder(
            llama_only,
            PERMISSIVE_MODEL_LADDER.metadata_verified_on,
        )
