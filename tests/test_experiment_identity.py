"""Tests for canonical experiment, run, and candidate identity derivation."""

from __future__ import annotations

import math

import pytest

from modelsurgeon.experiments import (
    DatasetTarget,
    ExperimentIdentityError,
    ExperimentIdentitySpec,
    ModelTarget,
    PathAlias,
    SeedContext,
    canonical_identity_json,
    derive_candidate_identity,
    derive_experiment_identity,
    derive_run_identity,
)


def _dataset() -> DatasetTarget:
    return DatasetTarget(
        "dataset/example",
        "dataset-rev-1",
        "validation",
        "manifest-1",
        "tokenizer/example",
        "tokenizer-rev-1",
    )


def _spec(
    *,
    model_identifier: str,
    model_revision: str = "model-rev-1",
    config: dict[str, object] | None = None,
    alias: PathAlias,
    seed: int = 7,
) -> ExperimentIdentitySpec:
    return ExperimentIdentitySpec(
        ModelTarget(model_identifier, model_revision, "llama", "safetensors", 128),
        _dataset(),
        config
        or {
            "evaluation": {"max_tier": 2, "threshold": 0.1},
            "model_cache": f"{alias.local_root}/cache",
        },
        SeedContext(seed, 11, 13),
        "tool-rev-1",
        "evaluator-v1",
        1,
        1,
        (alias,),
    )


def test_key_order_and_machine_local_path_aliases_do_not_change_identity() -> None:
    windows_alias = PathAlias("model-root", r"D:\models")
    linux_alias = PathAlias("model-root", "/srv/models")
    windows = _spec(
        model_identifier=r"D:\models\tiny-model",
        config={
            "model_cache": r"D:\models\cache",
            "evaluation": {"threshold": 0.1, "max_tier": 2},
        },
        alias=windows_alias,
    )
    linux = _spec(
        model_identifier="/srv/models/tiny-model",
        config={
            "evaluation": {"max_tier": 2, "threshold": 0.1},
            "model_cache": "/srv/models/cache",
        },
        alias=linux_alias,
    )

    first = derive_experiment_identity(windows)
    second = derive_experiment_identity(linux)
    assert first.experiment_id == second.experiment_id
    assert first.config_digest == second.config_digest
    assert "D:" not in first.canonical_payload
    assert "/srv/models" not in second.canonical_payload
    assert "@model-root/tiny-model" in first.canonical_payload


def test_meaningful_config_revision_and_seed_changes_change_identity() -> None:
    alias = PathAlias("model-root", "/models")
    baseline = derive_experiment_identity(
        _spec(model_identifier="/models/tiny", alias=alias)
    )
    changed_revision = derive_experiment_identity(
        _spec(model_identifier="/models/tiny", model_revision="model-rev-2", alias=alias)
    )
    changed_config = derive_experiment_identity(
        _spec(
            model_identifier="/models/tiny",
            config={"evaluation": {"max_tier": 3}, "model_cache": "/models/cache"},
            alias=alias,
        )
    )
    changed_seed = derive_experiment_identity(
        _spec(model_identifier="/models/tiny", alias=alias, seed=8)
    )

    assert len(
        {
            baseline.experiment_id,
            changed_revision.experiment_id,
            changed_config.experiment_id,
            changed_seed.experiment_id,
        }
    ) == 4
    assert baseline.config_digest != changed_config.config_digest
    assert baseline.config_digest == changed_revision.config_digest


def test_run_and_candidate_ids_are_stable_but_logically_separate() -> None:
    alias = PathAlias("model-root", "/models")
    experiment = derive_experiment_identity(
        _spec(model_identifier="/models/tiny", alias=alias)
    )
    run = derive_run_identity(experiment.experiment_id, "campaign-shard-0")
    same_run = derive_run_identity(experiment.experiment_id, "campaign-shard-0")
    other_run = derive_run_identity(experiment.experiment_id, "campaign-shard-1")

    assert run.run_id == same_run.run_id
    assert run.run_id != other_run.run_id
    assert run.run_id.startswith("run_")

    candidate = derive_candidate_identity(run.run_id, "mutation-a")
    same_candidate = derive_candidate_identity(run.run_id, "mutation-a")
    other_candidate = derive_candidate_identity(run.run_id, "mutation-b")
    assert candidate.candidate_id == same_candidate.candidate_id
    assert candidate.candidate_id != other_candidate.candidate_id
    assert candidate.candidate_id.startswith("cand_")


def test_canonical_json_rejects_unsupported_and_nonfinite_values() -> None:
    assert canonical_identity_json({"b": 2, "a": [3, 1]}) == '{"a":[3,1],"b":2}'
    with pytest.raises(ExperimentIdentityError, match="non-finite"):
        canonical_identity_json({"value": math.inf})
    with pytest.raises(ExperimentIdentityError, match="JSON-compatible"):
        canonical_identity_json({"value": {1, 2}})
    with pytest.raises(ExperimentIdentityError, match="string keys"):
        canonical_identity_json({1: "bad"})


def test_invalid_identity_and_alias_contracts_fail_closed() -> None:
    with pytest.raises(ExperimentIdentityError, match="delimiters"):
        PathAlias("bad/name", "/models")
    with pytest.raises(ExperimentIdentityError, match="experiment ID"):
        derive_run_identity("not-an-experiment")
    with pytest.raises(ExperimentIdentityError, match="run ID"):
        derive_candidate_identity("not-a-run", "mutation")
