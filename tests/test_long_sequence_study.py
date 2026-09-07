import pytest

from modelsurgeon.evaluation import (
    DEFAULT_LONG_SEQUENCE_STUDY,
    AdversarialInteraction,
    AdversarialKind,
    AdversarialOutcome,
    HorizonOutcome,
    LongSequenceEvidence,
    LongSequenceKey,
    LongSequencePolicy,
    LongSequenceStudyError,
    summarize_long_sequence_study,
)


def _evidence(
    family: str,
    policy: LongSequencePolicy,
    horizon: int,
    seed: int,
) -> LongSequenceEvidence:
    return LongSequenceEvidence(
        LongSequenceKey(family, policy, horizon, seed),
        HorizonOutcome.MEASURED,
        f"state_{family}_{policy.value}_{horizon}_{seed}",
        "a" * 64,
        0.9,
        1.0 / horizon,
        0,
        horizon,
        0,
        float(horizon),
        None,
        tuple(
            AdversarialInteraction(kind, AdversarialOutcome.REJECTED, 0.1, 0, "safe rejection")
            for kind in AdversarialKind
        ),
        (("checkpoint", "fixture-checkpoint-v1"), ("corpus", "heldout-corpus-v1")),
    )


def test_default_long_sequence_study_retains_unavailable_physical_cells() -> None:
    study = DEFAULT_LONG_SEQUENCE_STUDY
    assert len(study.evidence) == 2 * 3 * 3 * 3
    assert study.claim.value == "unsupported"
    assert all(item.outcome is HorizonOutcome.UNSUPPORTED for item in study.evidence)


def test_measured_horizons_retain_adversarial_cases_and_bootstrap_summary() -> None:
    evidence = tuple(
        _evidence(family, policy, horizon, seed)
        for family in ("llama", "qwen")
        for policy in LongSequencePolicy
        for horizon in (10, 20, 50)
        for seed in (11, 23, 47)
    )
    study = summarize_long_sequence_study(
        evidence,
        families=("llama", "qwen"),
        seeds=(11, 23, 47),
        bootstrap_repetitions=100,
    )
    assert len(study.summaries) == 9
    assert all(item.regret_low is not None for item in study.summaries)
    assert all(len(item.adversarial) == 3 for item in study.evidence)


def test_long_sequence_contract_rejects_missing_checkpoint_or_adversarial_cell() -> None:
    with pytest.raises(LongSequenceStudyError, match="checkpoint"):
        LongSequenceEvidence(
            LongSequenceKey("llama", LongSequencePolicy.STATE_AWARE, 10, 11),
            HorizonOutcome.MEASURED,
            None,
            None,
            0.9,
            0.1,
            0,
            10,
            0,
            1.0,
            None,
            (),
            (("protocol", "fixture"),),
        )
