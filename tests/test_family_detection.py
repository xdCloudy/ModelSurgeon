import json

import pytest

from modelsurgeon.adapters import (
    ArchitectureEvidence,
    ConflictingArchitectureError,
    ModelFamily,
    UnknownArchitectureError,
    detect_model_family,
)


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("llama", ModelFamily.LLAMA),
        ("Mistral", ModelFamily.MISTRAL),
        ("qwen2", ModelFamily.QWEN),
        ("qwen2_moe", ModelFamily.QWEN),
        ("qwen3-moe", ModelFamily.QWEN),
        ("gemma", ModelFamily.GEMMA),
        ("gemma2", ModelFamily.GEMMA),
        ("gemma3_text", ModelFamily.GEMMA),
    ],
)
def test_model_type_aliases_are_deterministic(
    model_type: str,
    expected: ModelFamily,
) -> None:
    selection = detect_model_family(ArchitectureEvidence(model_type=model_type))

    assert selection.family is expected


@pytest.mark.parametrize(
    ("architecture", "expected"),
    [
        ("LlamaForCausalLM", ModelFamily.LLAMA),
        ("MistralForCausalLM", ModelFamily.MISTRAL),
        ("Qwen2ForCausalLM", ModelFamily.QWEN),
        ("Qwen3MoeForCausalLM", ModelFamily.QWEN),
        ("Gemma2ForCausalLM", ModelFamily.GEMMA),
        ("Gemma3ForConditionalGeneration", ModelFamily.GEMMA),
    ],
)
def test_hf_architecture_aliases_are_deterministic(
    architecture: str,
    expected: ModelFamily,
) -> None:
    selection = detect_model_family(
        ArchitectureEvidence(architecture_names=(architecture,))
    )

    assert selection.family is expected


@pytest.mark.parametrize("architecture", ["llama", "mistral", "qwen2", "qwen3", "gemma3"])
def test_gguf_architecture_aliases_are_supported(architecture: str) -> None:
    selection = detect_model_family(ArchitectureEvidence(gguf_architecture=architecture))

    assert selection.matched_evidence == (f"gguf:{architecture}",)


def test_agreeing_evidence_is_retained_as_provenance() -> None:
    selection = detect_model_family(
        ArchitectureEvidence(
            model_type="qwen2",
            architecture_names=("Qwen2ForCausalLM",),
            gguf_architecture="qwen2",
        )
    )

    assert selection.family is ModelFamily.QWEN
    assert len(selection.matched_evidence) == 3
    assert json.loads(json.dumps(selection.to_record()))["family"] == "qwen"


def test_conflicting_explicit_evidence_fails_closed() -> None:
    with pytest.raises(ConflictingArchitectureError, match="conflicts"):
        detect_model_family(
            ArchitectureEvidence(model_type="llama", gguf_architecture="mistral")
        )


@pytest.mark.parametrize(
    "evidence",
    [
        ArchitectureEvidence(),
        ArchitectureEvidence(model_type="gpt_neox"),
        ArchitectureEvidence(architecture_names=("UnknownForCausalLM",)),
        ArchitectureEvidence(gguf_architecture="future_model"),
    ],
)
def test_unknown_architectures_fail_instead_of_being_guessed(
    evidence: ArchitectureEvidence,
) -> None:
    with pytest.raises(UnknownArchitectureError, match="no supported architecture alias"):
        detect_model_family(evidence)

