"""Optional-dependency smoke test for the real Hugging Face/PyTorch proof runtime."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from modelsurgeon.adapters.huggingface.loader import HuggingFaceDType
from modelsurgeon.adapters.huggingface.proof_runtime import (
    HuggingFaceMLPProofConfig,
    HuggingFaceMLPProofRuntime,
)
from modelsurgeon.cli.proof import FirstSurgeonProofConfig, run_first_surgeon_proof
from modelsurgeon.datasets.grouped_splits import SplitPartition, SplitRatios
from modelsurgeon.experiments.candidates import CandidateScope


def _write_tiny_llama(path: Path) -> None:
    config = transformers.LlamaConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = transformers.LlamaForCausalLM(config)
    model.save_pretrained(path, safe_serialization=True)

    vocabulary = {
        "[UNK]": 0,
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
        "e": 5,
        "f": 6,
        "g": 7,
    }
    tokenizer_backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocabulary, unk_token="[UNK]")
    )
    tokenizer_backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(path)


def test_real_hf_runtime_builds_leakage_safe_mlp_channel_dataset(tmp_path: Path) -> None:
    model_path = tmp_path / "tiny-llama"
    model_path.mkdir()
    _write_tiny_llama(model_path)
    calibration = tmp_path / "calibration.txt"
    calibration.write_text(
        "a b c d e f g a b c d e f g a b c d e f g a b c d e f g",
        encoding="utf-8",
    )

    runtime = HuggingFaceMLPProofRuntime(
        HuggingFaceMLPProofConfig(
            model=str(model_path),
            calibration_text=calibration,
            device_map="cpu",
            dtype=HuggingFaceDType.FLOAT32,
            local_files_only=True,
            sequence_length=8,
            max_tokens=16,
            safe_perplexity_delta=1_000_000.0,
            seed=7,
            tool_revision="test-revision",
        )
    )
    result = run_first_surgeon_proof(
        runtime,
        FirstSurgeonProofConfig(
            seed=7,
            split_seed=11,
            max_candidates=12,
            scopes=(CandidateScope.MLP_CHANNEL,),
            ratios=SplitRatios(0.5, 0.25, 0.25),
        ),
    )

    assert len(result.examples) == 12
    assert result.leakage.clean
    assert all(result.split.example_counts[partition] > 0 for partition in SplitPartition)
    first = result.examples[0]
    features = {feature.name: feature for feature in first.pre_mutation_features}
    for name in (
        "weight_count",
        "weight_l1_norm",
        "weight_l2_norm",
        "weight_max_magnitude",
        "activation_rms",
    ):
        assert name in features
    assert features["activation_rms"].sample_context is not None
    assert first.baseline_metrics
    assert first.post_metrics
    assert first.delta_metrics
