"""Optional-dependency smoke test for the real Hugging Face/PyTorch proof runtime."""

from __future__ import annotations

import weakref
from pathlib import Path

import pytest

from modelsurgeon.adapters.huggingface.loader import HuggingFaceDType
from modelsurgeon.adapters.huggingface.proof_runtime import (
    HuggingFaceMLPProofConfig,
    HuggingFaceMLPProofRuntime,
)
from modelsurgeon.cli.proof import FirstSurgeonProofConfig, run_first_surgeon_proof
from modelsurgeon.datasets.grouped_splits import SplitPartition, SplitRatios
from modelsurgeon.experiments.candidates import CandidateScope
from modelsurgeon.graph import ComponentId

torch = pytest.importorskip("torch")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")


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
            max_tokens=18,
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

    gradient_components = tuple(
        ComponentId.parse(f"model.layers.0.mlp.channel.{channel}") for channel in (0, 1)
    )
    live_logits: list[weakref.ReferenceType[object]] = []

    def verify_previous_graph_released(_module: object, _inputs: object, output: object) -> None:
        if live_logits:
            assert live_logits[-1]() is None
        logits = output.logits
        live_logits.append(weakref.ref(logits))

    handle = runtime.model.register_forward_hook(verify_previous_graph_released)
    try:
        gradients = runtime.collect_mlp_channel_gradient_features(
            tuple((component, 0, channel) for channel, component in enumerate(gradient_components))
        )
    finally:
        handle.remove()
    assert len(live_logits) == len(runtime._chunks)
    assert live_logits[-1]() is None
    assert gradients.token_count == 15
    assert tuple(component for component, _ in gradients.records) == gradient_components
    assert all(
        {feature.name for feature in features}
        >= {"gradient_l2_norm", "first_order_removal_estimate"}
        for _, features in gradients.records
    )
    assert all(
        feature.sample_context is not None
        for _, features in gradients.records
        for feature in features
    )

    # Compare the reported first-order score with the token-weighted gradient
    # of the same independently chunked causal-LM corpus. The final chunk has
    # two input tokens (one target), while the first two chunks have seven each.
    runtime.model.zero_grad(set_to_none=True)
    summed_nll = None
    token_count = 0
    for chunk in runtime._chunks:
        input_ids = torch.tensor(
            [chunk], dtype=torch.long, device=runtime._input_device
        )
        logits = runtime.model(input_ids=input_ids, use_cache=False).logits[:, :-1, :]
        targets = input_ids[:, 1:].to(logits.device)
        nll = torch.nn.functional.cross_entropy(
            logits.float().contiguous().view(-1, int(logits.shape[-1])),
            targets.contiguous().view(-1),
            reduction="sum",
        )
        summed_nll = nll if summed_nll is None else summed_nll + nll
        token_count += int(targets.numel())
    assert summed_nll is not None
    (summed_nll / token_count).backward()
    gate, up, down = runtime._projection_weights(0)
    expected_removal = -(
        torch.sum(gate[0].detach().float() * gate.grad[0].detach().float())
        + torch.sum(up[0].detach().float() * up.grad[0].detach().float())
        + torch.sum(
            down.detach().float().transpose(0, 1)[0]
            * down.grad.detach().float().transpose(0, 1)[0]
        )
    )
    reported = dict(gradients.records)[gradient_components[0]]
    first_order = next(
        feature.value
        for feature in reported
        if feature.name == "first_order_removal_estimate"
    )
    assert first_order == pytest.approx(float(expected_removal), rel=1e-5, abs=1e-7)
    runtime.model.zero_grad(set_to_none=True)
