"""Turnkey Hugging Face command for the v0.5 First Surgeon empirical proof."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from modelsurgeon.adapters.huggingface.loader import HuggingFaceDependencyError, HuggingFaceDType
from modelsurgeon.adapters.huggingface.proof_runtime import (
    HuggingFaceMLPProofConfig,
    HuggingFaceMLPProofError,
    HuggingFaceMLPProofRuntime,
)
from modelsurgeon.cli.proof import (
    FirstSurgeonProofConfig,
    run_first_surgeon_proof,
    write_first_surgeon_proof,
)
from modelsurgeon.datasets.grouped_splits import SplitPartition, SplitRatios
from modelsurgeon.experiments.candidates import CandidateScope
from modelsurgeon.experiments.identity import canonical_identity_json


def first_surgeon_hf_proof_command(
    model: Annotated[str, typer.Argument(help="Hugging Face causal-LM model ID or local path")],
    calibration_text: Annotated[
        Path,
        typer.Argument(help="UTF-8 calibration text used for activation capture and perplexity"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="New directory for proof dataset and audit artifacts"),
    ],
    revision: Annotated[str | None, typer.Option(help="Immutable model revision")] = None,
    tokenizer: Annotated[
        str | None,
        typer.Option(help="Optional tokenizer ID/path; defaults to the model source"),
    ] = None,
    tokenizer_revision: Annotated[
        str | None,
        typer.Option(help="Immutable tokenizer revision when different from the model"),
    ] = None,
    device_map: Annotated[
        str,
        typer.Option(help="Transformers placement strategy"),
    ] = "auto",
    dtype: Annotated[
        HuggingFaceDType,
        typer.Option(help="Requested model compute dtype"),
    ] = HuggingFaceDType.AUTO,
    trust_remote_code: Annotated[
        bool,
        typer.Option(help="Allow model repository code; disabled by default"),
    ] = False,
    local_files_only: Annotated[
        bool,
        typer.Option(help="Forbid Hub downloads and use only local cache/files"),
    ] = False,
    sequence_length: Annotated[
        int,
        typer.Option("--sequence-length", min=2, help="Tokens per evaluation chunk"),
    ] = 256,
    max_tokens: Annotated[
        int,
        typer.Option("--max-tokens", min=2, help="Maximum calibration tokens per candidate"),
    ] = 4096,
    safe_perplexity_delta: Annotated[
        float,
        typer.Option(
            "--safe-perplexity-delta",
            min=0.0,
            help="Maximum allowed post-minus-baseline perplexity for a safe label",
        ),
    ] = 0.25,
    max_candidates: Annotated[
        int,
        typer.Option("--max-candidates", min=3, help="Seeded MLP-channel sample size"),
    ] = 5000,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 42,
    split_seed: Annotated[int, typer.Option("--split-seed", min=0)] = 43,
    train_ratio: Annotated[float, typer.Option("--train-ratio", min=0.01, max=0.98)] = 0.8,
    validation_ratio: Annotated[
        float,
        typer.Option("--validation-ratio", min=0.01, max=0.98),
    ] = 0.1,
    test_ratio: Annotated[float, typer.Option("--test-ratio", min=0.01, max=0.98)] = 0.1,
    tool_revision: Annotated[
        str | None,
        typer.Option(help="Exact ModelSurgeon commit/revision for provenance"),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the completed campaign record as JSON"),
    ] = False,
) -> None:
    """Run real MLP-channel masks and build the leakage-safe First Surgeon proof dataset."""

    try:
        runtime = HuggingFaceMLPProofRuntime(
            HuggingFaceMLPProofConfig(
                model=model,
                calibration_text=calibration_text,
                revision=revision,
                tokenizer=tokenizer,
                tokenizer_revision=tokenizer_revision,
                device_map=device_map,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                sequence_length=sequence_length,
                max_tokens=max_tokens,
                safe_perplexity_delta=safe_perplexity_delta,
                seed=seed,
                tool_revision=tool_revision,
            )
        )
        proof = run_first_surgeon_proof(
            runtime,
            FirstSurgeonProofConfig(
                seed=seed,
                split_seed=split_seed,
                max_candidates=max_candidates,
                scopes=(CandidateScope.MLP_CHANNEL,),
                ratios=SplitRatios(train_ratio, validation_ratio, test_ratio),
            ),
        )
        paths = write_first_surgeon_proof(output, proof)
        if output_json:
            typer.echo(canonical_identity_json(proof.to_record()))
            return
        counts = proof.split.example_counts
        typer.echo(
            "Hugging Face First Surgeon dataset ready: "
            f"examples={len(proof.examples)} "
            f"train={counts[SplitPartition.TRAIN]} "
            f"validation={counts[SplitPartition.VALIDATION]} "
            f"test={counts[SplitPartition.TEST]}"
        )
        typer.echo(f"examples: {paths['examples']}")
        typer.echo(f"split: {paths['split']}")
        typer.echo(f"leakage audit: {paths['leakage']}")
        typer.echo(
            "regressor: modelsurgeon train-surgeon "
            f"{paths['examples']} --split {paths['split']} "
            f"--registry {output / 'registry-regression'} --target perplexity "
            "--baseline lightgbm-regressor --seed 42"
        )
        typer.echo(
            "classifier: modelsurgeon train-surgeon "
            f"{paths['examples']} --split {paths['split']} "
            f"--registry {output / 'registry-classification'} --target safe_mutation "
            f"--safe-threshold perplexity={safe_perplexity_delta} "
            "--baseline lightgbm-classifier --seed 42 --top-n 50"
        )
    except KeyboardInterrupt:
        typer.echo("Hugging Face proof interrupted; active mutation hook removed", err=True)
        raise typer.Exit(130) from None
    except (HuggingFaceDependencyError, HuggingFaceMLPProofError, OSError, RuntimeError, ValueError) as error:
        typer.echo(f"first-surgeon-hf-proof error: {error}", err=True)
        raise typer.Exit(2) from error
