"""Download pinned small HF fixtures and convert them with a supplied llama.cpp tree."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download


@dataclass(frozen=True, slots=True)
class FixtureSource:
    name: str
    identifier: str
    revision: str


_SOURCES = (
    FixtureSource(
        "llama",
        "HuggingFaceTB/SmolLM2-135M",
        "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    ),
    FixtureSource(
        "mistral",
        "HuggingFaceM4/tiny-random-MistralForCausalLM",
        "687d960c4d20b757867bb3284cf1a55d88e6c348",
    ),
    FixtureSource(
        "qwen",
        "Jiqing/tiny-random-qwen2",
        "969ee18c66962024684058e7859adc0a4420c9eb",
    ),
    FixtureSource(
        "gemma",
        "tiny-random/gemma-2",
        "ebd61568c424fcc6a4e42a69b2aba7c64447ffc9",
    ),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cpp", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quantize", required=True, type=Path)
    return parser.parse_args()


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, shell=False)


def main() -> None:
    arguments = _arguments()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    converter = arguments.llama_cpp.resolve() / "convert_hf_to_gguf.py"
    if not converter.is_file():
        raise FileNotFoundError(f"llama.cpp converter does not exist: {converter}")

    for source in _SOURCES:
        model_dir = output / f"hf-{source.name}"
        snapshot_download(
            repo_id=source.identifier,
            revision=source.revision,
            local_dir=model_dir,
        )
        destination = output / f"{source.name}-f16.gguf"
        _run(
            [
                sys.executable,
                str(converter),
                str(model_dir),
                "--outfile",
                str(destination),
                "--outtype",
                "f16",
            ]
        )

    _run(
        [
            str(arguments.quantize.resolve()),
            str(output / "llama-f16.gguf"),
            str(output / "llama-q4_k_m.gguf"),
            "Q4_K_M",
        ]
    )


if __name__ == "__main__":
    main()
