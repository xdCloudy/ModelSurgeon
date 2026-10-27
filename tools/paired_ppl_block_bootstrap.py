#!/usr/bin/env python3
"""Paired moving-block bootstrap for llama.cpp --ppl-output-type 1 logs.

Each row reports the cumulative mean negative log-likelihood through the
512-token chunk whose starting offset is in column one. Recover per-chunk
mean NLLs, pair chunks by their common offset, then resample contiguous blocks
of paired chunks. This estimates within-corpus uncertainty for fixed artifacts;
it does not estimate variation across models, calibration samples, or datasets.
The default CLI retains the original REAP/router/random comparison. Named
inputs permit other matched candidate/baseline comparisons without changing
the bootstrap implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
import sys
from pathlib import Path

_ROW = re.compile(r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$")


def _read(path: Path) -> tuple[list[int], list[float], float, str]:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    rows: list[tuple[int, float, float, float]] = []
    for line in text.splitlines():
        match = _ROW.match(line)
        if match is not None:
            rows.append(
                (
                    int(match.group(1)),
                    float(match.group(2)),
                    float(match.group(3)),
                    float(match.group(4)),
                )
            )
    if not rows:
        raise ValueError(f"no llama.cpp PPL rows found in {path}")
    offsets = [row[0] for row in rows]
    if any(right - left != 512 for left, right in itertools.pairwise(offsets)):
        raise ValueError(f"expected 512-token row offsets in {path}")
    cumulative_nll = [row[2] for row in rows]
    chunk_nll = [cumulative_nll[0]]
    chunk_nll.extend(
        (index + 1) * cumulative_nll[index] - index * cumulative_nll[index - 1]
        for index in range(1, len(cumulative_nll))
    )
    total_ppl = math.exp(cumulative_nll[-1])
    recovered_ppl = math.exp(sum(chunk_nll) / len(chunk_nll))
    if not math.isclose(total_ppl, recovered_ppl, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError(f"cumulative/chunk reconstruction mismatch for {path}")
    return offsets, chunk_nll, total_ppl, hashlib.sha256(raw).hexdigest()


def _named_input(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path or any(char.isspace() for char in name):
        raise argparse.ArgumentTypeError("expected NAME=PATH with a non-empty name and path")
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reap", type=Path)
    parser.add_argument("--router-norm", type=Path)
    parser.add_argument("--random-20261026", type=Path)
    parser.add_argument("--random-20261027", type=Path)
    parser.add_argument("--candidate", type=_named_input, help="candidate NAME=PATH")
    parser.add_argument(
        "--baseline",
        type=_named_input,
        action="append",
        default=[],
        help="baseline NAME=PATH (repeatable)",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--block-chunks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_261_027)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.bootstrap_samples < 1 or args.block_chunks < 1:
        parser.error("bootstrap-samples and block-chunks must be positive")

    legacy = {
        "reap": args.reap,
        "router_norm": args.router_norm,
        "random_20261026": args.random_20261026,
        "random_20261027": args.random_20261027,
    }
    if args.candidate is not None:
        if any(path is not None for path in legacy.values()):
            parser.error("named candidate/baseline inputs cannot be mixed with legacy inputs")
        if not args.baseline:
            parser.error("named candidate input requires at least one --baseline NAME=PATH")
        candidate, candidate_path = args.candidate
        inputs = {candidate: candidate_path}
        for baseline, path in args.baseline:
            if baseline in inputs:
                parser.error(f"duplicate PPL input name: {baseline}")
            inputs[baseline] = path
    else:
        if args.baseline:
            parser.error("--baseline requires --candidate NAME=PATH")
        if any(path is None for path in legacy.values()):
            parser.error(
                "provide all four legacy inputs or use --candidate NAME=PATH and "
                "one or more --baseline NAME=PATH options"
            )
        candidate = "reap"
        inputs = {name: path for name, path in legacy.items() if path is not None}
    offsets_by_method: dict[str, list[int]] = {}
    losses: dict[str, list[float]] = {}
    metrics: dict[str, dict[str, float | str | int]] = {}
    for method, path in inputs.items():
        offsets, chunk_nll, ppl, digest = _read(path)
        offsets_by_method[method] = offsets
        losses[method] = chunk_nll
        metrics[method] = {
            "rows": len(chunk_nll),
            "tokens": len(chunk_nll) * 512,
            "perplexity_recovered": ppl,
            "stdout_sha256": digest,
        }
    lengths = {len(rows) for rows in losses.values()}
    if len(lengths) != 1:
        parser.error("all PPL logs must contain the same number of chunks")
    if any(offsets != offsets_by_method[candidate] for offsets in offsets_by_method.values()):
        parser.error("all PPL logs must contain the same chunk offsets")

    count = lengths.pop()
    rng = random.Random(args.seed)
    comparisons = []
    ppl_by_method = {
        method: float(values["perplexity_recovered"]) for method, values in metrics.items()
    }
    for baseline in (name for name in inputs if name != candidate):
        observed = ppl_by_method[candidate] - ppl_by_method[baseline]
        samples: list[float] = []
        for _ in range(args.bootstrap_samples):
            indices: list[int] = []
            while len(indices) < count:
                start = rng.randrange(count)
                indices.extend((start + offset) % count for offset in range(args.block_chunks))
            indices = indices[:count]
            candidate_nll = sum(losses[candidate][index] for index in indices) / count
            baseline_nll = sum(losses[baseline][index] for index in indices) / count
            samples.append(math.exp(candidate_nll) - math.exp(baseline_nll))
        samples.sort()
        comparisons.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "observed_delta_ppl": observed,
                "moving_block_percentile_95_ci_delta_ppl": [
                    samples[int(0.025 * args.bootstrap_samples)],
                    samples[min(args.bootstrap_samples - 1, int(0.975 * args.bootstrap_samples))],
                ],
            }
        )

    report = {
        "method": "paired circular moving-block bootstrap of per-chunk mean NLL",
        "scope": (
            "conditional uncertainty within this single ordered held-out corpus; "
            "not across models, samples, or datasets"
        ),
        "chunk_size_tokens": 512,
        "block_chunks": args.block_chunks,
        "block_size_tokens": args.block_chunks * 512,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "inputs": metrics,
        "comparisons": comparisons,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(serialized)
    else:
        args.output.write_text(serialized, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
