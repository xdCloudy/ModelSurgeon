#!/usr/bin/env python3
"""Paired moving-block bootstrap for llama.cpp --ppl-output-type 1 logs.

Each row reports the cumulative mean negative log-likelihood through the
512-token chunk whose starting offset is in column one. Recover per-chunk
mean NLLs, pair chunks by their common offset, then resample contiguous blocks
of paired chunks. This estimates within-corpus uncertainty for fixed artifacts;
it does not estimate variation across models, calibration samples, or datasets.
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
        (index + 1) * cumulative_nll[index]
        - index * cumulative_nll[index - 1]
        for index in range(1, len(cumulative_nll))
    )
    total_ppl = math.exp(cumulative_nll[-1])
    recovered_ppl = math.exp(sum(chunk_nll) / len(chunk_nll))
    if not math.isclose(total_ppl, recovered_ppl, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError(f"cumulative/chunk reconstruction mismatch for {path}")
    return offsets, chunk_nll, total_ppl, hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reap", type=Path, required=True)
    parser.add_argument("--router-norm", type=Path, required=True)
    parser.add_argument("--random-20261026", type=Path, required=True)
    parser.add_argument("--random-20261027", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--block-chunks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_261_027)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.bootstrap_samples < 1 or args.block_chunks < 1:
        parser.error("bootstrap-samples and block-chunks must be positive")

    inputs = {
        "reap": args.reap,
        "router_norm": args.router_norm,
        "random_20261026": args.random_20261026,
        "random_20261027": args.random_20261027,
    }
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
        parser.error("all four PPL logs must contain the same number of chunks")
    if any(offsets != offsets_by_method["reap"] for offsets in offsets_by_method.values()):
        parser.error("all four PPL logs must contain the same chunk offsets")

    count = lengths.pop()
    rng = random.Random(args.seed)
    comparisons = []
    ppl_by_method = {
        method: float(values["perplexity_recovered"])
        for method, values in metrics.items()
    }
    for baseline in ("router_norm", "random_20261026", "random_20261027"):
        observed = ppl_by_method["reap"] - ppl_by_method[baseline]
        samples: list[float] = []
        for _ in range(args.bootstrap_samples):
            indices: list[int] = []
            while len(indices) < count:
                start = rng.randrange(count)
                indices.extend(
                    (start + offset) % count for offset in range(args.block_chunks)
                )
            indices = indices[:count]
            reap_nll = sum(losses["reap"][index] for index in indices) / count
            baseline_nll = sum(losses[baseline][index] for index in indices) / count
            samples.append(math.exp(reap_nll) - math.exp(baseline_nll))
        samples.sort()
        comparisons.append(
            {
                "candidate": "reap",
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
