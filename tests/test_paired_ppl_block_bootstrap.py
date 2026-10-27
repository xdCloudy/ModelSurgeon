from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "paired_ppl_block_bootstrap.py"


def _write_ppl_log(path: Path, chunk_nll: list[float]) -> None:
    cumulative = 0.0
    rows = []
    for index, nll in enumerate(chunk_nll):
        cumulative += nll
        mean_nll = cumulative / (index + 1)
        rows.append(f"{index * 512} {math.exp(mean_nll):.12f} {mean_nll:.12f} 0.0")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_named_candidate_bootstrap_compares_matched_logs(tmp_path: Path) -> None:
    candidate = tmp_path / "ensemble.txt"
    baseline = tmp_path / "single.txt"
    _write_ppl_log(candidate, [1.0, 1.1, 0.9, 1.2])
    _write_ppl_log(baseline, [1.2, 1.3, 1.1, 1.4])
    output = tmp_path / "comparison.json"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--candidate",
            f"reap_ensemble={candidate}",
            "--baseline",
            f"reap_single={baseline}",
            "--bootstrap-samples",
            "100",
            "--block-chunks",
            "1",
            "--seed",
            "42",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    comparison = report["comparisons"][0]
    assert comparison["candidate"] == "reap_ensemble"
    assert comparison["baseline"] == "reap_single"
    assert comparison["observed_delta_ppl"] < 0
    assert comparison["moving_block_percentile_95_ci_delta_ppl"][1] < 0
    assert report["inputs"]["reap_ensemble"]["tokens"] == 4 * 512


def test_legacy_four_method_cli_remains_supported(tmp_path: Path) -> None:
    logs = {}
    for name, nll in (
        ("reap", [1.0, 1.1, 0.9, 1.2]),
        ("router", [1.2, 1.3, 1.1, 1.4]),
        ("random_a", [1.3, 1.4, 1.2, 1.5]),
        ("random_b", [1.4, 1.5, 1.3, 1.6]),
    ):
        logs[name] = tmp_path / f"{name}.txt"
        _write_ppl_log(logs[name], nll)
    output = tmp_path / "legacy.json"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reap",
            str(logs["reap"]),
            "--router-norm",
            str(logs["router"]),
            "--random-20261026",
            str(logs["random_a"]),
            "--random-20261027",
            str(logs["random_b"]),
            "--bootstrap-samples",
            "100",
            "--block-chunks",
            "1",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert [item["baseline"] for item in report["comparisons"]] == [
        "router_norm",
        "random_20261026",
        "random_20261027",
    ]
