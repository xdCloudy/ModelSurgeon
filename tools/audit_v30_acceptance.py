"""Run and audit the bounded v3.0 end-to-end acceptance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


class V30AcceptanceError(ValueError):
    """Raised when the v3.0 acceptance record is incomplete or overclaims."""


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "docs" / "research" / "v3.0-acceptance-matrix-v1.json"
EVIDENCE_PATH = ROOT / "docs" / "research" / "v3.0-acceptance-evidence-v1.json"
OUTCOME_VOCABULARY = {
    "supported",
    "unsupported",
    "failed",
    "unknown",
    "inconclusive",
    "known_skip",
}


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise V30AcceptanceError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise V30AcceptanceError(f"{label} must be non-empty text")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise V30AcceptanceError(f"{label} must be an array")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), label)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V30AcceptanceError(f"could not read {label}: {path}") from error


def _relative_file(root: Path, value: object, label: str) -> str:
    raw = _text(value, label)
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or not (root / path).is_file():
        raise V30AcceptanceError(f"{label} references a missing repository file")
    return raw


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command_text(command: list[str]) -> str:
    return " ".join(f'"{item}"' if " " in item else item for item in command)


def _run(command: list[str], root: Path, *, timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        output = (result.stdout + result.stderr).strip()
        return {
            "command": _command_text(command),
            "exit_code": result.returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "outcome": "supported" if result.returncode == 0 else "failed",
            "output_tail": output[-4000:],
            "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        }
    except subprocess.TimeoutExpired as error:
        output = str(error.stdout or "") + str(error.stderr or "")
        return {
            "command": _command_text(command),
            "exit_code": None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "outcome": "failed",
            "failure": "timeout",
            "output_tail": output[-4000:],
            "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        }
    except OSError as error:
        return {
            "command": _command_text(command),
            "exit_code": None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "outcome": "failed",
            "failure": f"could not start command: {error}",
            "output_tail": "",
            "output_sha256": hashlib.sha256(b"").hexdigest(),
        }


def _uv(*args: str) -> list[str]:
    return ["uv", "run", "--locked", "--extra", "dev", *args]


def _validate_matrix(
    root: Path, matrix: dict[str, Any], *, require_evidence: bool = True
) -> dict[str, dict[str, Any]]:
    if matrix.get("record_type") != "v30_acceptance_matrix":
        raise V30AcceptanceError("unexpected v3.0 matrix record type")
    if matrix.get("schema_version") != 1:
        raise V30AcceptanceError("unsupported v3.0 matrix schema")
    if matrix.get("protocol_revision") != "modelsurgeon-v3.0-acceptance-matrix-v1":
        raise V30AcceptanceError("v3.0 matrix protocol revision drifted")
    if matrix.get("issue") != 490 or matrix.get("status") != "bounded_gate":
        raise V30AcceptanceError("v3.0 matrix issue or status drifted")
    boundary = _object(matrix.get("authority_boundary"), "authority_boundary")
    if boundary.get("ux_completion_is_optimizer_correctness") is not False:
        raise V30AcceptanceError("UX/optimizer authority boundary was weakened")
    if boundary.get("provider_text_is_authoritative") is not False:
        raise V30AcceptanceError("provider authority boundary was weakened")
    if boundary.get("optimizer_quality_is_claimed_by_this_gate") is not False:
        raise V30AcceptanceError("optimizer quality was overclaimed")
    vocabulary = set(_text(item, "outcome_vocabulary item") for item in _array(
        matrix.get("outcome_vocabulary"), "outcome_vocabulary"
    ))
    if vocabulary != OUTCOME_VOCABULARY:
        raise V30AcceptanceError("outcome vocabulary is incomplete or drifted")

    cells: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(_array(matrix.get("cells"), "cells")):
        cell = _object(raw, f"cells[{index}]")
        cell_id = _text(cell.get("id"), f"cells[{index}].id")
        if cell_id in cells:
            raise V30AcceptanceError(f"duplicate acceptance cell: {cell_id}")
        expected = _text(cell.get("expected_outcome"), f"cells[{index}].expected_outcome")
        if expected not in OUTCOME_VOCABULARY:
            raise V30AcceptanceError(f"invalid expected outcome for {cell_id}")
        for test_index, test in enumerate(_array(cell.get("tests"), f"cells[{index}].tests")):
            _relative_file(root, test, f"cells[{index}].tests[{test_index}]")
        _text(cell.get("surface"), f"cells[{index}].surface")
        _text(cell.get("evidence_class"), f"cells[{index}].evidence_class")
        _text(cell.get("notes"), f"cells[{index}].notes")
        cells[cell_id] = cell
    required = {_text(item, "required_evidence item") for item in _array(
        matrix.get("required_evidence"), "required_evidence"
    )}
    if require_evidence:
        for path in required:
            _relative_file(root, path, "required_evidence")
    return cells


def _validate_evidence(root: Path, matrix: dict[str, Any], evidence: dict[str, Any]) -> None:
    if evidence.get("record_type") != "v30_acceptance_evidence":
        raise V30AcceptanceError("unexpected v3.0 evidence record type")
    if evidence.get("schema_version") != 1:
        raise V30AcceptanceError("unsupported v3.0 evidence schema")
    if evidence.get("protocol_revision") != "modelsurgeon-v3.0-acceptance-evidence-v1":
        raise V30AcceptanceError("v3.0 evidence protocol revision drifted")
    if evidence.get("issue") != 490:
        raise V30AcceptanceError("v3.0 evidence issue drifted")
    if evidence.get("status") not in {"passed_with_explicit_skips", "failed"}:
        raise V30AcceptanceError("v3.0 evidence status is not explicit")
    _text(evidence.get("tested_revision"), "tested_revision")
    _text(evidence.get("branch"), "branch")
    environment = _object(evidence.get("environment"), "environment")
    for key in ("os", "platform", "python", "uv", "dependency_setup"):
        _text(environment.get(key), f"environment.{key}")
    budget = _object(evidence.get("budgets"), "budgets")
    for key in ("command_timeout_seconds", "max_output_tail_chars"):
        if not isinstance(budget.get(key), int) or budget[key] <= 0:
            raise V30AcceptanceError(f"budgets.{key} must be a positive integer")

    matrix_cells = {
        _text(cell.get("id"), "matrix cell id")
        for cell in _array(matrix.get("cells"), "matrix.cells")
    }
    seen: set[str] = set()
    for index, raw in enumerate(_array(evidence.get("cells"), "evidence.cells")):
        cell = _object(raw, f"evidence.cells[{index}]")
        cell_id = _text(cell.get("id"), f"evidence.cells[{index}].id")
        if cell_id in seen or cell_id not in matrix_cells:
            raise V30AcceptanceError(f"unexpected or duplicate evidence cell: {cell_id}")
        seen.add(cell_id)
        outcome = _text(cell.get("observed_outcome"), f"evidence.cells[{index}].observed_outcome")
        if outcome not in OUTCOME_VOCABULARY:
            raise V30AcceptanceError(f"invalid observed outcome for {cell_id}")
        evidence_value = cell.get("evidence")
        if not isinstance(evidence_value, (str, dict)) or not evidence_value:
            raise V30AcceptanceError(
                f"evidence.cells[{index}].evidence must be non-empty text or object"
            )
        if outcome in {"failed", "unknown", "unsupported", "inconclusive", "known_skip"}:
            _text(cell.get("limitation"), f"evidence.cells[{index}].limitation")
    if seen != matrix_cells:
        raise V30AcceptanceError("acceptance evidence does not cover every matrix cell")

    claims = _object(evidence.get("claims"), "claims")
    for key in ("supported", "unsupported", "not_claimed"):
        values = _array(claims.get(key), f"claims.{key}")
        if not values or any(not isinstance(item, str) or not item.strip() for item in values):
            raise V30AcceptanceError(f"claims.{key} must retain explicit text")
    if not any("optimizer" in item.lower() for item in claims["not_claimed"]):
        raise V30AcceptanceError("optimizer limitation is not explicit")
    failures = _array(evidence.get("failures"), "failures")
    for index, failure in enumerate(failures):
        item = _object(failure, f"failures[{index}]")
        _text(item.get("cell"), f"failures[{index}].cell")
        _text(item.get("summary"), f"failures[{index}].summary")
    known_skips = _array(evidence.get("known_skips"), "known_skips")
    if not known_skips or any(
        not isinstance(item, str) or not item.strip() for item in known_skips
    ):
        raise V30AcceptanceError("known_skips must retain explicit text")
    quality = _object(evidence.get("quality_gate"), "quality_gate")
    if quality.get("status") not in {"passed", "not_run", "failed"}:
        raise V30AcceptanceError("quality_gate status is invalid")
    commands = _array(quality.get("commands"), "quality_gate.commands")
    if not commands:
        raise V30AcceptanceError("quality_gate.commands must not be empty")
    for index, command in enumerate(commands):
        item = _object(command, f"quality_gate.commands[{index}]")
        _text(item.get("command"), f"quality_gate.commands[{index}].command")
        _text(item.get("outcome"), f"quality_gate.commands[{index}].outcome")
    artifact_paths = _array(evidence.get("artifacts"), "artifacts")
    for index, raw in enumerate(artifact_paths):
        item = _object(raw, f"artifacts[{index}]")
        _relative_file(root, item.get("path"), f"artifacts[{index}].path")
        _text(item.get("role"), f"artifacts[{index}].role")


def audit_release(
    root: Path = ROOT,
    *,
    matrix: Path = MATRIX_PATH,
    evidence: Path = EVIDENCE_PATH,
) -> None:
    """Validate the checked-in v3.0 matrix and its bounded evidence record."""

    root = root.resolve()
    matrix_record = _read_json(matrix, "v3.0 acceptance matrix")
    _validate_matrix(root, matrix_record)
    evidence_record = _read_json(evidence, "v3.0 acceptance evidence")
    _validate_evidence(root, matrix_record, evidence_record)
    if evidence_record.get("status") == "passed_with_explicit_skips":
        quality = _object(evidence_record.get("quality_gate"), "quality_gate")
        if quality.get("status") != "passed":
            raise V30AcceptanceError("passed evidence must include a passed quality gate")


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _environment(root: Path) -> dict[str, str]:
    uv = subprocess.run(["uv", "--version"], capture_output=True, text=True, check=True)
    return {
        "os": platform.platform(),
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "uv": uv.stdout.strip(),
        "dependency_setup": "uv sync --locked --extra dev",
    }


def _test_cells(root: Path, cells: dict[str, dict[str, Any]], timeout: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for cell_id, cell in cells.items():
        tests = [str(item) for item in cell["tests"]]
        if not tests:
            results.append(
                {
                    "id": cell_id,
                    "observed_outcome": cell["expected_outcome"],
                    "evidence": "explicit matrix boundary; no command is claimed",
                    "limitation": cell["notes"],
                }
            )
            continue
        result = _run([*_uv("pytest", *tests, "-q")], root, timeout=timeout)
        observed = "supported" if result["outcome"] == "supported" else "failed"
        results.append(
            {
                "id": cell_id,
                "observed_outcome": observed,
                "evidence": result,
                "limitation": cell["notes"] if observed != "supported" else "",
            }
        )
    return results


def _cli_smokes(root: Path, timeout: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="modelsurgeon-v30-") as temporary:
        temp = Path(temporary)
        setup_command = _uv(
            "modelsurgeon",
            "setup",
            "init",
            "--data-dir",
            str(temp / "setup"),
            "--fixture",
            "tests/fixtures/tiny_hf_models_v1.json",
            "--offline",
            "--min-free-gb",
            "0.000001",
            "--json",
        )
        result = _run(setup_command, root, timeout=timeout)
        results.append(
            {
                "id": "smoke_setup_init",
                "observed_outcome": "supported" if result["outcome"] == "supported" else "failed",
                "evidence": result,
                "limitation": "metadata-only setup; no model weights are downloaded",
            }
        )
        no_llm = _run(
            _uv("modelsurgeon", "provider", "diagnostics", "--no-llm", "--json"),
            root,
            timeout=timeout,
        )
        results.append(
            {
                "id": "smoke_provider_no_llm",
                "observed_outcome": "supported" if no_llm["outcome"] == "supported" else "failed",
                "evidence": no_llm,
                "limitation": "no-LLM is intentionally provider-independent",
            }
        )
        offline = _run(
            _uv(
                "modelsurgeon",
                "provider",
                "diagnostics",
                "--provider",
                "compatible_endpoint",
                "--provider-id",
                "endpoint",
                "--provider-model",
                "model",
                "--provider-revision",
                "revision-1",
                "--provider-endpoint",
                "https://provider.example/v1",
                "--provider-api-key-env",
                "MODELSURGEON_V30_TEST_KEY",
                "--offline",
                "--json",
            ),
            root,
            timeout=timeout,
        )
        results.append(
            {
                "id": "smoke_provider_offline",
                "observed_outcome": (
                    "unsupported" if offline["outcome"] == "supported" else "failed"
                ),
                "evidence": offline,
                "limitation": "offline mode refuses remote provider execution by policy",
            }
        )
        plan = _run(_uv("modelsurgeon", "optimize", "--no-llm", "--json"), root, timeout=timeout)
        results.append(
            {
                "id": "smoke_optimize_no_llm_plan",
                "observed_outcome": "supported" if plan["outcome"] == "supported" else "failed",
                "evidence": plan,
                "limitation": (
                    "a read-only plan with unresolved model inputs is not optimizer evidence"
                ),
            }
        )
    return results


def _quality_gate(root: Path, timeout: int, *, full: bool) -> list[dict[str, Any]]:
    commands = [
        ["git", "diff", "--check"],
        _uv("ruff", "check", "src", "tests", "tools"),
        _uv("mypy", "src/modelsurgeon"),
        _uv("python", "tools/audit_adversarial_resistance.py"),
        _uv("python", "tools/audit_v29_conversational_security_release.py"),
        _uv("python", "tools/audit_migration_compatibility.py"),
    ]
    if full:
        commands.append(_uv("pytest", "-q"))
    return [_run(command, root, timeout=timeout) for command in commands]


def run_acceptance(
    root: Path = ROOT,
    *,
    matrix: Path = MATRIX_PATH,
    output: Path = EVIDENCE_PATH,
    timeout: int = 900,
    full: bool = False,
) -> dict[str, Any]:
    """Execute focused acceptance plus real CLI smokes and write evidence."""

    root = root.resolve()
    matrix_record = _read_json(matrix, "v3.0 acceptance matrix")
    cells = _validate_matrix(root, matrix_record, require_evidence=False)
    tested_revision = _git(root, "rev-parse", "HEAD")
    evidence: dict[str, Any] = {
        "record_type": "v30_acceptance_evidence",
        "schema_version": 1,
        "protocol_revision": "modelsurgeon-v3.0-acceptance-evidence-v1",
        "issue": 490,
        "milestone": "v3.0",
        "status": "failed",
        "tested_revision": tested_revision,
        "branch": _git(root, "branch", "--show-current"),
        "measurement_worktree_clean": not bool(_git(root, "status", "--porcelain")),
        "environment": _environment(root),
        "budgets": {
            "command_timeout_seconds": timeout,
            "max_output_tail_chars": 4000,
            "test_scope": "declared matrix cells plus CLI smokes",
            "optimizer_budget": "not measured; no model-quality claim",
        },
        "fixture": {
            "manifest": "tests/fixtures/tiny_hf_models_v1.json",
            "manifest_sha256": _sha256(root / "tests/fixtures/tiny_hf_models_v1.json"),
            "weights_committed": False,
            "families": ["gemma", "llama", "mistral", "qwen"],
            "upstream_revisions_are_pinned": True,
        },
        "cells": _test_cells(root, cells, timeout),
        "cli_smokes": _cli_smokes(root, timeout),
        "quality_gate": {
            "status": "not_run",
            "commands": [
                {
                    "command": "quality gate pending until the evidence manifest is materialized",
                    "outcome": "known_skip",
                }
            ],
        },
        "claims": {
            "supported": [
                "bounded local contract and direct API acceptance",
                "offline metadata fixture setup with pinned upstream revisions",
                "no-LLM CLI/Python parity and explicit negative outcomes",
                "approval, evidence, provenance, artifact, adversarial, and replay contracts",
            ],
            "unsupported": [
                "live hosted provider execution and quality",
                "external local provider runtime availability",
                "CUDA execution in this environment",
            ],
            "not_claimed": [
                "optimizer correctness, optimality, or model-quality improvement",
                "representative model-family surgery from committed or downloaded weights",
                "hostile operating-system process containment",
                "universal architecture, filesystem, device, or deployment support",
            ],
        },
        "known_skips": [
            (
                "The pinned tiny fixture manifest is metadata-only; target weights are not "
                "committed or downloaded."
            ),
            "The full pytest suite retains environment-dependent skips reported by pytest.",
            (
                "No CUDA device, hosted credential, live endpoint, or installed llama.cpp "
                "runtime was required."
            ),
        ],
    }
    evidence["failures"] = []
    evidence["artifacts"] = [
        {
            "path": "docs/research/v3.0-acceptance-matrix-v1.json",
            "role": "declarative acceptance matrix",
            "sha256": _sha256(matrix),
        },
        {
            "path": "docs/research/v3.0-acceptance-evidence-v1.json",
            "role": "machine-readable execution evidence",
        },
        {"path": "tools/audit_v30_acceptance.py", "role": "executable audit and harness"},
        {
            "path": "docs/release/v3.0-acceptance-security-reproducibility-review.md",
            "role": "final review",
        },
    ]
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    quality_commands = _quality_gate(root, timeout, full=full)
    evidence["quality_gate"]["commands"] = quality_commands
    all_results = evidence["cells"] + evidence["cli_smokes"] + quality_commands
    failures = []
    for item in all_results:
        if item.get("outcome") == "failed" or item.get("observed_outcome") == "failed":
            failures.append(
                {
                    "cell": item.get("id", item.get("command", "unknown")),
                    "summary": item.get("failure") or item.get("output_tail") or "command failed",
                }
            )
    evidence["failures"] = failures
    evidence["quality_gate"]["status"] = "passed" if not any(
        item.get("outcome") == "failed" for item in evidence["quality_gate"]["commands"]
    ) else "failed"
    evidence["status"] = "passed_with_explicit_skips" if not failures else "failed"
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE_PATH)
    parser.add_argument("--run", action="store_true", help="execute the acceptance gate")
    parser.add_argument("--full", action="store_true", help="include the complete pytest suite")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.run:
        record = run_acceptance(
            args.root,
            matrix=args.matrix,
            output=args.evidence,
            timeout=args.timeout,
            full=args.full,
        )
        print(f"v3.0 acceptance evidence written: {args.evidence}")
        return 0 if record["status"] == "passed_with_explicit_skips" else 1
    audit_release(args.root, matrix=args.matrix, evidence=args.evidence)
    print("v3.0 acceptance matrix and evidence verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
