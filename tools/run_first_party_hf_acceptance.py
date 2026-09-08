"""Run a revision-pinned multi-model first-party Hugging Face acceptance campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from modelsurgeon.config import (
    CalibrationConfig,
    ComputeDType,
    ConstraintConfig,
    ModelConfig,
    Settings,
)
from modelsurgeon.first_party_optimize_runtime import build_first_party_optimize_runtime
from modelsurgeon.optimization import build_optimize_plan
from modelsurgeon.optimization_orchestrator import OptimizeOrchestrator

TOOL_REVISION = "first-party-hf-acceptance-v1"


def _model_reference(value: str) -> tuple[str, str]:
    identifier, separator, revision = value.rpartition("@")
    if not separator or not identifier.strip() or not revision.strip():
        raise argparse.ArgumentTypeError(
            "models must use IDENTIFIER@IMMUTABLE_REVISION"
        )
    return identifier.strip(), revision.strip()


def _slug(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._")
    if not result:
        raise ValueError(f"cannot derive an artifact directory from {value!r}")
    return result.lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _resolve_model(identifier: str, revision: str) -> Path:
    local = Path(identifier).expanduser()
    if local.exists():
        if not local.is_dir():
            raise ValueError(f"model path is not a directory: {local}")
        return local.absolute().resolve()
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=identifier, revision=revision))
    except Exception as error:
        raise RuntimeError(
            f"could not resolve Hugging Face model {identifier!r}@{revision!r}: {error}"
        ) from error


def _parse_detail(detail: object) -> dict[str, object] | None:
    if not isinstance(detail, str):
        return None
    try:
        value = json.loads(detail)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _stage_summary(run_record: dict[str, object]) -> list[dict[str, object]]:
    raw_stages = run_record.get("stages")
    if not isinstance(raw_stages, list):
        return []
    summary: list[dict[str, object]] = []
    for raw_stage in raw_stages:
        if not isinstance(raw_stage, dict):
            continue
        result = raw_stage.get("result")
        if not isinstance(result, dict):
            continue
        item: dict[str, object] = {
            "stage": raw_stage.get("stage"),
            "outcome": result.get("outcome"),
            "measured": result.get("measured"),
            "constraints_passed": result.get("constraints_passed"),
            "artifact_digest": result.get("artifact_digest"),
        }
        detail = _parse_detail(result.get("detail"))
        if detail is not None:
            for key in (
                "baseline",
                "measurement",
                "measured_frontier",
                "candidate",
                "deployment",
                "artifact",
                "source_digest",
                "resource_preflight",
                "state_updates",
                "stages",
                "failed_index",
                "rejection_reason",
            ):
                if key in detail:
                    item[key] = detail[key]
        summary.append(item)
    return summary


def _run_cell(
    identifier: str,
    revision: str,
    *,
    calibration_text: Path,
    artifact_root: Path,
    preset: str,
    quality_profile: str,
    evaluations: int | None,
) -> dict[str, object]:
    model_path = _resolve_model(identifier, revision)
    slug = _slug(identifier)
    cell_root = artifact_root / slug
    cell_root.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        artifact_dir=cell_root / "artifacts",
        model=ModelConfig(path=str(model_path), revision=revision, dtype=ComputeDType.FP32),
        calibration=CalibrationConfig(
            dataset=str(calibration_text),
            samples=2,
            max_sequence_length=32,
            seed=1729,
        ),
        constraints=ConstraintConfig(min_quality_retention_ratio=0.99),
    )
    plan = build_optimize_plan(
        settings,
        preset=preset,
        quality_profile=quality_profile,
        dry_run=False,
    )
    if evaluations is not None:
        plan = replace(plan, budget=replace(plan.budget, evaluations=evaluations))
    cell: dict[str, object] = {
        "model": identifier,
        "revision": revision,
        "resolved_model_path": str(model_path),
        "plan": plan.to_record(),
    }
    if not plan.executable:
        cell.update(
            {
                "status": "not_executed",
                "outcome": plan.outcome.value,
                "uncertainties": list(plan.uncertainties),
                "limitations": list(plan.limitations),
            }
        )
        return cell
    state_path = cell_root / "run.json"
    approvals = tuple(item.code for item in plan.approvals if item.required)
    run = OptimizeOrchestrator(
        plan,
        state_path,
    ).run(
        build_first_party_optimize_runtime(plan),
        approvals=approvals,
    )
    run_record = run.to_record()
    cell.update(
        {
            "status": run.status.value,
            "outcome": run.outcome.value,
            "run_id": run.run_id,
            "source_artifact_digest": run.source_artifact_digest,
            "accepted_artifact_digest": run.accepted_artifact_digest,
            "run_record_path": str(state_path),
            "stage_summary": _stage_summary(run_record),
            "run": run_record,
        }
    )
    return cell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        type=_model_reference,
        help="revision-pinned IDENTIFIER@REVISION; repeat for each campaign cell",
    )
    parser.add_argument("--calibration-text", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--preset", default="balanced")
    parser.add_argument("--quality-profile", default="quality")
    parser.add_argument("--evaluations", type=int)
    args = parser.parse_args()
    if len(args.model) < 2:
        parser.error("acceptance campaigns require at least two model cells")
    if args.evaluations is not None and args.evaluations <= 0:
        parser.error("--evaluations must be positive")
    calibration_text = args.calibration_text.expanduser().absolute().resolve()
    if not calibration_text.is_file():
        parser.error(f"calibration text does not exist: {calibration_text}")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    identifiers = [identifier for identifier, _ in args.model]
    if len(identifiers) != len(set(identifiers)):
        parser.error("campaign model identifiers must be unique")
    cells: list[dict[str, object]] = []
    failures = False
    for identifier, revision in args.model:
        try:
            cell = _run_cell(
                identifier,
                revision,
                calibration_text=calibration_text,
                artifact_root=args.artifact_root,
                preset=args.preset,
                quality_profile=args.quality_profile,
                evaluations=args.evaluations,
            )
        except Exception as error:
            failures = True
            cell = {
                "model": identifier,
                "revision": revision,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        if cell.get("status") == "failed" or cell.get("outcome") == "failed":
            failures = True
        cells.append(cell)
    record = {
        "record_type": "first_party_hf_acceptance_campaign",
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "recorded_at": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "command_contract": {
            "preset": args.preset,
            "quality_profile": args.quality_profile,
            "evaluations": args.evaluations,
            "min_quality_retention_ratio": 0.99,
            "calibration_text": str(calibration_text),
            "calibration_sha256": _sha256(calibration_text),
        },
        "models": cells,
    }
    args.output.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "record_type": record["record_type"],
        "output": str(args.output),
        "models": [
            {
                "model": item.get("model"),
                "revision": item.get("revision"),
                "status": item.get("status"),
                "outcome": item.get("outcome"),
                "accepted_artifact_digest": item.get("accepted_artifact_digest"),
            }
            for item in cells
        ],
    }, sort_keys=True))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
