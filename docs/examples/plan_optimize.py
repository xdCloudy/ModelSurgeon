"""Build a deterministic, read-only ModelSurgeon optimization plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from modelsurgeon.config import ModelConfig, Settings
from modelsurgeon.optimization import build_optimize_plan, write_optimize_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Immutable local source path or identifier")
    parser.add_argument("--revision", required=True, help="Immutable source revision")
    parser.add_argument("--output", type=Path, required=True, help="New plan JSON path")
    parser.add_argument("--preset", default="balanced", choices=("fast", "balanced", "quality"))
    parser.add_argument("--hardware-profile", default="cpu-small")
    args = parser.parse_args()

    settings = Settings(model=ModelConfig(path=args.model, revision=args.revision))
    plan = build_optimize_plan(
        settings,
        preset=args.preset,
        hardware_profile=args.hardware_profile,
        quality_profile=args.preset,
    )
    if not plan.executable:
        print(f"{plan.outcome.value}: {plan.uncertainties}")
        return 2
    write_optimize_plan(args.output, plan)
    print(f"supported {plan.plan_id} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
