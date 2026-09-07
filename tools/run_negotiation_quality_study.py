"""Replay the bounded v2.5 negotiation decision-quality protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modelsurgeon.conversation import load_and_run_negotiation_study

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / "tests" / "fixtures" / "negotiation_decision_quality_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the complete canonical retained run instead of the summary",
    )
    args = parser.parse_args()
    run = load_and_run_negotiation_study(args.study)
    if args.json:
        print(run.canonical_json())
        return
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "passed": run.passed,
                "stopped": run.stopped,
                "policies": [item.to_record() for item in run.metrics],
                "retained_results": len(run.results),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
