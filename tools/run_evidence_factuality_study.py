"""Replay the bounded v2.8 evidence-grounding factuality protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modelsurgeon.evaluation.evidence_factuality_study import (
    load_and_run_evidence_factuality_study,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / "tests" / "fixtures" / "evidence_factuality_study_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the complete canonical retained run instead of the summary",
    )
    args = parser.parse_args()
    run = load_and_run_evidence_factuality_study(args.study)
    if args.json:
        print(run.canonical_json())
        return 0
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "passed": run.passed,
                "stop_reason": run.stop_reason,
                "methods": [item.to_record() for item in run.metrics],
                "retained_results": len(run.results),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
