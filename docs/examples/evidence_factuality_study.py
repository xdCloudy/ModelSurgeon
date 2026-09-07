"""Run the bounded v2.8 evidence-grounding factuality study."""

from __future__ import annotations

import json
from pathlib import Path

from modelsurgeon.evaluation.evidence_factuality_study import (
    load_and_run_evidence_factuality_study,
)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    study = root / "tests" / "fixtures" / "evidence_factuality_study_v1.json"
    run = load_and_run_evidence_factuality_study(study)
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "passed": run.passed,
                "stop_reason": run.stop_reason,
                "metrics": [item.to_record() for item in run.metrics],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
