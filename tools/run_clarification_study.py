"""Run the bounded v2.4 clarification measurement protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modelsurgeon.conversation import load_and_run_clarification_study

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / "tests" / "fixtures" / "clarification_measurement_v1.json"
DEFAULT_INTENT_CORPUS = ROOT / "tests" / "fixtures" / "intent_compiler_corpus_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--intent-corpus", type=Path, default=DEFAULT_INTENT_CORPUS)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the complete canonical retained run instead of the summary",
    )
    args = parser.parse_args()
    run = load_and_run_clarification_study(args.study, args.intent_corpus)
    if args.json:
        print(run.canonical_json())
        return
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "passed": run.passed,
                "policies": [item.to_record() for item in run.metrics],
                "retained_results": len(run.results),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
