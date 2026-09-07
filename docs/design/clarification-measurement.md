# Clarification measurement protocol

This is the preregistered v2.4 measurement for necessary-question rate and
clarification safety. It measures the existing deterministic clarification
state machine; it does not give a provider, transcript, or baseline policy
authority to change an objective.

## Protocol and bound

The protocol is `v24-clarification-measurement-v1` and the metadata is in
[`clarification_measurement_v1.json`](../../tests/fixtures/clarification_measurement_v1.json).
The source intents are bound to the exact
`intent-compiler-corpus/v2.1-intent-corpus-v1` revision. The corpus has 11
typed request cases: fixture cases, two held-out paraphrases, and adversarial
contradiction, refusal, unsupported-operation, unsupported-budget, and
instruction-like-text cases. Inconclusive low-confidence and missing-hard-
constraint cases remain in the corpus rather than being discarded.

The run is provider-independent and uses no replay seeds (`[]`). It is bounded
to at most 64 cases and 16 questions per case. Each policy/case cell retains
the deterministic result ID, intent ID, question IDs/categories, expected and
observed outcomes, spec-equivalence result, mismatch list, and error. A
provider replay can be added only as one of at most three explicitly recorded
non-negative seeds; no live provider is required for this evidence.

Run it from the repository root:

```text
python tools/run_clarification_study.py
python tools/run_clarification_study.py --json
```

The JSON form is the complete retained evidence record. The run ID is derived
from canonical JSON, so repeating the same corpus and implementation produces
the same identifier and byte-stable output.

## Policies and metrics

The comparison is intentionally small:

- `schema_driven`: the existing `ClarificationMachine` question set;
- `schema_only`: no question layer around policy evaluation;
- `minimal_question`: at most one question when clarification is required;
- `over_questioning`: schema-driven questions plus deterministic confirmation
  questions for every declared field.

Gold labels are declared in the study metadata, not inferred from the policy
under test. The preregistered metrics are:

- necessary-question recall: necessary cases with at least one question;
- unnecessary-question rate: no-question-needed cases that receive a question;
- silent-constraint-invention rate: non-executable gold cases that become
  executable;
- executable-spec precision/recall and exact spec-equivalence rate;
- refusal correctness and mean clarification cost (questions per case).

The shipping thresholds for `schema_driven` are necessary-question recall
`>= 1.0`, unnecessary-question rate `<= 0.0`, silent-constraint-invention
rate `== 0.0`, exact spec equivalence `== 1.0`, and refusal correctness
`== 1.0`. If a future corpus or run misses a threshold, the policy remains
fail-closed and the negative/inconclusive results are retained; no threshold
is relaxed to make the result pass.

## Results and decision

The bounded CPU run on this revision retained 44 cells (11 cases x 4
policies). `schema_driven` measured necessary-question recall `1.0`,
unnecessary-question rate `0.0`, silent-constraint-invention rate `0.0`,
executable spec precision/recall `1.0/1.0`, exact spec equivalence `1.0`,
and refusal correctness `1.0`. The minimal baseline reached recall `1.0`
with no unnecessary questions on this corpus but does not preserve the
schema-driven question targeting contract. `schema_only` reached recall `0.0`.
The over-questioning baseline reached an unnecessary-question rate of `1.0`.
The retained run ID is
`clarification_study_run_9d8144b2aafceceea092c8914aeadb2ce7e95a00ae7d7f3be57b3e6c1a3e46ae`.

Decision: retain the existing schema-driven policy and its fail-closed
behavior. These fixture results do not justify provider-dependent thresholds
or a relaxation of hard-constraint handling. The complete negative and
inconclusive cells remain reproducible with `--json` and are covered by
`tests/test_clarification_study.py`.
