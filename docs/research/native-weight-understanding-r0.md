# Native weight understanding R0 fixture protocol

**Status: synthetic protocol fixture only. No learned Meta-Surgeon result.**

Date: 2026-11-02

Study version: `1`

Implementation: `modelsurgeon.evaluation.weight_understanding`

## Question

Can a small independent evaluator verify known function and hidden-unit intervention labels when an analytic network is changed by function-preserving hidden-unit permutations and positive ReLU rescalings?

This is the first protocol check for the proposed native weight-understanding research direction. It checks whether the fixture and causal labels survive simple parameterization symmetries before teacher or student annotations are collected. It does **not** test whether a model can infer a function from weights.

## Protocol

The runner constructs four one-hidden-layer ReLU circuits over the exhaustive binary input domain `{0, 1}²`: `AND`, `OR`, `XOR`, and `copy_first`. Their expected output tables are specified independently of the candidate evaluator. The task probe evaluates each network on all four inputs and compares the resulting output vector with the four pinned function fingerprints.

For each circuit, the runner samples a hidden-unit permutation and positive scale factors from `[0.125, 8]`. It permutes the incoming and outgoing connections together and applies the exact positive-homogeneity transform for ReLU. The measured circuit target is the root mean square output change over the same complete input domain after deleting each hidden unit's outgoing edge. That target is compared by original-to-transformed unit identity.

Run command:

```bash
uv run python -c 'import json; from modelsurgeon.evaluation.weight_understanding import run_weight_understanding_study; print(json.dumps(run_weight_understanding_study(seed=42, repetitions=100).to_record(), indent=2, sort_keys=True))'
```

## Observed result

With seed `42` and 100 transformations per task, the runner evaluated 400 transformed networks.

| Measurement | Result |
| --- | ---: |
| Exact functional-probe task accuracy | 1.0 |
| Maximum output difference after symmetry transform | 3.33e-16 |
| Maximum hidden-unit causal-effect difference after mapping identities | 1.11e-16 |

These are deterministic fixture checks, not estimates over a sampled model population. The near-zero errors are floating-point roundoff and show that the implementation preserves these known outputs and intervention labels across the tested transformations.

## Interpretation and limits

- The task classifier uses exhaustive black-box outputs and known task fingerprints. It is a verifier for these fixtures, not a neural predictor and not a weight-reading algorithm.
- The circuits are hand-constructed, tiny, dense, and noise-free. No trained or pretrained checkpoint, real task distribution, held-out architecture family, teacher annotation, hardware budget, physical surgery, or deployment metric was evaluated.
- There is no random, magnitude, activation, or learned-surgeon comparison because this protocol validates labels and invariances rather than optimization quality.
- The study does not reproduce or validate external weight-space-programming findings and makes no claim about the function of pretrained model circuits.

## Next experiment

Use these fixtures to test typed hypothesis verification and explicit abstention, then add independently trained small networks with lineage-held-out functional and causal tasks. Only after labels and splits are independently audited should teacher-annotated and teacher-free students be compared under identical measured targets. Any claimed optimization benefit still requires physical artifacts and matched random and heuristic controls.
