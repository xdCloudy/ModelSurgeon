# Hardware-cost predictors

`modelsurgeon.surgeon.cost_predictors` is the v1.3 predictor boundary built on
the profile-partitioned `HardwareCostDataset`.

## Contract

`fit_cost_predictors()` trains one deterministic log-ridge model for each
requested target: load time, prefill throughput, decode throughput, latency,
RAM, and VRAM. The feature vocabulary is fitted on the training partition and
retains model family/architecture, hardware profile/context, runtime revision,
runtime settings, and log artifact size. Numeric normalization and categorical
vocabularies are retained in the typed `CostFeatureSchema`.

The split is consumed exactly as declared by `HardwareCostSplit`. The predictor
also checks the configured model, artifact, and hardware held-out dimensions;
overlap produces a failed evidence cell rather than a silently optimistic
result. Unsupported, failed, unknown, and negative cells remain in the
`CostPredictorStudy` record.

## Calibration and claims

Training fits the log target. Validation absolute log residuals calibrate a
prediction interval at the configured confidence level. Test evidence reports
coverage, MAE, RMSE, rank correlation, p95 absolute error, serialized model
size, and a stable operation-count inference-cost proxy.

The learned predictor is compared with two preregistered baselines: a training
mean and an artifact-size analytic scaling rule. A positive claim is emitted
only when the predictor has lower held-out MAE than both baselines. Otherwise
the target is retained as `negative_result`; prose or an unsupported cell never
becomes a success claim.

Inference rejects unseen categorical context, missing artifact size, and
numeric values outside the training range by default with
`out_of_distribution`. A caller may request a flagged research prediction,
but the result remains marked out of distribution and is not eligible for
optimization use.

## Reproduction

```python
from modelsurgeon.surgeon import CostPredictorConfig, fit_cost_predictors

study = fit_cost_predictors(
    hardware_cost_dataset,
    config=CostPredictorConfig(seed=0, confidence=0.95),
)
record = study.to_record()
prediction = study.predict(example, "latency_seconds")
```

The study, model IDs, source dataset ID, split dimensions, feature schema,
training/calibration example IDs, benchmark metrics, and baseline comparisons
are all retained in the versioned record.
