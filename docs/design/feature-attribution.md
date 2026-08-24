# Feature attribution contract

ModelSurgeon produces local additive explanations only where the persisted surgeon model
has a bounded, deterministic attribution method. `attribute_predictions` accepts already
preprocessed finite rows in the model's exact feature order and returns one contribution
per feature plus a bias term.

Linear regression and logistic regression use exact coefficient-times-input
contributions. Regression reconciles in prediction space. Logistic regression reconciles
in raw-logit space because applying the sigmoid would destroy additivity. LightGBM uses
its TreeSHAP contribution mode and likewise reconciles against the booster's raw score;
this applies to both regression and classification. Every report names its output space,
technique, tolerance, reconstructed prediction, and absolute reconciliation error.

Feature provenance is derived from the leakage-safe preprocessing schema:

- `num:name` identifies a normalized numeric source;
- `missing:name` identifies its explicit missingness indicator;
- `cat:name=value` identifies a categorical source and encoded category;
- any other stable model feature is labeled as a derived feature.

When a missingness indicator is active, both the numeric value and its indicator are
marked missing so consumers do not interpret a mean-imputed zero as an observed zero.
Contributions remain separate: the numeric coefficient effect and missing-indicator effect
are never merged or invented.

MLP and unknown surgeon model types return an `AttributionUnavailable` record naming the
model kind, reason, and the available permutation fallback. The attribution API does not
silently run a more expensive approximation. Empty rows, non-finite inputs, incompatible
feature widths, non-finite contributions, and reconciliation beyond the declared tolerance
fail closed.
