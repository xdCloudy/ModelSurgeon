# Active-learning retraining and promotion

Issue #115 triggers retraining when any configured threshold is reached: new example count,
elapsed budget, or a non-negative drift signal. The decision records every firing reason so
the new surgeon version has an auditable cause.

Challenger promotion is separate and explicit. Each unique validation metric declares
maximize/minimize direction and minimum improvement. A successful challenger must provide
finite incumbent/challenger values and pass every criterion. The versioned decision records
each improvement and threshold.

A training failure immediately retains the incumbent regardless of any partial or supplied
challenger metrics. Missing, non-finite, or insufficient validation evidence also retains
the incumbent. Promotion therefore requires an affirmative all-criteria-passed decision;
creating a challenger artifact never changes the active version by itself.
