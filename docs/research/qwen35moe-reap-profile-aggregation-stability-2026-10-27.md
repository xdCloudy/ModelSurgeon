# REAP profile aggregation stability: leave-one-profile-out

This CPU-only analysis asks whether combining score profiles can make REAP's bottom-expert selection more similar to a calibration profile that did not participate in that fold's aggregation. It uses the three retained 8,192-token Qwen3.5 profiles from the [pinned profile evidence in PR #738](https://github.com/xdCloudy/ModelSurgeon/tree/30aaac1b75a503c1f21fc4608509557ead703a58). The analysis does not create a pruned artifact or measure model quality.

## Method

For each held-out profile, aggregate the other two profiles separately with:

- `mean_minmax`: per-layer min-max normalize each score vector, then average; this matches the aggregation used by the existing POE two-profile candidate.
- `max_minmax`: per-layer min-max normalize, then take the maximum score across profiles. This conservatively protects experts important in any calibration sample.
- `mean_rank`: convert scores to average percentile ranks with ties averaged, then average ranks.

For each of 41 layers, select the bottom 32 of 256 experts with POE's deterministic tie handling and compare that set to the held-out profile's own bottom 32 using Jaccard similarity. The single-profile baseline is the mean similarity of each held-out profile's selection to the other two profiles' selections. The executable implementation is [`tools/compare_reap_profile_aggregation.py`](../tools/compare_reap_profile_aggregation.py); its machine-readable output is [`records/qwen35moe-reap-aggregation-loo-2026-10-27.json`](records/qwen35moe-reap-aggregation-loo-2026-10-27.json).

## Results

| Selector | Mean held-out Jaccard | Worst held-out profile mean |
| --- | ---: | ---: |
| Single-profile selections (pairwise baseline) | 0.3129 | 0.2614 |
| Mean of min-max normalized scores | **0.3618** | **0.2736** |
| Maximum of min-max normalized scores | 0.3517 | 0.2389 |
| Mean rank | 0.3128 | 0.2534 |

On these three profiles, mean-minmax aggregation increased average held-out set agreement by 4.9 percentage points and the worst fold by 1.2 points over the single-profile baseline. Max-minmax and mean-rank did not improve both summaries. These are descriptive measurements from one checkpoint and three profiles, not a statistical significance result; the 41 layers are not independent samples.

The existing physical two-profile comparison remains the quality check: its paired held-out PPL delta versus single-profile REAP was -0.00419 with 95% interval [-0.03598, 0.03031], which includes zero, and both candidates failed the 98% PPL-retention gate. Greater profile-set agreement therefore has **not** been shown to improve model quality.

## Reproduction and limits

Run from the repository root:

```powershell
python tools/compare_reap_profile_aggregation.py `
  --profile docs/research/records/qwen35moe-reap-train-8192.poeprofile `
  --profile docs/research/records/qwen35moe-reap-train-independent-8192-2026-10-27.poeprofile `
  --profile docs/research/records/qwen35moe-reap-validation-8192.poeprofile `
  --output docs/research/records/qwen35moe-reap-aggregation-loo-2026-10-27.json
```

The tool canonicalizes CRLF to LF before hashing profile JSON so its SHA-256 identities match the committed LF files on Windows as well as Linux. The profile hashes are retained in the JSON result. This analysis reuses existing profiles; it runs no model inference and used no GPU. It does not establish an aggregation winner, cross-model transfer, quality retention, deployment improvement, or production support. A next test needs more independent calibration profiles and an untouched held-out quality evaluation for every produced candidate.
