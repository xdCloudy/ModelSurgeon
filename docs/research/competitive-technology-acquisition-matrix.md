# Competitive technology acquisition matrix

This is a living record of upstream implementation reviews that have led to
concrete ModelSurgeon work. A source review does not imply that ModelSurgeon has
reproduced the upstream benchmark results.

## Reviewed opportunities

| Field | Torch-Pruning grouped channel importance |
| --- | --- |
| Source | [VainF/Torch-Pruning](https://github.com/VainF/Torch-Pruning), commit [`e80127d7c6935c319a7dd6719b8a72e3c1fcbff9`](https://github.com/VainF/Torch-Pruning/commit/e80127d7c6935c319a7dd6719b8a72e3c1fcbff9); reviewed `torch_pruning/pruner/importance.py` and the DepGraph/Group documentation. |
| Component | `GroupMagnitudeImportance` scores a dependency group as a pruning unit, with per-layer/channel contributions aggregated into group importance. DepGraph discovers coupled modules from an example forward pass. |
| Licence | MIT, verified from the upstream repository metadata and licence at the reviewed revision. No upstream implementation code is included in ModelSurgeon. |
| Current ModelSurgeon equivalent | The Hugging Face MLP physical adapter removes the same intermediate channel index across every transformer layer and updates the shared intermediate dimension. Before this change, proof labels masked one layer, proof candidates duplicated a channel once per layer, and learned inference features represented only one layer. The new proof path enumerates a canonical group candidate, extracts group features, and masks every layer for the training label. |
| Technical advantages | Training labels, candidate identity, learned features, and physical surgery now describe the same coupled group. The implementation aggregates L1 sums, combines L2 norms, and records group parameter count while preserving per-feature aggregation semantics. |
| Hardware requirements | Upstream dependency tracing uses PyTorch autograd and an example input; it can execute on CPU or accelerator but carries model-forward and graph-tracing costs. The ModelSurgeon adaptation requires no new dependency and uses the existing loaded HF proof runtime. |
| Supported architectures | Torch-Pruning documents broad PyTorch module support subject to traceable forward execution. This integration is scoped to ModelSurgeon's existing Hugging Face MLP adapter and its already verified architecture support; it does not expand that support boundary. |
| Integration approach | **ADAPT** the grouped-importance principle independently. Explicit ModelSurgeon architecture adapters already know the coupled mutation, so importing a dynamic dependency-graph implementation would add complexity without improving this path. No code was copied or modified from upstream. |
| Expected benefit | Hypothesis: more representative Meta-Surgeon rankings for global MLP channel mutations. Measure ranking regret and physical quality/compression outcomes against the former single-layer feature path and unguided baselines on held-out multi-layer models. No ranking or quality improvement is claimed yet. |
| Risk | Aggregated static/activation/gradient records are a summary, not a full interaction model. Precision-incompatible groups cannot be safely collapsed and therefore cause Meta guidance to abstain. Existing bundles must explicitly declare the all-layer grouping and provide group features. |
| Validation | Unit-test aggregation and precision mismatch; verify proof-label perplexity against a repeated all-layer mask measurement; run the first-party runtime integration and artifact reload path; compare learned and unguided candidate rankings on held-out multi-layer models; record physical quality, size, and latency measurements before making an efficacy claim. |
| Priority | **P0**: physical candidate semantics must match learned candidate features before learned ordering is relied upon. |

## Acquisition classifications

| Opportunity | Classification | Status |
| --- | --- | --- |
| Torch-Pruning cross-layer group importance | ADAPT | Implemented in the HF MLP proof feature path; evidence of improved downstream decisions remains to be collected. |
