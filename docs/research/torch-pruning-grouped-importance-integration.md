# Torch-Pruning grouped-importance integration

This implementation note records the specific adaptation and validation in
ModelSurgeon's all-layer MLP channel path. The broader competitor acquisition
matrix is tracked in [PR #735](https://github.com/xdCloudy/ModelSurgeon/pull/735).

## Upstream and provenance

- Repository: [VainF/Torch-Pruning](https://github.com/VainF/Torch-Pruning)
- Reviewed revision: [`e80127d7c6935c319a7dd6719b8a72e3c1fcbff9`](https://github.com/VainF/Torch-Pruning/commit/e80127d7c6935c319a7dd6719b8a72e3c1fcbff9)
- Licence: MIT, verified from upstream metadata and the repository licence.
- Reviewed entry points: `torch_pruning/pruner/importance.py` and the DepGraph/Group documentation.
- Reuse: none. The group aggregation and mask coordination in ModelSurgeon are independently implemented; no upstream code or dependency was added.

## Adaptation

Torch-Pruning treats coupled modules as one pruning group and aggregates their
importance. ModelSurgeon has explicit architecture adapters that already know
the physical coupling: removing an intermediate MLP channel index removes it
from all transformer layers. The HF proof path now uses one canonical
layer-0 candidate, aggregates per-layer pre-mutation features, and masks the
same channel in every layer when measuring the proof label. Inference checks a
signed bundle's declared grouping and abstains when group features or compatible
precision are unavailable.

This is an adaptation of the grouped-pruning principle, not an adoption of
Torch-Pruning's dynamic autograd dependency graph. That tracing path would add a
model forward and handler requirements where ModelSurgeon's adapter already
defines the coupling explicitly.

## Validation and limits

The tiny two-layer Llama proof integration verifies canonical candidate
enumeration, group feature fields, and equality between the recorded proof
perplexity and a repeated all-layer channel mask. The first-party runtime test
checks signed compatibility and physical artifact reload. These tests establish
semantic consistency only; they do not show improved quality retention, ranking
regret, compression, or inference speed. Those claims require held-out real
models and equivalent unguided, magnitude, and activation baselines.
