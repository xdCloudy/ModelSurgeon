# Experiment reproducibility manifest

Every persisted experiment run can publish one immutable `ReproducibilityManifest` as a content-addressed artifact with role `reproducibility_manifest`.

## Captured evidence

The schema records the run, experiment, and attempt identities together with:

- the exact full Git object ID and clean/dirty worktree state;
- the resolved configuration digest;
- model identifier, exact model revision, family, format, parameter count, and quantization;
- dataset identifier, exact revision, split, calibration manifest ID, tokenizer, and tokenizer revision;
- experiment, data, and mutation seeds;
- tool/evaluator and referenced schema versions;
- the full hardware/software inventory, including Python, optional PyTorch, CUDA runtime/driver/device evidence, CPU, RAM, disk, and OS details; and
- a SHA-256 digest of the dependency lock file such as `uv.lock`.

The manifest ID is a SHA-256 identity over the canonical immutable evidence. Status fields are derived and do not affect identity.

## Captured does not mean reproducible

`ReproducibilityManifest.reproducible` is derived from the evidence. `require_reproducible()` fails closed when required evidence is absent or the source worktree is dirty.

Existing experiment schema types already reject blank model, dataset, tokenizer, tool, and evaluator revision fields before a run can be built. The reproducibility layer additionally requires:

- an exact full Git SHA-1 or SHA-256 object ID;
- a clean Git worktree; and
- a SHA-256 dependency lock digest.

A missing optional runtime such as PyTorch or CUDA is recorded as absent by the hardware inventory rather than invented. CPU-only runs remain representable and reproducible when their actual environment is completely captured.

## Immutable run linkage

`publish_reproducibility_manifest()` verifies that the supplied `PersistedExperiment` and stored candidate belong to the manifest run before publication. It then writes canonical JSON through the existing content-addressed artifact store and records the immutable digest in the experiment metadata store.

Repeated publication of the same manifest reuses the same artifact and reference. A manifest cannot be attached to a different run.

## Collection helpers

`collect_git_revision()` records the exact `HEAD` object ID and whether tracked or untracked worktree changes exist. `digest_lock_file()` streams a regular, non-symlink lock file in bounded chunks and returns its SHA-256 identity.
