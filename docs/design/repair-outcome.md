# Unified repair outcome and artifact lineage

`RepairRequest` and `RepairResult` normalize no-repair, LoRA,
selected/full fine-tuning, and logit/feature distillation. Every request names
source, candidate, optional teacher, data revision, tokenizer digest,
compatibility evidence, trainable scope, equal resource budgets, best-state
policy, and quantization order.

Results separate mechanics from effectiveness. Accepted repairs require a
published child artifact, immutable parent/source/teacher/data/tokenizer
lineage, and measured held-out improvement evidence. Artifact sizes reconcile
with the disk-cost field, including separate LoRA and merged/full-checkpoint
outputs.

Rejected, failed, unsupported, and unknown repairs must restore the exact
candidate, publish no search artifact, and remain ineligible as search parents.
`RepairOutcomeManifest` records results append-only for safe resume; it never
replaces a parent or an earlier result.
