# Dense four-family meta-surgeon transfer study

The v1.6 transfer study is an execution and evidence boundary for Llama,
Qwen, Mistral, and Gemma. `build_meta_surgeon_transfer_matrix` creates a
deterministic matrix of target checkpoints, source-family combinations, and
feature views. The matrix includes cold-start, within-target, and multi-family
meta controls, with raw and architecture-normalized feature variants.

Each cell names its transfer axis explicitly: checkpoint, size, family, or
corpus. A target model is never included in its source model IDs. The source
model revisions, corpus revisions, hardware profile, seed set, and example
counts remain in cell provenance so a result cannot be detached from the
measurement context.

## Evidence and claims

Every measured or negative-result cell reports five separate metrics:

- ranking;
- calibration;
- regression;
- cumulative frontier quality; and
- evaluation savings.

Intervals are grouped-bootstrap intervals over model-level units. Three
training seeds are required by the default protocol. Unsupported, failed, and
unknown cells remain explicit outcomes with reasons; they are not converted to
zeroes or omitted from the denominator.

The matrix is a plan and result manifest, not fabricated benchmark output.
Licensed model/data artifacts and the applicable hardware runner produce the
measurements, which are attached incrementally with
`record_meta_surgeon_transfer_result` for safe resume. Claims must name the
transfer axis and must not generalize beyond the four configured families,
declared sizes, corpus revisions, and measured cells.
