# Chat inspection context

`modelsurgeon chat` attaches a versioned `chat_inspection_context` record to
the session bootstrap and to each provider interpretation request. The record
is engine-owned context for the intent compiler; it is not a second
optimization authority.

The context is assembled from direct ModelSurgeon APIs:

- GGUF metadata and `discover_gguf_components()` provide detected architecture,
  family, content revision, and canonical component evidence.
- `build_default_hardware_profile()` provides CPU, CUDA, memory, disk,
  software, runtime-availability, and retained probe outcomes. CPU-only and
  low-memory hosts remain valid records; missing measurements are `unknown`.
- The architecture compatibility matrix provides explicit per-operation
  `verified`, `experimental`, `unsupported`, and `unknown` cells. Its static
  capability matrix is not a live benchmark.
- The provider capability card and accepted runtime revision are labeled as
  provider declarations. Provider-generated text is never promoted to detected
  inventory or measured evidence.

Every section has an explicit `supported`, `unsupported`, `failed`, or
`unknown` outcome. Incomplete provider-only GGUF metadata can support chat
startup while structural discovery remains `unknown`; conflicting or invalid
metadata fails closed. The context is content-addressed by `inspection_id`
and bounded to one MiB before it is passed to a provider.

No-provider mode retains explicit unavailable model/runtime evidence and does
not infer architecture or capabilities from the request path. Direct
inspection remains available through `inspect_local_chat_model()` and does not
start a provider or mutate a model.
