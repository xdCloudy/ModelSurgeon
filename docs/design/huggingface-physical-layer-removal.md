# Hugging Face physical transformer-layer removal

Issue #121 removes complete blocks from the standard `model.layers` ModuleList, rejects
noncanonical/all-layer removal, updates global layer count, and records every old-to-new
canonical index (removed indices map to null).

Retained module objects and parameters are reused without copying or rewriting. Adapter
execution indices such as Llama attention `layer_idx` are renumbered to match the shortened
KV-cache layout; failing to do this produces an out-of-range cache access even when tensor
state is otherwise valid. The result reconciles whole-model parameter counts.

A behavioral reference bypasses the removed block by returning its residual input. Physical
forward output must match that bypass before repair, and the updated checkpoint must save,
reconstruct, reload, and forward successfully.
