# Native GGUF MLP-channel removal planning

The native MLP planner supports validated Llama and dense Qwen surgery adapters.
One canonical channel set is expanded into exactly three physical edits: gate and
up projection output axes plus the down projection input axis. No attention,
normalization, embedding, or unrelated layer tensor descriptor enters the plan.

New tensor shapes and exact codec storage sizes are derived before payload access.
The new feed-forward width is written to the architecture-specific metadata key,
and parameter and file-size deltas must reconcile with all three new descriptors.
The downstream GGUF alignment gate classifies each edit as whole-slice copy,
direct block copy, or decoded repack and rejects widths that the down-projection
codec cannot represent.

Identity mappings explicitly remove selected logical channel IDs and renumber
every retained channel. The three physical tensor identities remain stable with
new shapes. The resulting physical and quantized plans are serializable and ready
for selective decode/requantization without loading any tensor during planning.
