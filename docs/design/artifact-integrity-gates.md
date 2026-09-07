# Physical artifact integrity gates

The shared integrity gate applies the same publication lifecycle to
Safetensors/Hugging Face and GGUF artifacts while allowing each adapter to
provide its structural validator. The bounded stages are source validation,
staging, checksum, structure, reload, generation, and promotion.

Source files must be regular non-symlink files and stay under the byte budget.
The destination is published through the existing non-overwriting atomic
checkpoint boundary. A checksum mismatch, malformed structure, reload failure,
generation failure, interruption, or destination collision rejects the child
and leaves the source and accepted parent untouched.

Fault injection is deterministic at every boundary. Staging recovery removes
only sibling paths owned by the named destination and never performs broad
cleanup. Re-running after an interrupted stage is safe; re-running after an
accepted destination is rejected rather than overwriting the last-known-good
artifact.
