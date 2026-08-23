# Architecture Family Detection

ModelSurgeon selects architecture adapters only from explicit model metadata. It does not infer a family from repository names, filesystem paths, tensor-name resemblance, or parameter shapes.

`ArchitectureEvidence` accepts Hugging Face `model_type`, Hugging Face architecture class names, and the GGUF `general.architecture` value. `detect_model_family` normalizes and matches these fields against an explicit, version-controlled alias table for Llama, Qwen, Mistral, and Gemma.

All matching evidence must agree. Contradictory fields raise `ConflictingArchitectureError`; absent or unknown aliases raise `UnknownArchitectureError`. The selected family retains every normalized matching field as persistence-safe provenance. Adding an architecture or version therefore requires an explicit alias and test rather than a fuzzy fallback.

