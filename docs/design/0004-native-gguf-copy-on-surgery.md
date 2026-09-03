# ADR 0004: Native GGUF copy-on-surgery

Status: Accepted

GGUF is a first-class physical-surgery format. Source files are memory-mapped; unchanged tensors are copied byte-for-byte when possible; only affected quantization blocks are decoded and re-encoded; output is streamed transactionally under explicit RAM/VRAM/disk budgets. Unsupported architectures or layouts fail closed. Untrusted and malformed container bytes, including empty files and mmap/open failures, are exposed through the typed GGUF parser-error boundary rather than native exceptions.

