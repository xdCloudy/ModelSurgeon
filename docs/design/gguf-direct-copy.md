# Byte-for-byte GGUF tensor copying

An unchanged tensor is represented by its validated source-index handle and an explicit
encoded chunk ceiling. The bridge resolves the original `ggml_type`, dimensions, and
physical name directly from that handle and yields complete codec blocks from the
read-only memory map into the transactional writer. It never decodes, requantizes, or
collects a complete tensor.

The maximum retained payload allocation is the largest whole-block multiple fitting
the configured chunk ceiling. The writer records a SHA-256 digest for every tensor as
it streams, allowing copied payloads to be compared with source or conformance digests
without another destination pass. Invalid, foreign, or stale handles and chunk limits
smaller than one encoded block fail before output construction.
