# GGUF disk preflight

Before opening a staged output, native GGUF surgery computes a conservative physical
allocation estimate containing the complete output size, output-alignment padding,
maximum scratch allocation, and a fixed safety margin. Sparse files, reflinks, and
byte-copy optimizations never reduce this required estimate because their availability
is not guaranteed.

Preflight resolves the output parent and scratch directory to physical filesystem
identities. When both share a filesystem their allocations are combined and one safety
margin is reserved. Separate filesystems each reserve the margin. Any shortfall raises
before an output file is created or source content can be changed.

The successful plan records required and observed free bytes per filesystem. Writers
call `monitor_gguf_disk` with their worst-case remaining allocations at bounded
intervals. A concurrent loss of free space then fails closed while output is still
staged, leaving transactional cleanup to the writer lifecycle.
