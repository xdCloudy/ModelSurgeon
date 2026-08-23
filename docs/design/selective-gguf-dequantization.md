# Selective GGUF dequantization

Selective dequantization consumes the already validated GGUF mutation plan and
reads only block spans marked for contiguous-axis repacking. Direct aligned block
removal and outer whole-slice edits require no decode. When outer slices are also
removed, their rows are excluded from the repack read set; untouched prefix
blocks before the first affected block are never loaded.

Encoded reads are split by three simultaneous ceilings: encoded chunk bytes,
decoded value count, and combined working bytes. Decoded buffers use configured
`float32` or `float64` arrays rather than unbounded Python-float lists. The codec
must match the exact tensor quantization type and the live tensor handle must
match the validated type and old shape.

Every actual read records component, tensor, block, element, and byte offsets.
The final report includes total encoded bytes, decoded values, observed peak
working bytes, and whether iteration completed. Stopping iteration early leaves
the report explicitly incomplete.
