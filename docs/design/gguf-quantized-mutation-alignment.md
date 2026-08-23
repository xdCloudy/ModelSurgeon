# GGUF quantized mutation alignment

The GGUF alignment gate binds every physical tensor edit to one exact native
write codec and recomputes its encoded size from the concrete block layout.
Bindings must cover the edits exactly; unsupported types fail rather than
substituting another member of the same quantization family.

Contiguous-axis edits have two safe strategies. Removing complete aligned blocks
permits direct copying of the remaining encoded blocks. Any other removal whose
new dimension is still block-representable requires decoded/requantized repacking
from the first touched block through the row tail. Outer-axis edits operate only
as whole encoded slices. No API represents naive deletion of arbitrary packed
bytes.

If the new contiguous dimension is not a block multiple, validation fails before
decode or output. A separate proposal API can explicitly expand requested indices
to complete touched blocks and reports every additional index. It never applies
that semantic expansion automatically; callers must accept it and recompile the
physical plan.
