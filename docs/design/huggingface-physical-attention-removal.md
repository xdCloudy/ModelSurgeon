# Hugging Face physical attention-head removal

Issue #120 supports standard MHA and grouped-query attention without semantic regrouping.
Query heads may be removed only as complete contiguous groups belonging to KV heads; MHA is
the group-size-one case. Invalid query/KV divisibility, partial groups, missing projections,
shape drift, or removal of all query/KV heads fails before mutation.

Every layer removes Q output and O input scalar blocks for the selected query heads, plus K
and V output blocks for their complete KV groups. Global query/KV counts and available
per-layer head/group metadata are updated while head dimension and hidden output width stay
fixed. Replacement projections preserve module type, dtype, device, bias, and gradient flags.
