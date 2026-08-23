# Post-surgery component identity remapping

Every source component receives an explicit mapping to zero or more canonical
post-surgery IDs. One unchanged target is retained, no targets is removed, one
different target is renumbered, multiple targets is split, and multiple sources
sharing a target is merged. Combined split/merge topology is retained rather than
collapsed to an arbitrary single ID.

Resolution fails for both removed and unrecorded sources. There is no implicit
“unchanged” fallback, because that could silently redirect experiment evidence
after physical edits. Sequential remaps compose old-to-middle and middle-to-new
only when every intermediate ID has an explicit disposition. Removal propagates,
targets are deduplicated canonically, and reasons from each stage remain attached.

The mutation-record serializer uses this graph-owned mapping record directly, so
runtime composition and persisted outcomes share one representation.
