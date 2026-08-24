# Reproducible reports

The report generator accepts a fully resolved run, campaign, or search record and emits
canonical JSON plus a dependency-free HTML rendering. It does not query mutable state or
read the clock. A timestamp appears only when the caller supplies it, so identical inputs
produce byte-identical outputs and SHA-256 identities.

The shared record contains resolved configuration, checkpoint/generation lineage, named
metrics, inline plot points, structured failures, hardware/runtime context, and immutable
artifact IDs. HTML renders those same values with internal identity anchors, inline CSS,
and inline SVG; it has no network resources or executable plotting dependency. A safely
escaped copy of the complete JSON record is embedded as `application/json` for offline
inspection.

Redaction happens before either output is rendered. Configured path prefixes are replaced
with `<redacted-path>` while preserving a non-sensitive suffix, and canonical secret keys
such as `token`, `password`, and `api_key` are replaced wholesale. Redaction recurses
through configuration, lineage, failures, hardware, links, and plot labels. Unsupported
objects, non-string mapping keys, non-finite values, duplicate metrics/plots/identities,
and implicit timestamps fail closed.
