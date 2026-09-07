# Pretrained meta-surgeon registry

`PretrainedSurgeonRegistry` adds a signed, offline-resolvable model-card layer
around the existing content-addressed surgeon bundle store. A card declares
the source model revisions and evidence digests, feature/target/state schema
versions, supported mutations/codecs/hardware, calibration policy,
compatibility metadata, license, metrics, limitations, and optional immutable
parent/request lineage for adapted children.

Publishing signs the canonical card with an HMAC key and refuses references to
missing bundles. Verification checks the card schema, signature, content
identity, and bundle digest without deserializing model bytes. Resolution
performs expected-schema checks against the verified card first, then loads
the bundle. This preserves the fail-closed boundary for incomplete or
incompatible cards and avoids implicit downloads.

Adapted children carry a distinct content digest and parent bundle digest;
the registry never overwrites a parent. `verify`, `resolve`, and `list_cards`
are local APIs, so callers can audit and reproduce a card without network
access.
