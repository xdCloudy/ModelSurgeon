# Local artifact registry v1

`modelsurgeon.registry.LocalArtifactRegistry` is the offline provider boundary
for local model, surgeon, and evidence objects. It wraps the existing
content-addressed artifact store with a catalog that controls visibility. No
registry command makes an implicit network request.

## Catalog and identity

Objects remain immutable SHA-256 content-addressed bytes. Catalog records retain
kind, schema, license, tags, source/accepted/external-protection flags,
lineage, and detached signature metadata. Aliases and tags are secondary
references: they never change the object digest. Alias reassignment is rejected
unless it names the same object.

The provider supports deterministic listing with kind/tag filters and digest
cursors, inspection, offline content verification, and metadata comparison.
The CLI and Python API serialize the same stable records.

## Visibility and garbage collection

An object is protected from collection when it is marked as a source, accepted,
or externally unresolved; named by an alias; held by a lease; or referenced by
another catalog object. Collection defaults to a dry-run preview. `--apply` is
required to remove unprotected objects, and removal only occurs after the
catalog has identified the exact digest set.

## Bundles and trust

Exports are deterministic ZIP bundles containing exactly `manifest.json` and
`payload`. Import validates the bundle schema, complete member set, payload
digest and size, license allow-list, and optional detached HMAC signature before
adding the object to the visible catalog. Evidence bundles require a signature
unless the caller explicitly opts into the retained unsigned negative path.
Import publishes only after all checks pass; failed imports do not create a
visible catalog record.

```text
modelsurgeon registry list --root artifacts/registry --json
modelsurgeon registry verify sha256:... --root artifacts/registry
modelsurgeon registry export sha256:... bundle.zip --root artifacts/registry
modelsurgeon registry gc --root artifacts/registry
modelsurgeon registry gc --root artifacts/registry --apply
```
