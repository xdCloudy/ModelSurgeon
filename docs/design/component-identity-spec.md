# Canonical Component Identity Specification

Status: v1 draft, normative for v0.1 implementation

## Purpose

A component identity addresses one logical part of one model architecture. The same identity is used by discovery, instrumentation, feature records, mutation plans, evaluations, datasets, surgeon predictions, and reports. It describes logical structure rather than a tensor's current memory address or Python object identity.

Physical surgery can remove or renumber components. An identity therefore never silently changes meaning: every structural mutation emits an explicit identity map from the input graph revision to the output graph revision.

## Text grammar

The canonical UTF-8 text form is a dot-separated path rooted at `model`.

```abnf
component-id = "model" *( "." segment )
segment      = simple-name / index / escaped-name
simple-name  = ( ALPHA / "_" ) *( ALPHA / DIGIT / "_" / "-" )
index        = "0" / ( %x31-39 *DIGIT )
escaped-name = "~" 1*( unreserved / pct-encoded )
unreserved   = ALPHA / DIGIT / "_" / "-"
pct-encoded  = "%" HEXDIG HEXDIG
```

ABNF terminals are ASCII. Hexadecimal digits in a canonical percent escape are uppercase. An escaped name encodes UTF-8 bytes, must decode to non-empty valid UTF-8, and cannot contain NUL or a Unicode control character.

`escaped-name` is used only when the decoded provider name cannot be represented by `simple-name`. This gives every name one representation:

| Provider name | Canonical segment | Reason |
|---|---|---|
| `self_attn` | `self_attn` | Simple name |
| `123` | `~123` | Distinct from numeric index `123` |
| `with.dot` | `~with%2Edot` | Dot is the path separator |
| `50%` | `~50%25` | Percent introduces an escape |
| `snowman☃` | `~snowman%E2%98%83` | Non-ASCII UTF-8 bytes are escaped |

Escaping a name that is already a valid `simple-name`, using lowercase hexadecimal, or escaping an unreserved byte is non-canonical and rejected. Parsers do not normalize malformed or ambiguous input silently.

## Segment types and semantic kinds

A raw numeric segment is a non-negative index. A simple or escaped segment is a name. Component kind is graph metadata, not an extra syntactic marker in the identity. The architecture adapter validates path roles and assigns a versioned `ComponentKind` such as:

- `model`
- `transformer_layer`
- `attention`
- `attention_head` and `kv_head`
- `projection`
- `mlp`
- `mlp_channel`
- `embedding`
- `normalization`
- `residual_path`
- `moe_expert` and `moe_router`
- `parameter` and `tensor`

This separation allows architecture-specific names while preventing a persisted ID from depending on a Python class. A syntactically valid ID can still be absent from a particular graph or invalid for an adapter-declared kind.

## Canonical examples

```text
model
model.embed_tokens
model.norm
model.layers.17
model.layers.17.self_attn
model.layers.17.self_attn.head.4
model.layers.2.self_attn.kv_head.1
model.layers.8.self_attn.q_proj
model.layers.12.mlp.up_proj.channel.1830
model.layers.5.mlp.expert.3
model.layers.5.mlp.router
model.layers.0.module.~with%2Edot
```

The required examples cover transformer layers, attention and KV heads, MLP channels, Q/K/V/O-style projections, embeddings, normalization, residual paths, and MoE experts/routers. Architecture adapters may introduce additional simple names but cannot change grammar or canonicalization.

## Roots and revisions

`model` is the only v1 root. Runtime observations, datasets, runs, and artifacts reference a component ID but do not masquerade as components by adding other roots.

An ID is unique only within a `ModelGraphIdentity`, which includes:

```json
{
  "model_revision": "immutable provider revision or content digest",
  "adapter": "qwen",
  "adapter_version": "1",
  "graph_schema_version": "1"
}
```

Persisted records carry both the graph identity and component ID. A mutable local path alone is not a model revision.

## Component sets

A mutation can target a coupled set of components. A component set is a separate value and is never encoded by comma-joining paths.

1. Validate every member against one graph revision.
2. Remove exact duplicates.
3. Sort by canonical component-ID bytes.
4. Serialize the graph identity and ordered member array as canonical JSON.
5. Compute SHA-256 over the UTF-8 serialization.

The stable set identity is `component-set:v1:<lowercase-sha256>`. The persisted set record retains the ordered members and digest; a bare digest is insufficient provenance. Empty sets are invalid. Set identity is order-independent, while mutation execution order is stored separately in the mutation plan.

## Post-surgery remapping

A structural mutation emits a versioned `IdentityMap` associated with its input and output graph identities. Each affected input ID has exactly one disposition:

```json
{
  "source": "model.layers.3.mlp.up_proj.channel.7",
  "status": "removed",
  "targets": []
}
```

```json
{
  "source": "model.layers.4",
  "status": "renamed",
  "targets": ["model.layers.3"]
}
```

Allowed statuses are `retained`, `removed`, `renamed`, `split`, and `merged`. `retained` and `renamed` have one target, `removed` has none, `split` has two or more, and `merged` can share one target with other source records. Mappings compose only when the intermediate graph identities match exactly.

Historical feature and evaluation records continue to reference their original graph revision and IDs. They are not rewritten after surgery. A lookup using an old ID against a new graph must supply the identity map or fail.

## JSON representation

An individual component ID serializes as its canonical string. APIs can wrap it with graph provenance:

```json
{
  "graph": {
    "model_revision": "sha256:...",
    "adapter": "llama",
    "adapter_version": "1",
    "graph_schema_version": "1"
  },
  "component_id": "model.layers.17.self_attn.head.4"
}
```

Ordering is bytewise ordering of canonical UTF-8 text. Equality and hashing operate on the typed canonical segments, producing the same result as canonical text equality.

## Invalid and ambiguous inputs

Parsers reject, with the failing segment and reason:

- empty input, empty segments, leading/trailing dots, or a root other than `model`;
- negative indices, signs, whitespace, or leading-zero indices such as `01`;
- raw provider names containing percent characters, slashes, brackets, commas, or whitespace; a raw dot is parsed as a path boundary and therefore cannot represent a dot inside one provider name;
- an empty escaped name, incomplete/invalid/lowercase percent escapes, invalid UTF-8, NUL, or controls;
- escaped names that have a valid simple representation, such as `~head`;
- percent-encoding an unreserved byte, such as `~with%2Ddash`;
- selectors, globs, ranges, or component-set syntax passed where one component ID is required.

Selection expressions such as “all heads in layer 3” belong to a separate query API. They resolve to explicit component IDs before an experiment or mutation receives a stable identity.

## Conformance vectors

Machine-readable valid and invalid examples live in `tests/fixtures/component_id_conformance.json`. The component-ID implementation issue must consume these vectors directly and add property tests for canonical round trips.

