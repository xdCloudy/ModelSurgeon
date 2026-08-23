# Component dependency and coupling graph schema

Status: accepted v1 schema

The component graph persists only canonical IDs, semantic kinds, primitive attributes, edges, and mutation constraints. It cannot contain live PyTorch modules, tensors, GGUF readers, or other model objects. `ComponentGraph.build` sorts all records canonically and validates unique nodes, edge endpoints, constraint membership, and constraint references.

The graph, edge semantics, and constraint documents each carry an independent version. Version 1 defines these directed edge meanings:

| Edge | Source → target meaning |
|---|---|
| `parent` | child component → its canonical parent |
| `child` | parent component → one canonical child |
| `consumes` | consumer component → consumed component or tensor node |
| `produces` | producer component → produced component or tensor node |
| `coupled` | canonically ordered endpoints of a symmetric mutation relationship |
| `constrained` | component → related component, with a `constraint_id` attribute |

Coupled edges use a single record whose endpoints are in canonical order. A constrained edge references a versioned `MutationConstraint`, whose sorted members define the full affected set and whose primitive parameters describe the rule. Version 1 constraint kinds cover grouped mutation, hidden-size equality, head-set equality, shape equality, and explicitly named custom rules.

Changing the meaning, direction, required attributes, or validation behavior of an edge requires a new edge-semantics version. Changing constraint membership or parameter interpretation requires a new constraint schema version. Additive graph container changes require a new graph schema version.
