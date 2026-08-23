# Mutation target resolution

Target resolution converts a canonical mutation request into the complete set of
components that must change together. It first validates the component graph and
rejects absent targets or incomplete constraint topology. It then computes a
fixed-point closure across overlapping mutation constraints and explicit coupled
edges, so indirect coupling cannot be omitted from a plan.

The result is canonically ordered and records whether each component was directly
requested plus every constraint or coupled edge that caused its inclusion. It can
be converted directly into the framework-neutral transactional mutation plan.
No model or tensor is touched during resolution.
