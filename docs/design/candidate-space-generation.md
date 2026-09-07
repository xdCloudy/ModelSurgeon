# Bounded multi-axis candidate spaces

The candidate-space contract represents a complete architecture choice as
canonical axis assignments plus a hardware placement profile. Each axis domain
contains analytic parameter, storage, and source-distance costs. Global and
profile-specific ceilings are applied before emission, and divisibility rules
reject illegal head relationships without constructing a model.

Generation walks the Cartesian product lazily, retaining only the configured
deterministic seed-ranked ceiling. It never materializes the full product.
Canonical sorting makes input ordering irrelevant; candidate identities include
the base state, all axes, and hardware profile. A page cursor resumes the same
retained sequence, and each page records rejection counts by explicit reason.

The generator is a state-space boundary, not an evaluator: emitted descriptors
carry the complete legal static state needed by the existing multi-axis
sequence compiler. Dynamic mutation plans, measured deployment objectives, and
candidate ranking remain separate stages. Real integrations must pass each
descriptor through the sequence compiler before mutation publication.

Fixtures cover CPU/GPU ceilings, head divisibility, deterministic ordering,
bounded retention, page resumes, and invalid configuration rejection.
