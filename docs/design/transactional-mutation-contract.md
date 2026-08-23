# Transactional mutation contract

Mutation requests are immutable, canonically ordered, framework-neutral records.
Their SHA-256 identity derives only from the schema version, kind, canonical
component targets, and primitive parameters. A plan adds the complete affected
component set, sorted preconditions, and signed parameter, FLOP, memory, and
storage deltas.

Implementations report compatibility before planning and expose explicit apply
and rollback operations. Apply is permitted only through a prepared transaction
that declares ownership of the mutable inputs. This keeps source objects
immutable by default and gives format-specific transaction engines a stable
interface without exposing framework or storage objects in serialized plans.
