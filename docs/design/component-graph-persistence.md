# Component graph persistence

Component graphs persist as canonical UTF-8 JSON artifacts containing an artifact schema version, required adapter/model provenance, and the independently versioned graph record. Canonical output uses sorted object keys, compact separators, stable record ordering, and no framework objects.

`GraphProvenance` records the adapter name, adapter version, and immutable model revision. `dump_component_graph` canonicalizes node, edge, and constraint ordering before serialization. `load_component_graph` strictly validates every object key, primitive attribute, component ID, enum, record order, reference, and graph/artifact/edge/constraint version.

Unknown versions and non-canonical records fail closed with `GraphSerializationError`; loaders never guess a migration. A future version must add an explicit migration path and preserve the original artifact for provenance.
