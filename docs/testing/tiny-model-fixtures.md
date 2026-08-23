# Tiny model fixtures

Unit tests use `modelsurgeon.testing.tiny_transformer`, a pure-Python, HF-shaped
transformer double for Llama, Qwen, Mistral, and Gemma. It exposes deterministic
`named_modules`, `named_parameters`, and configuration metadata without importing
Torch, allocating tensor payloads, using a GPU, or making network requests. Parameter
values are sampled independently from a SHA-256 stream keyed by fixture seed, physical
name, and scalar index; no weight files are generated or committed. The fixture
fingerprint covers its schema, family, seed, config, modules, parameter names, and
shapes.

The default contract is two layers, hidden size 8, four query heads, two KV heads,
intermediate size 12, vocabulary size 32, and seed 1729. Tests may override the seed
or tied-embedding behavior explicitly. A changed default or generation algorithm is a
fixture schema change and must update affected expected fingerprints.

`tests/fixtures/tiny_hf_models_v1.json` separately records optional public Hugging
Face integration sources at exact 40-character revisions. The manifest is metadata
only and is never consulted by unit tests. Any integration test that uses it must be
explicitly network-enabled, request the pinned revision, and keep downloaded weights
outside the repository. Revision metadata was checked through the public Hugging Face
model API on 2026-08-23.

`tests/test_foundation_integration.py` is the golden foundation path. It substitutes
the offline Llama double at the real loader import boundary, requires CPU-safe loader
options and immutable revision provenance, performs family detection and discovery,
builds and validates the complete component graph, and invokes the public CLI twice
to assert byte-stable JSON records, component IDs, and counts. It never imports Torch
or contacts Hugging Face.
