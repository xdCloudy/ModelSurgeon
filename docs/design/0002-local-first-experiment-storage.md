# ADR 0002: Local-first experiment storage

Status: Accepted

Use SQLite for metadata/state, Parquet for large tables and content-addressed filesystem artifacts. Distributed services are outside v1.0. This supports offline consumer hardware, atomic resumability and portable experiment bundles.

