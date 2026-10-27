# ADR 0002: Local-first experiment storage

Status: Accepted

Use SQLite for metadata/state, Parquet for large tables and content-addressed filesystem artifacts. Distributed services are outside v1.0. This supports offline consumer hardware, atomic resumability and portable experiment bundles.

The Parquet store publishes partitions through an atomic visibility manifest. The manifest is limited to 64 MiB on both read and write so corrupted or unexpectedly large store metadata cannot trigger unbounded JSON loading; an over-limit manifest fails closed with `ParquetStoreError`.

