# ADR 0001: Stable typed component identities

Status: Accepted

Component identities are immutable typed paths shared by discovery, features, mutations, storage and reports. Structural edits emit an explicit identity mapping. String parsing is strict and adapter aliases cannot silently change canonical IDs.

