# ADR 0003: Progressive surgeon complexity

Status: Accepted

Advance from heuristics to linear/tree/MLP models and only then to structural neural architectures when held-out evidence shows a simpler model has saturated. Every added model class must justify accuracy, calibration, latency and memory cost.

