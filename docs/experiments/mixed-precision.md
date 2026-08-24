# Mixed-precision experiment policy

ModelSurgeon selects experiment precision explicitly before execution. A precision decision records the requested mode, device capability source, execution mode, compute dtype, accumulation dtype, and any fallback reason. Metrics can then be bound to that decision so dtype changes cannot occur silently.

## Supported policy dtypes

The v1 policy names three compute/accumulation dtypes:

- `float32`
- `float16`
- `bfloat16`

Requests may select one dtype directly or request `autocast`. Autocast preference defaults to BF16 then FP16; FP32 is not treated as an autocast target.

## Conservative hardware capability resolution

`precision_capabilities_from_hardware()` derives only hardware support that can be justified from the recorded inventory:

- CPU advertises direct FP32 and FP32 accumulation only. Low-precision CPU support is not guessed from the processor name.
- CUDA requires `cuda.available=true`.
- FP16 is advertised only when every recorded CUDA device has a parseable compute capability of at least 5.3.
- BF16 is advertised only when every recorded CUDA device has a parseable compute capability of at least 8.0.
- unknown or missing CUDA compute capability advertises no low-precision dtype.
- v1 conservatively records FP32 accumulation.

This is a hardware-level resolver. A backend with stricter runtime/library limitations should construct a narrower `PrecisionCapabilities` record with its own `source` before selection rather than pretending the framework supports the full hardware set.

## Failure and fallback

Unsupported compute or accumulation requests fail closed by default. Setting `allow_fallback=true` permits an explicit safe fallback to direct FP32 compute and/or FP32 accumulation. Every fallback is retained in `PrecisionDecision.fallback_reason`; there is no silent dtype substitution.

Autocast also fails when none of the requested preference dtypes are supported unless fallback is explicitly allowed, in which case the decision records a direct FP32 fallback.

## Metric precision binding

`bind_metric_precision()` compares an observed `PrecisionExecutionContext` against the selected decision. Compute dtype, accumulation dtype, and autocast state must all match exactly. A mismatch raises `PrecisionPolicyError`; only a matching metric receives a `MetricPrecisionRecord` linked to the stable `precision_...` decision ID.

This keeps the execution policy and measured metric provenance separate while making an accidental dtype change observable and rejectable.
