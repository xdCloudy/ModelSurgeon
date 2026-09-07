# Hardware-conditioned surgeon predictions

`modelsurgeon.surgeon.hardware_conditioning` conditions quality, safety, and
utility predictions on an explicit runtime profile containing CPU threads,
VRAM, accelerator availability, offload fraction, quantization, runtime, and
feature schema. Unsupported profiles are retained as unsupported predictions;
missing held-out evidence remains unknown.

Samples separate pre-mutation state features from targets. A post-mutation
runtime target may be retained for audit, but it is never included in the
quality, safety, or utility design matrix. This prevents deployment targets
from leaking into a quality prediction.

The model is a deterministic ridge-linear predictor with a versioned feature
schema. The same candidate state can therefore receive different utility on
CPU-only, low-VRAM, and full-GPU profiles. The ablation compares a
hardware-conditioned model with a hardware-agnostic baseline on held-out
profiles and retains a `null_result` when context does not improve error.

