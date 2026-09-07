# Consumer troubleshooting and support boundaries

Every workflow retains an explicit outcome. Use the outcome and the retained
message to decide whether to adjust a bounded input or stop.

| Outcome | Meaning | Safe next step |
| --- | --- | --- |
| `supported` | The requested plan or operation is covered by the selected contract. | Review the plan and approval points before running it. |
| `unknown` | A source revision, measurement, runtime, or platform fact is unresolved. | Pin or measure the missing fact; do not treat the plan as executable. |
| `unsupported` | The requested format, family, operation, or capability is outside the verified boundary. | Use the documented workflow/matrix or retain the negative result. |
| `failed` | A hard budget, validation, or safety check was violated. | Reduce the bounded workload or fix the input; never bypass the check. |

## Decision tree

1. Read the JSON record and save the exact command, revision, profile, and
   environment. Do not retry with a mutable revision or overwrite the source.
2. For `unknown`, resolve the named uncertainty first: supply an immutable
   revision, retain a hardware/runtime measurement, or install the documented
   optional dependency.
3. For `unsupported`, check the [compatibility matrix](../architecture-compatibility.md).
   A successful neighboring family, dtype, or quantization does not transfer
   support.
4. For `failed`, keep the record and reduce one budget at a time: sequence
   length/tokens, candidate count, batch size, or output disk envelope. Resume
   only when the source, plan identity, and output identity are unchanged.
5. For an interrupted write, inspect the staging manifest before removing the
   failed staging directory. A published destination is never partially
   replaced.

Common safe fixes are `--device-map cpu`, a shorter sequence length, a smaller
calibration sample, or the `fast` profile. Do not add `--trust-remote-code`,
use a moving branch, paste credentials into a config, or use the source path as
the output path to make a consumer walkthrough pass.
