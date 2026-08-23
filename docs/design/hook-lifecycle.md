# Activation hook lifecycle

`ActivationHookManager` owns one ordered set of canonical component targets for one
context lifetime. Construction rejects duplicate and unknown targets. Entry registers
in canonical order and rolls back already-created handles if any later registration
fails. Re-entry while active is forbidden.

Exit removes every handle in reverse registration order and clears all captured
objects after normal completion, exceptions, and `BaseException` subclasses such as
interrupts. Outputs pass through an adapter-supplied detach function before capture;
framework integrations must use it to sever autograd/device ownership. Removal errors
are aggregated only after all handles and captures have been processed.
