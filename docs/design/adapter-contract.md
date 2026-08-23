# Model Adapter Capability Contract

Status: Accepted for v0.1

## Boundary

Adapters isolate model formats and framework runtimes from the rest of ModelSurgeon. Hugging Face/PyTorch, safetensors, GGUF, and llama.cpp objects remain private to an adapter session. Discovery, persistence, experiments, and reports consume canonical component IDs and framework-neutral records.

The public contract is defined in `modelsurgeon.adapters.base`.

## Lifecycle

1. Construct a `ModelSource` with format, locator, and immutable revision or digest when available.
2. Select an adapter using `formats` and `can_open`.
3. Check the required `AdapterCapability` values before expensive work.
4. Open a session with explicit `OpenOptions`, including RAM/VRAM ceilings and remote-code trust.
5. Discover components and tensors, resolve canonical IDs, read bounded tensor bytes, or ask whether a mutation is supported.
6. Close the context-managed session after success, failure, or interruption.

Sessions are opaque: callers cannot obtain a `torch.nn.Module`, `Tensor`, GGUF reader, mmap handle, or llama.cpp context through the interface. Format adapters can offer private implementation helpers inside their own package.

## Capabilities

Capabilities are independent rather than inferred from an adapter name:

- loading;
- component discovery;
- tensor metadata and bounded reads;
- activation instrumentation;
- masking and physical mutation;
- checkpoint writing;
- native quantized surgery.

Callers use `require_capability` and receive `UnsupportedCapabilityError` before an operation starts. A GGUF inspection adapter can therefore advertise discovery and tensor reads without accidentally claiming physical mutation. Mutation compatibility is a second, target-specific decision represented by `MutationSupport`, including a required reason and primitive constraints.

## Persisted records

`AdapterIdentity`, `ModelSource`, `ComponentDescriptor`, `TensorDescriptor`, and `MutationSupport` serialize only strings, numbers, booleans, nulls, arrays, and maps of those values. They carry no live framework object. `TensorChunk` is explicitly ephemeral and exposes bounded bytes through `memoryview`; it has no persistence method.

Component descriptors use stable `ComponentId` values and primitive adapter attributes. Tensor descriptors identify logical ownership, physical name, shape, dtype, storage bytes, and quantization without materializing payloads.

## Safe failure

- Unsupported capabilities fail before model mutation or large allocation.
- Unknown formats and architectures are rejected rather than guessed.
- Resource and trust options are explicit at session open.
- Component resolution either returns the exact canonical descriptor or fails.
- Tensor byte ranges must be validated by concrete adapters before I/O.
- Mutation support decisions state why a target is supported or rejected.

Physical mutation, transactional writing, identity remapping, and graph construction have separate contracts built on this boundary.

