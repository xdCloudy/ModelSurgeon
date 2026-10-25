# PEFT LoRA integration record

## Acquisition matrix entry

| Field | Assessment |
| --- | --- |
| Source | Hugging Face PEFT, [pinned revision `7d6315ae910bf62b6044a7ca917bccf5881f5e43`](https://github.com/huggingface/peft/tree/7d6315ae910bf62b6044a7ca917bccf5881f5e43). Integration points reviewed: [`get_peft_model`](https://github.com/huggingface/peft/blob/7d6315ae910bf62b6044a7ca917bccf5881f5e43/src/peft/mapping_func.py) and [LoRA layer implementation](https://github.com/huggingface/peft/blob/7d6315ae910bf62b6044a7ca917bccf5881f5e43/src/peft/tuners/lora/layer.py). |
| Component | LoRA module injection, trainable adapter parameters, and safe merge. |
| Licence | Apache-2.0, verified from the [upstream licence at the pinned revision](https://github.com/huggingface/peft/blob/7d6315ae910bf62b6044a7ca917bccf5881f5e43/LICENSE). Used as a locked optional dependency; upstream source is not copied or modified. |
| ModelSurgeon equivalent | Previously, `lora_repair.py` implemented a private `Linear` wrapper, initialization, forward pass, and weight merge. It only supported exact `torch.nn.Linear` targets. |
| Technical advantage | PEFT supplies maintained adapter injection and `safe_merge`; ModelSurgeon retains exact target preflight and parameter caps, selects adapter checkpoints on disjoint held-out examples, records data digests, and owns output contracts, telemetry, and rollback. |
| Hardware requirements | Same base model and activation requirements as existing LoRA repair, plus rank-sized adapter and optimizer state. This change does not claim lower peak memory or faster optimization. |
| Supported architectures | No model family is certified by this integration. Runtime support remains limited to caller-selected exact `torch.nn.Linear` paths and models accepting the supplied repair-example keyword inputs. |
| Integration approach | **INTEGRATE** PEFT `0.21.2` through the `hf` optional extra (`peft>=0.21.2,<0.22`); lock resolves `0.21.2`. No copied upstream implementation. |
| Expected benefit | Reduce custom LoRA algorithm maintenance while retaining bounded behavior and choosing a checkpoint by held-out loss. Success criteria: unchanged target count and state export, deterministic adapter digest, selected-step evidence, merged reloadable model, and rollback after failed repair. Real-model recovery after surgery or speed improvement is not yet claimed. |
| Risk | PEFT target-name matching may wrap additional modules if paths share suffixes; exact post-injection trainable-parameter count check aborts and restores the original modules. Upstream API changes are constrained by the minor-version range and lock. |
| Validation | Tiny physical PyTorch tests cover separate adapters, deterministic digest, unchanged base weights, safe merge, budget rejection, rollback, exact overlap refusal, and best-step restoration. A repeated RTX 3060 run on revision-pinned SmolLM2-135M verifies held-out selection mechanics, state export, telemetry, and base-weight immutability; it is explicitly not a post-surgery recovery claim. Full suite and CI results are recorded with the PR. |
| Priority | High: replace duplicated foundational LoRA mechanics with a compatible established library while preserving ModelSurgeon authority and constraints. |

## Integration boundary

`run_bounded_lora_repair` validates each requested target as an exact `torch.nn.Linear` and computes the expected adapter parameter total before PEFT mutates the model. It freezes base parameters, asks PEFT to inject only the declared targets, then compares PEFT's actual trainable parameter count to the preflight count. Training, deterministic result evidence, resource telemetry, held-out selection, and publication remain ModelSurgeon responsibilities. Validation input IDs must be disjoint from training IDs. The adapter snapshot with minimum per-example mean validation loss is restored and selected-step plus train/validation digests are recorded. The first-party optimizer excludes its default held-out calibration chunk from training; an external validation corpus is also supported. For merged output PEFT's `merge_and_unload(safe_merge=True)` is used; for separate output the adapter remains available through `lora_adapter_state_dict`. Any exception restores the original module objects and `requires_grad` values.
