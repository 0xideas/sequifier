# Training resource controls

The I12 source audit found these missing effects before this change:

| Configuration | Missing behavior |
| --- | --- |
| `run.execution.data_loading.max_ram_gb` | No runtime consumer or data-loading budget. |
| `run.execution.precision.layer_autocast` | Ignored by local and DDP forward/validation; only FSDP applied mixed precision. |
| `run.execution.precision.float32_matmul_precision` | Set in the launcher, but lost in spawned training processes. |
| `run.reproducibility.enforce_determinism` | Requested deterministic algorithms in warning-only mode; false did not undo a previous true setting in the same process. |

Device, distribution, FSDP offload, compilation, layer parameter dtypes, worker
counts, concatenation limits and episode-cache sizes already have consumers on
their applicable routes. Policy-containing plans reject unsupported resource
controls. `load_full_data_to_ram` selects eager/lazy **folder** loading; single
prepared files are eager.

## Data-loading RAM

`max_ram_gb` is a positive finite budget in decimal gigabytes for prepared-data
loading in each training rank, including its loader workers. It covers all cached
training/validation loaders together, decoded data, view/order indices, static
tables, decoded-frame caches, chunk/conversion working space and prefetched
batches. It is not an operating-system limit on total process RSS: model,
optimizer, activation, interpreter and third-party allocator memory require
additional headroom. Multiple ranks each have their own budget.

Preflight uses decoded sizes rather than compressed file sizes and reserves
conversion/concatenation and worker/prefetch headroom. The largest applicable
phase batching/view requirements are considered before loading data. Actual
resident dataset storage is checked against the plan after construction, and
streamed shards are checked against their working allowance before gathering. An
over-budget plan raises `MemoryError` with required/available sizes and remedies;
it does not silently change an authored eager/lazy choice. Use lazy folders,
smaller shards, fewer workers, smaller batches or a larger budget as appropriate.

Lazy folders gather at most a microbatch at a time, including when view expansion
produces many logical examples from one stored chunk. Chunk loading remains
bounded by the shard size, so an individual shard must fit its working allowance.

## Policy run resource accounting

Policy run reports include a `resource_accounting` section. Policy phase reports
carry the cumulative snapshot in `provenance.resource_accounting`. The snapshots
show current and peak estimated bytes by category and the peak of their combined
demand at a single point in execution. A resumed invocation starts a fresh
accounting session.

Accounting covers prepared policy catalogs and partitions, decoded episode and
label caches, reconstructed scoring and update windows, rollout estimates and
archive working space, parent-side simulator request and response payloads, and
retained frozen-image features. Prepared partitions are shared through the last
phase that needs them. Each rollout is released after its update and any archive
or summary work, before evaluation begins.

`episode_cache_size` still counts episodes, `max_buffer_bytes` still uses the
algorithm's conservative payload estimates, and `image_feature_cache_bytes`
still applies to each distinct cache. A cache shared by the actor and reference
contributes once. The combined estimate is observational. Model and optimizer
storage, activations, allocator overhead, and simulator process internals are
outside its scope. The supervised data-loading budget continues to apply to
supervised loaders.

## Autocast and matrix precision

`layer_autocast: true` applies Torch autocast to local/DDP training forward and
loss calculation, validation and warm-up. Backward runs outside autocast.
Parameter storage dtypes remain governed by `layer_type_dtypes`. A configured
linear FP16/BF16 dtype also selects that autocast dtype; otherwise the device's
Torch default is used (CPU BF16, CUDA FP16). MPS uses BF16 and requires macOS 14
or newer; native MPS FP16 autocast is rejected because the native gradient-scaler
protocol supports CPU/CUDA. Unsupported device/dtype requests fail explicitly.
FSDP retains its BF16 mixed-precision policy by default.

FP16 autocast with FP32 optimizer parameters enables gradient scaling. Explicit
FP16 parameters retain the existing unscaled update protocol. The scaler is
persisted/restored with native recovery. FSDP uses its sharded mixed-precision
policy rather than a second enclosing autocast context.

`float32_matmul_precision` is applied inside each native training process before
model construction/compilation, including spawned ranks and direct builder use.
It is independent of autocast and parameter storage precision.

## Determinism

`enforce_determinism: true` enables deterministic algorithms with errors for
unsupported nondeterministic operations and configures CUDA workspace policy
before seeding. False disables Torch's deterministic-algorithm requirement,
including after an earlier deterministic run in the same process. Seeds and
intentional stochastic action selection remain independent controls.

No tests have been written or run for this change, as requested. Runtime/device
and memory-estimation validation remains pending.
