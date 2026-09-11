# Training runtime architecture

Sequifier has one weight-owning model type: `ComposableTransformerNetwork`.
The network owns the shared backbone and named ingestion/decoder interfaces; it
does not own datasets, objectives, optimizers, metrics, checkpoints, exports,
random state, or run lifecycle state.

`RunBuilder` is the composition root for training. It builds a `TrainingRun`
containing the network and its callable distributed/compiled view, a
`DatasetRuntimeRegistry`, `OptimizationRuntime`, `RunState`, a distributed
strategy, random and loader-state services, integrations, evaluation, metrics,
export, and checkpoint services. `TrainingEngine.run()` only coordinates those
services while traversing configured phases and sources.

## Artifact contracts

Portable `.pt` model artifacts use `artifact_type=sequifier_model` and contain a
`ModelExecutionConfig`, canonical model state, and trace/provenance metadata.
Their state keys are limited to:

```text
backbone.*
interfaces.<interface-name>.*
```

Exact run checkpoints use `artifact_type=sequifier_run_checkpoint`. They embed
the same portable model payload plus optimizer/scheduler/scaler state, run
state, per-rank random state, loader state, integration state, and the resolved
training configuration. Only the current formats are accepted; historical
checkpoint layouts are not migrated at load time.

Sibling packages should import model contracts from `sequifier.api`. Update-aware
training integrations should import runtime primitives from
`sequifier.training_api`.

## Resume ordering

Restore is staged: load and validate the checkpoint, construct and prepare the
network, restore model weights, build and restore optimization, construct data
runtimes and restore loader/integration state, compile and warm up, then restore
the rank-local random state. This keeps setup-time random consumption from
changing the first resumed batch or update.

## Named depth execution

`TransformerEncoderArchitectureConfig` carries capacity-independent encoder
settings. `TransformerEncoderStack` owns the shared layer loop, sharing, final
normalization, and optional captures. `TransformerBackbone` defers layer
construction until after its temporal position modules, preserving the flat
registration and constructor order. Its existing state keys and sorted
architecture fingerprint fields remain intact. Depth encoders are owned by
interfaces and never enter the backbone repository.

`ExecutionSchema` defines ordered feature/mask descriptors and sequence-major
outputs for warm-up, portable artifacts, export, and runtime validation. Host
validation precedes compiled training execution. Tensor `where` sanitization
precedes deep feature arithmetic in eager, compiled, and exported forwards.
Unique layers in all temporal and depth stacks are compiled once with their
shared aliases restored. Warm-up preserves the run RNG, including on failure.

New portable model artifacts use format version 2; version 1 flat models remain
readable. Tensor payload versions and stored-window versions remain separate.
Run checkpoints record initialization-policy and seed-derivation versions in
addition to resolved branch policies and the existing optimizer/scaler/RNG state.
Inactive legacy layout/payload fields are normalized in resume comparisons.

## Export preflight and publication

Before depth training with ONNX enabled, rank zero runs capability preflight in
a disposable subprocess; errors are communicated through the existing rank
coordination. Successful graph/session/output validation is cached under
`.sequifier/export-preflight`. Cache keys include the complete execution graph,
depth and temporal architectures, composite order and widths, selected embedding
sites, decoder/objective semantics, masks and capacities, dropout mode, FP32 and
attention-lowering policy, exporter options, source-file hashes, exact package
versions, CPU/provider/session identity, and protocol/seed policy. Training
weights, paths, and epochs do not establish graph capability. Failed checks are
never cached. The child does not receive the live training network or optimizer.

Publication rebuilds a disposable network, preserves the parent's RNG, converts
parameters and buffers to FP32, and exports with opset 18 and the dynamo exporter.
An explicit attention lowering retains attention-weight dropout without altering
ordinary eager SDPA. Stochastic publication checks retained ONNX Dropout nodes
against the individually executed module scopes and call counts, including
shared-layer aliases; the graph stores that inventory. This structural check is
separate from statistical acceptance of each dropout site's effect.

Every artifact passes graph-contract checking, ONNX checking, CPU session
creation, and finite-output validation before publication. Evaluation export
also compares its FP32 reference with `rtol=2e-4`, `atol=2e-5`. External tensor
files receive unique names and publish before an atomic graph replacement, so
an existing graph stays usable if publication fails. Preserve its sidecars when
moving an ONNX artifact. Old sidecars from replaced graphs may remain on disk.

The implemented exporter API floor is PyTorch 2.6, ONNX 1.17, ONNXScript 0.5.4,
and ONNX Runtime 1.20; older installations receive a capability error. This is
an API requirement, **not a demonstrated compatibility matrix**. The current
implementation handoff intentionally includes no executed acceptance evidence;
see [the handoff report](../plans/named-depth-layouts-handoff.md).
