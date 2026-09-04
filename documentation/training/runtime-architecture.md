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
