# Maintainer-controlled integrations

`sequifier.integration` is an experimental, deliberately narrow integration
surface for packages released in lockstep with Sequifier. It is not a public
plugin framework: there is no entry-point discovery, registry, marketplace, or
cross-version compatibility negotiation.

The stable architectural boundaries are:

- `build_transformer_network(...)` constructs a batch-first
  `TransformerNetwork` that owns tensor computation only.
- `TrainingSession` composes the run and `TrainingEngine` owns optimizer-facing
  lifecycle synchronization.
- `IntegrationSpec` lets every DDP/FSDP worker construct a trusted observer or
  the single optional controller from an explicit `module:factory` path.
- Typed lifecycle events distinguish accumulated/scaled gradients, unscaled
  gradients, clipped gradients, and completed parameter updates.
- Observers may implement `capture_request(BatchPrepared)` and
  `interventions(BatchPrepared)`; the resulting tensors are delivered on
  `ForwardCompleted` without changing the no-integration fast path.
- A controller can only return a validated `TrainingDirective`; it cannot
  mutate architecture or data-loading semantics through the engine.
- `ParameterCatalog` describes physical parameters and preserves logical aliases
  for shared transformer layers.
- `TransformerNetwork.trace(...)` captures logical execution sites and applies
  differentiable interventions. Requesting attention internals selects the eager
  analysis attention path; the ordinary path retains fused attention.
- `load_model_for_analysis(...)` loads PT exports and new run checkpoints without
  inference compilation and can preserve gradients.

Core training YAML remains unchanged. Programmatic callers pass integration
specifications to `run_training(...)`; non-distributed callers may instead pass
direct instances. Setting `semantic_optimizer_grouping=True` gives optimizer
groups stable `group_id` values for targeted directives. Sequifier never imports
proprietary packages automatically. Integration state is stored under
`integration_state` in new run checkpoints, alongside the complete resolved
`training_config` and explicit `training_state`.

Tracing, interventions, and higher-order gradients require eager execution.
FSDP training rejects integrations that request interventions, higher-order
gradients, or full parameters. Offline analysis should use a complete PT export.
