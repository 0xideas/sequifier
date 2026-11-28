# Configuration ownership

Training defines named resources and executes the phase names in `plan`.
A shared fragment can supply phases to several entry files. Each entry file selects
its own order and uses each selected phase once. Search parameters and CLI patches address a phase by name.

| Owner | Settings |
| --- | --- |
| `run` | Identity, resources, seed, recovery, logging, outputs and artifact publication |
| `model` | Backbone, named interfaces, input contracts and durable prediction geometry |
| `datasets.<name>` | Physical sources, optional partition and named views |
| `policy_bindings.<interface>` | Actions, feedback, units and return interpretation |
| `environments.<name>` | Environment factory, configuration and units |
| `phases.<name>` | Algorithm, inputs, objective, budget, collection, optimization and lifecycle |
| `plan` | Phase names in execution order |

## Datasets and views

Phases address `dataset.view`. Each view selects a set of source parts and,
where applicable, a split. The same reference selects the same rows in
training, evaluation and metadata fitting.

* A `prepared` source points to a preprocessing metadata file. With no explicit
  views, the dataset exposes the named splits common to its sources.
* An `indexed_sequence` source points to one recorded split manifest. Each
  source is available as a view under its own name.
* A `sequence` source supplies episode data. A `preferences` source supplies
  a preference manifest whose spans refer to episode sources or recorded splits.
  Separate physical sources can serve as presplit views.
* Set `partition.validation_fraction` on a sequence or preferences dataset to
  generate `train` and `validation` views. Sequifier partitions the entire
  dataset using the run seed and keeps connected episodes, groups and duplicate
  content together. Restricting a view to particular parts preserves partition
  membership. Offline phases select both training and evaluation views explicitly.

This dataset combines both feeds for training and evaluates on recent customers.

```yaml
datasets:
  customers:
    sources:
      historical: {type: prepared, metadata_config_path: configs/historical.json}
      recent: {type: prepared, metadata_config_path: configs/recent.json}
    views:
      fit: {split: train}
      recent_holdout: {parts: [recent], split: validation}
```

## Model and phase settings

`model.execution.context_length` selects the temporal view. Backbone capacity
defaults to that length plus any static prefix. A phase selects its supervised
objective in `learning.objective`, including masking settings for BERT or the
trigger for next-occurrence training. The shared model has one durable
prediction view, inherited from an initialization artifact, inferred from the final supervised phase, or declared in
`model.execution.prediction_view`. Causal, final-value and next-occurrence objectives can follow one another because they share forward prediction geometry. A forward phase defaults to causal loss. BERT phases declare their masking settings. Each supervised dataset declares one
`learning.datasets.<dataset>.losses.<target>` entry with its criterion and
optional weight and class weights. A policy declares each lagged input under
`feedback.<input>` with its action and optional episode-start value.

| Algorithm | Phase budget | Input or collection |
| --- | --- | --- |
| Supervised, policy imitation | `budget.epochs` | `sources` |
| IQL, DPO | `budget.iterations` | Episode or preference `sources` |
| PPO | `budget.env_steps` | `collection.environment`, `collection.env_steps_per_iteration` |
| GRPO | `budget.groups`, optional `budget.max_env_steps` | `collection.environment`, `collection.group_size`, `collection.max_steps` |

Online phases reference named environments separately in collection and evaluation.
Evaluation owns its episode seeds, monitor and cadence.

## Run settings

`run.execution.device`, `run.execution.distribution`, and
`run.execution.precision` apply throughout training. Distribution selects
`local`, `ddp` or `fsdp`. FSDP also requires `cpu_offload`.
Supervised phases consume `run.execution.supervised` for compilation, data
loading and concatenation limits. See the [training precision reference](train.md#training-precision).
Policy phases select `none` (the default), `inner`, or `outer` with
`run.execution.policy.compilation.torch_compile`.
Offline policy phases consume `run.execution.offline.episode_cache_size`.
Batch and iteration logging have separate intervals. A random run seed resolves
once and is recorded for recovery. See [training resource controls](../training/resource-controls.md).

By default training publishes a portable prediction model. A supervised run
can select multiple artifact formats and give each graph its own options.

```yaml
run:
  model_name: demand
  export:
    prediction:
      pt: {}
      onnx: {batch_size: 8}
    embedding:
      layers: [backbone.final_norm]
      onnx: {batch_size: 1, with_dropout: true}
```

Policy plans publish the portable prediction actor. Inference has its own
execution settings for device and batch size. Preprocessing workers are set
under `run.execution.data_loading.num_workers`. Every command accepts
`run.project_root`.

## Phase settings

```yaml
phases:
  fit:
    algorithm: supervised
    mode: sequential
    sources:
    - source: events.train
    learning:
      datasets:
        events:
          losses:
            event:
              criterion: CrossEntropyLoss
              weight: 1.0
    trainability:
      roles:
        model:
          freeze:
          - component: backbone
            semantic_groups:
            - attention.qkv
    evaluation:
      sources:
      - source: events.validation
      monitor:
        source: events.validation
        metric: loss
        direction: min
      cadence:
        epochs: 1
      batch_size: 64
      before_phase: false
      on_snapshot: false
    model_selection:
      handoff: best
    stopping:
      patience: 3
      min_delta: 0.001
      on_trigger: end_phase
    checkpoints:
      cadence:
        epochs: 2
      snapshot_interval_minutes: 30
    budget:
      epochs: 10
plan:
- fit
```

Declare phase learning losses for all training and evaluation datasets. Source
scheduling still uses supervised `selection: round_robin | weighted_random`;
`model_selection` is distinct. Curriculum is `curriculum: {column: name}` on a
supervised phase. Sampling stride is `learning.window_stride`.

All algorithms share these owners. Supervised cadence uses `{epochs: N}`;
policy cadence uses `{iterations: N}`. A policy iteration is a complete algorithm
transaction, possibly containing multiple optimizer steps. Final progress always
evaluates when enabled and checkpoints. Supervised initial and snapshot evaluations
are diagnostic: they do not select best weights or consume stopping patience.
Policy initial and snapshot evaluations can select best weights without consuming
stopping patience. Patience counts
scheduled finite evaluation observations; `on_trigger` explicitly ends the phase
or the entire plan. Stopping state is persisted so resume cannot count twice.

Recovery refreshes rolling resumable state at safe boundaries. Phase snapshots
are retained checkpoints. Best-weight handoff is distinct from latest recovery;
it does not restore older optimizer/environment state. Snapshot time intervals
request the next safe committed boundary. Policy batch snapshots are rejected;
use transaction cadence or time instead.

## Parameter selection

Freezing and optimizer routing use the same catalog selectors: `component`,
`semantic_groups` (glob patterns), `parameter_kind`, `ndim`, named `interface`,
`branch_path`, or a `module` subtree. Conditions in a selector are conjunctive;
selector lists form a union. `exclude_modules` and `exclude_semantic_groups`
express explicit subtractions, for nested component policies.
Unknown or empty selections fail preflight. Every `semantic_groups` pattern must
match at least one eligible parameter within the selector's structural scope;
every `exclude_semantic_groups` pattern must match the positively selected scope
before exclusions are applied. Patterns are case-sensitive and are never
corrected or ignored. A valid alternative does not hide an unmatched pattern.
Overlapping and repeated matching patterns remain valid union selections.

Each role accepts either `freeze` or `freeze_except`. Empty `freeze` masks nothing;
empty `freeze_except` masks everything and fails if an update has no trainable
parameters. Structural constraints remain: fixed Gaussian scales and frozen image
encoders cannot be enabled by a phase mask. Gradient freezing leaves model mode
semantics intact.

Supervised `trainability.by_dataset.<name>.roles.model` uses exactly the same
policy shape. A supplied role policy replaces the phase policy for that dataset;
absence inherits it. Only training datasets can have update overrides. Policy
algorithms reject source-dependent masks, and support their own declared role set.
Frozen gradients are removed before AMP handling and stepping, so momentum and
weight decay do not update them. Masks are recalculated at phase boundaries.

## Recovery

Run checkpoints contain the current resolved settings, phase state and input
identities. Recovery requires these to agree with the requested run. Portable
model artifacts contain only the facts needed to reconstruct model execution.
