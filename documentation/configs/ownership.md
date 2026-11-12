# Configuration ownership

Every authored setting has one owner. Configuration files use the ownership
structure below directly.

| Owner | Purpose |
| --- | --- |
| `run` | Identity, execution resources, reproducibility, recovery, logging, outputs and export. |
| `model.execution` | Durable objective/window/attention facts required by the artifact. |
| `model.interfaces.<name>.input_contract` | Metadata and frozen normalization assertions. |
| `dataset_training.<name>` | Named interface reference, source parts and split policy. |
| `policy_bindings.<name>` | Actions, task units, return interpretation and action behavior. |
| `training_plan.phases[]` | Learning, trainability, budgets, optimization, evaluation, stopping and checkpoint snapshots. |
| `training_plan` | Ordered phase transitions. |

Preprocessing and inference share `run.project_root`, `run.execution` and
`run.reproducibility`. Their command-specific source/output and operation fields
remain on the command. Inference batching is `run.execution.batch_size`;
preprocessing workers are `run.execution.data_loading.num_workers`. Unsupported
execution controls fail validation. Component initialization and hash seeds stay
on their components; evaluation seeds define a benchmark and stay in evaluation.
Unsupported fields must be omitted even when their values match the defaults.
Inference does not accept `run.execution.data_loading`, `distribution`,
`precision`, `compilation`, or `max_concat_tensors`. Preprocessing accepts
only `run.execution.data_loading.num_workers` within execution, and only `seed`
within reproducibility. Sequence import accepts `run.project_root` but no
execution or reproducibility settings.

## Run settings

```yaml
run:
  project_root: .
  model_name: example
  execution:
    device: cpu
    distribution: {strategy: local}
    precision: {float32_matmul_precision: highest}
    compilation: {torch_compile: none}
    data_loading: {num_workers: 0, load_full_data_to_ram: true}
  reproducibility: {seed: 1010, enforce_determinism: false}
  recovery:
    resume: {policy: never}
    latest_checkpoint_interval_minutes: 10
  logging: {log_interval_batches: 10}
  outputs: {model_path: models/example/model.pt}
  export: {generative_model: true, onnx: true, pt: false, batch_size: 1}
```

Distribution is one coherent choice: `local`, `ddp` or `fsdp`. FSDP additionally
requires an explicit `cpu_offload`. Current policy execution is local, without
custom precision, compilation, concatenation limits, batch logging, or worker/lazy-loader/RAM-budget settings.
Policy-containing plans export the portable PT actor; graph, embedding, dropout and
batched exports are rejected. Runs default to PT export. Supervised runs can request graph exports explicitly. Defaults do not
request those capabilities. Random run seeds resolve once; derived phase streams
and resolved phase declarations are recorded in native recovery/provenance.

For policy-containing plans, omit `run.execution.distribution`, `precision`,
`compilation`, and `max_concat_tensors`, as well as `run.logging.log_interval_batches`.
Data loading may configure `episode_cache_size`, but not `num_workers`,
`load_full_data_to_ram`, or `max_ram_gb`. Omit the fixed export controls
`onnx`, `pt`, `generative_model`, `embedding_model`, `with_dropout`, and
`batch_size`; the PT actor export is selected automatically.

## Dataset and model contracts

Prepared parts own `read_format`; sequence parts own `format`. Raw datasets own
`split.validation_fraction`. Combining raw sources into one automatic, leakage-safe
partition requires equal fractions; use explicit `evaluation.sources` when the
partitions differ. Prepared split metadata remains authoritative. Fitting uses
training partitions only.

`model.execution.context_length` is the selected temporal view;
`model.backbone.architecture.max_context_length` is capacity. Frozen input contracts
and artifact assertions must agree. Phases cannot redefine immutable window,
attention or codec facts. `learning.bert_spec` and `learning.next_occurrence_config`
configure objective behavior for each supervised phase. Every next-occurrence
phase declares its trigger column and values; these define training and evaluation
targets and are not part of the portable execution contract.

## Phase settings

```yaml
training_plan:
  phases:
  - name: fit
    algorithm: supervised
    epochs: 10
    mode: sequential
    sources: [{source: events}]
    learning:
      datasets:
        events:
          criterion: {event: CrossEntropyLoss}
          loss_weights: {event: 1.0}
    trainability:
      roles:
        model:
          freeze:
          - {component: backbone, semantic_groups: [attention.qkv]}
    evaluation:
      sources: [{source: events}]
      monitor: {source: events, metric: loss, direction: min}
      cadence: {epochs: 1}
      batch_size: 64
      before_phase: false
      on_snapshot: false
    model_selection: {handoff: best}
    stopping: {patience: 3, min_delta: 0.001, on_trigger: end_phase}
    checkpoints:
      cadence: {epochs: 2}
      snapshot_interval_minutes: 30
```

Declare phase learning losses for all training and evaluation datasets. Source
scheduling still uses supervised `selection: round_robin | weighted_random`;
`model_selection` is distinct. Curriculum is `curriculum: {column: name}` on a
supervised phase. Sampling stride is `learning.window_stride`.

All algorithms share these owners. Supervised cadence uses `{epochs: N}`;
policy cadence uses `{iterations: N}`. A policy iteration is a complete algorithm
transaction, possibly containing multiple optimizer steps. Final progress always
evaluates when enabled and checkpoints. Initial evaluation and snapshot evaluation
can select best weights but do not consume stopping patience. Patience counts
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
Unknown or empty selections fail preflight.

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
