# Configuration naming reference

Sequifier accepts one canonical spelling and location for each authored setting.
See [configuration ownership](ownership.md) for the supported locations.

## Named structures and fixed shapes

Use `model.interfaces.<name>`, `dataset_training.<name>.parts.<name>`,
`policy_bindings.<interface>`, and `training_plan.phases` for every run size.
`default` is an ordinary name, not an implicit authoring syntax. Omitted unique
references can still be inferred. Direct `training_plan.epochs`, singular
`model.interface`, `dataset`, dataset `part`, and training `policy_binding` are
rejected. Sequence import still has one singular `policy_binding`.

Supervised evaluation is an explicit `{sources, monitor?}` declaration or `null`;
boolean shorthand is rejected. Optimizers and schedulers use `{name, arguments}`.
Initialization methods use `{method, ...}`, including parameterless methods.

`model_paths` is always a non-empty list. `image_paths`,
`preconditioning_image_paths`, and preprocessing `preconditioning_image.paths`
are view-to-path mappings even for a single view. `additional_config_paths` is
a list or `null`. Preprocessing `window_lengths` supplies one width per split.
Generated metadata retains its selected-split `window_length`.

## Dimensions, counts, and selection

| Setting |
| --- |
| `model_dim` |
| `feed_forward.hidden_dim` |
| `num_heads` / `num_kv_heads` |
| `total_embedding_dim` |
| `position_encoding_scope` |
| `post_conv_normalization` |
| `frame_pooling` / `image_frame_pooling` |
| `pooling` |
| `support_length` |
| `decoder_support_length` |
| `max_depth_length` |
| `max_concat_tensors` |
| `run.execution.data_loading.num_workers` |
| `selection_weight` |

`model_dim`, `hidden_dim`, `output_dim`, `cell_dim`, `hidden_dims`, and
`feature_embedding_dims` describe different representation widths.
`total_embedding_dim` allocates aggregate static feature width. `context_length`
is temporal model input length; `max_context_length` is backbone capacity;
`max_depth_length` is child-collection capacity. `support_length` counts temporal
representations consumed by a decoder; `prediction_length` counts decoded positions.

Source `selection_weight` changes ordering and burst frequency, not the total
amount consumed in an epoch. Class weights and loss weights retain their separate
loss-scaling meanings.

## Training clocks and optimization

A batch is one forward/backward microbatch. An optimizer step applies gradients
for a role. An iteration is a complete policy transaction: imitation, IQL, and
DPO perform one optimizer step per role; PPO and GRPO may perform several.
An epoch is one pass through the applicable data. Environment steps count
transitions; generation steps count temporal autoregressive output steps.

| Setting |
| --- |
| `accumulation_batches` |
| `batch_size` |
| `iterations` |
| `groups_per_iteration` |
| `rollout_env_steps` |
| `evaluation.cadence.epochs` |
| `evaluation.cadence.iterations` |
| `checkpoints.cadence.epochs` |
| `checkpoints.cadence.iterations` |
| `log_interval_batches` |
| `stopping.patience` (evaluation observations) |
| `scheduler_step_on: optimizer_step` |
| `gradient_clip_norm` |
| `value_clip_delta` |
| `old_policy_target_kl` |
| `reference_kl_coefficient` |
| `max_advantage_weight` |

Keep `epochs`, `optimization_epochs`, `total_env_steps`, `max_env_steps`,
`total_groups`, `group_size`, `max_steps`, `max_episode_steps`, and
`generation_steps`: their enclosing contracts identify different budgets.
`batch_size` always describes a microbatch; its elements are examples,
transitions, preference pairs, or trajectories depending on the algorithm.

Keep optimizer roles `model`, `actor`, `value`, and `critic`. Third-party
constructor names and arguments remain unchanged under `arguments`.
`learning_rate` is Sequifier-owned; optimizer constructor `lr` is forbidden.
Scheduler `total_steps` remains an assertion against the resolved scheduler budget.

## Freezing, checkpoints, and direction

Freezing now belongs to `phase.trainability.roles.<role>` for all algorithms;
source-dependent supervised masks use `phase.trainability.by_dataset`.
`freeze`/`freeze_except` contain catalog selector objects, shared with optimizer
routing. Explicit module selectors and semantic scopes share the same selection language.

Recovery is `run.recovery`; retained snapshots are `phase.checkpoints`;
evaluation triggers are `phase.evaluation.before_phase` and `.on_snapshot`.
Handoff is `phase.model_selection.handoff`; source scheduling retains `selection`.
See [ownership](ownership.md) for the complete structure and stopping units.

## Inference, sources, and seeds

Inference requires `task: forecast`, `task: policy`, or `task: embedding`.
`model_type` is removed. The artifact validates the requested task; it never
chooses the task implicitly. Policy requires exactly one entry in `model_paths`.
`autoregressive` becomes `autoregressive_generation`. Inference's
`preprocessing_data_path` metadata shortcut is removed; use `metadata_config_path`.

| Setting |
| --- |
| `model_paths` |
| `image_paths` |
| `preconditioning_image_paths` |
| `paths` |
| `model.interfaces.<name>.input_contract.metadata_config_path` |
| `manifest_path` |

Interface input contracts select metadata files. Per-part and inference
`metadata_config_path` still select one file. Source-local `project_root` is
removed; the run supplies it during resolution.

Run `run.reproducibility.seed` accepts an integer or `random`. Random seeds resolve once to an integer
recorded in the effective configuration. `sampler_seed` controls an independent
search stream. Initialization and hash seeds stay scoped integer declarations.
`enforce_determinism`, `action_selection`, and `seed` remain independent controls.

Keep `seed_window_index` and `seed_row_index`, model `previous_action_inputs` and
data `previous_action_columns`, scalar `actions` and structured `action_record`,
and model-owned versus fixed policy `scale`: these are meaningful distinctions.

## Search expressions

| Setting |
| --- |
| `sampler_seed` |
| `method: random` |
| `trial_config_directory` |
| `evaluation_script_path` |
| `evaluation_inference_config_path` |
| `$choices` |
| `$fixed` |
| `$variants` |
| `$range: {low, high, step?, log?, type?}` |

Primitive scalars are fixed replacements; ordinary mappings recursively patch.
`$fixed` replaces a complete mapping or list. `$choices` samples any complete
value. `$variants` patches with selected partial mappings and may have sampled
siblings. Integer list keys patch indexed entries. Bare lists are rejected.
Search operators reserve `$` and never depend on the base field's type.

`evaluation_metric_directions` uses `min` and `max`, matching monitor `direction`.
Training parameter paths are exactly the canonical training paths.

## Cardinality, initialization, and dtype values

| Setting |
| --- |
| `num_buckets` |
| `min_count` |
| `max_categories` |
| `column_data_types` |
| `xavier_uniform` |
| `xavier_normal` |
| `std` |

Numeric storage dtypes use canonical `Float32`, `Int64`, etc.; lowercase aliases
are rejected. Compute dtypes remain PyTorch-style `float32`, `bfloat16`, etc.
Standard methods such as `zeros`, `ones`, and `constant` remain separate methods,
even when some parameterizations are mathematically equivalent.

Probability types describe their unchanged mathematics:

| Previous type | Canonical type | Definition |
| --- | --- | --- |
| `normal_discretized_floor` | `clipped_rounded_normal_plus_one` | `max(round(Normal(mean, std)), 0) + 1` |
| `log_normal_discretized_floor` | `rounded_log_normal_plus_one` | `round(exp(Normal(mean, std))) + 1` |
| `poisson_floor` | `poisson_plus_one` | `Poisson(rate) + 1` |

## CLI

CLI flags use the YAML name with hyphens: preprocessing
`--preprocessing-data-path`, inference `--model-paths`, and
`--task forecast|policy|embedding`. All short aliases and abbreviated flags are
removed. `--randomize` becomes `--seed random`. Boolean configuration overrides
take explicit `true` or `false`, including `--enforce-determinism false`.
