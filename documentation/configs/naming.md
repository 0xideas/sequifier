# Configuration naming reference

Sequifier accepts one canonical spelling and location for each authored setting.
Removed names produce replacement guidance; they are not compatibility aliases.
These are breaking configuration changes. Update authored files and any embedded
configuration used to reconstruct a model or resume a run before reusing them.
Settings not listed here retain their names, scopes, and defaults.

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

| Previous spelling | Canonical spelling |
| --- | --- |
| `dim_model` | `model_dim` |
| `feed_forward.dim` | `feed_forward.hidden_dim` |
| `n_heads` / `n_kv_heads` | `num_heads` / `num_kv_heads` |
| Static `embedding_size` | `total_embedding_dim` |
| `positional_encoding_scope` | `position_encoding_scope` |
| `post_conv_norm` | `post_conv_normalization` |
| `frame_pool` / `image_frame_pool` | `frame_pooling` / `image_frame_pooling` |
| Axis-pooling `mode` | `pooling` |
| Decoder `support` | `support_length` |
| Inference `decoder_support` | `decoder_support_length` |
| Depth-layout `context_length` | `max_depth_length` |
| `device_max_concat_length` | `max_concat_tensors` |
| Preprocessing `n_cores` | `num_workers` |
| Source `weight` | `selection_weight` |

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

| Previous spelling | Canonical spelling |
| --- | --- |
| `accumulation_steps` | `accumulation_batches` |
| PPO `minibatch_size` | `batch_size` |
| IQL/DPO `updates` | `iterations` |
| GRPO `groups_per_update` | `groups_per_iteration` |
| PPO `rollout_length` | `rollout_env_steps` |
| Supervised `evaluation_interval` | `evaluation_interval_epochs` |
| Policy `evaluation_interval` | `evaluation_interval_iterations` |
| Supervised `checkpoint_interval` | `checkpoint_interval_epochs` |
| Policy `checkpoint_interval` | `checkpoint_interval_iterations` |
| `log_interval` | `log_interval_batches` |
| `early_stopping_epochs` | `early_stopping_patience_epochs` |
| Supervised `scheduler_step_on: batch` | `scheduler_step_on: optimizer_step` |
| `gradient_clip` | `gradient_clip_norm` |
| PPO `value_clip` | `value_clip_delta` |
| PPO `target_kl` | `old_policy_target_kl` |
| GRPO `kl_coefficient` | `reference_kl_coefficient` |
| IQL `max_weight` | `max_advantage_weight` |

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

The dataset `freeze` container remains. Inside its component tree, `freeze`
becomes `freeze_groups` and `freezing_except` becomes `freeze_except_groups`.
Policy phase `freeze` becomes `freeze_modules`: its entries are actual module
paths, not semantic group selectors. Empty `freeze_groups` unfreezes a scope;
empty `freeze_except_groups` freezes it.

| Previous spelling | Canonical spelling |
| --- | --- |
| `save_interval_batches` | `snapshot_interval_batches` |
| `save_interval_minutes` | `snapshot_interval_minutes` |
| `save_latest_interval_minutes` | `latest_checkpoint_interval_minutes` |
| `save_interval_val_loss` | `evaluate_on_snapshot` |
| `calculate_validation_loss_on_initialization` | `evaluate_before_training` |
| Monitor `mode` | `direction: min` or `direction: max` |

Snapshots are retained checkpoints triggered by the global batch/time intervals.
Refreshing `latest` is distinct. `resume`, `initialize_from`, `optimizer_state`,
and `handoff` keep their separate lifecycle meanings.

## Inference, sources, and seeds

Inference requires `task: forecast`, `task: policy`, or `task: embedding`.
`model_type` is removed. The artifact validates the requested task; it never
chooses the task implicitly. Policy requires exactly one entry in `model_paths`.
`autoregressive` becomes `autoregressive_generation`. Inference's
`preprocessing_data_path` metadata shortcut is removed; use `metadata_config_path`.

| Previous spelling | Canonical spelling |
| --- | --- |
| Inference `model_path` | `model_paths` |
| `image_path` | `image_paths` |
| `preconditioning_image_path` | `preconditioning_image_paths` |
| Preconditioning image `path` | `paths` |
| Global interface `metadata_config_path` | `interface_metadata_paths` |
| Sequence/preference `manifest` | `manifest_path` |

`interface_metadata_paths` maps interface names to files. Per-part and inference
`metadata_config_path` still select one file. Source-local `project_root` is
removed; the run supplies it during resolution.

Run `seed` accepts an integer or `random`. Random seeds resolve once to an integer
recorded in the effective configuration. `sampler_seed` controls an independent
search stream. Initialization and hash seeds stay scoped integer declarations.
`enforce_determinism`, `action_selection`, and `seed` remain independent controls.

Keep `seed_window_index` and `seed_row_index`, model `previous_action_inputs` and
data `previous_action_columns`, scalar `actions` and structured `action_record`,
and model-owned versus fixed policy `scale`: these are meaningful distinctions.

## Search expressions

| Previous spelling | Canonical spelling |
| --- | --- |
| `global_seed` | `sampler_seed` |
| `method: sample` | `method: random` |
| `model_config_write_path` | `trial_config_directory` |
| `evaluation_script` | `evaluation_script_path` |
| `evaluation_inference_config` | `evaluation_inference_config_path` |
| `choices`, implicit categorical list | `$choices` |
| `fixed` | `$fixed` |
| `variants` | `$variants` |
| Bare numeric range mapping | `$range: {low, high, step?, log?, type?}` |

Primitive scalars are fixed replacements; ordinary mappings recursively patch.
`$fixed` replaces a complete mapping or list. `$choices` samples any complete
value. `$variants` patches with selected partial mappings and may have sampled
siblings. Integer list keys patch indexed entries. Bare lists are rejected.
Search operators reserve `$` and never depend on the base field's type.

`evaluation_metric_directions` uses `min` and `max`, matching monitor `direction`.
Training parameter paths are exactly the canonical training paths.

## Cardinality, initialization, and dtype values

| Previous spelling | Canonical spelling |
| --- | --- |
| `num_buckets`, `buckets` | `num_buckets` |
| `min_freq`, `min_count` | `min_count` |
| `top_k`, `k`, `max_categories` | `max_categories` |
| `column_types`, `column_data_types` | `column_data_types` |
| `xavier`, `glorot`, `glorot_uniform`, `xavier_uniform` | `xavier_uniform` |
| `glorot_normal`, `xavier_normal` | `xavier_normal` |
| Distribution `standard_deviation` | `std` |

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
