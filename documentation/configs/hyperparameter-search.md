# Hyperparameter Search Command Guide

`sequifier hyperparameter-search` searches over complete canonical training
configurations with Optuna. It supports Bayesian optimization, random sampling,
finite grid search, cooperative pruning, and custom single- or multi-objective
evaluation.

```console
sequifier hyperparameter-search --config-path configs/hyperparameter-search.yaml
```

## Canonical configuration

Every search starts from a canonical training config named by
`base_config_path`. The `overrides` tree describes fixed replacements and
search spaces using the same paths as the training schema.

```yaml
base_config_path: train.yaml
hp_search_name: transformer-width-search
model_config_write_path: configs/hp-search
search_strategy: bayesian
n_samples: 40

overrides:
  global_training_spec:
    context_length: [64, 128]
    batch_size: [16, 32]
    learning_rate:
      low: 0.0001
      high: 0.001
      log: true
  model_spec:
    backbone:
      architecture:
        num_layers: {low: 4, high: 8, step: 2}
  training_plan:
    phases:
      0:
        epochs: [2, 4]
```

`base_config_path` resolves relative to the hyperparameter-search entry file.
The base may itself be a composed training config. If the search config supplies
`project_root`, that value is used in every generated trial; otherwise the base
training config's root is inherited.

When the base has exactly one value at the corresponding level, overrides may
use the singleton training paths. For example, `model_spec.interface` targets
the base's only interface, `dataset.part` targets its only dataset and part, and
a direct `training_plan.epochs` targets its only phase. These paths are
translated to canonical names before search spaces are compiled.

The historical self-contained search schema and historical flat training base
configs are not accepted. Every generated trial is validated as an authored
canonical `SequifierConfig` before training begins, so unknown paths, invalid
references, incompatible component types, and cross-field violations fail with
their canonical validation paths.

`model_name` and `project_root` cannot appear in `overrides`. Generated model
names use `[hp_search_name]-run-[index]`, and `project_root` is controlled by the
top-level search field.

## Override expressions

Fields omitted from `overrides` retain their base values. Override expressions
have these forms:

| Form | Meaning |
| --- | --- |
| Scalar | Fixed replacement. |
| List on a scalar field | Categorical choices. |
| `{low, high, step?, log?, type?}` | Integer or float distribution. `type` may be `int` or `float`; otherwise the base value and bounds determine it. |
| `{choices: [...]}` or `{$choices: [...]}` | Categorical choices for an entire value, including mappings and lists. |
| `{fixed: value}` or `{$fixed: value}` | Unambiguous fixed replacement for a mapping or list. |
| Numeric keys under a base list | Recursive overrides of zero-based list entries. |
| `{variants: [...]}` or `{$variants: [...]}` | Paired partial mapping variants, optionally followed by independently sampled sibling fields. |

Integer ranges default to `step: 1`. Grid search requires a `step` for float
ranges because an unstepped float interval is infinite. Logarithmic integer
ranges require `step: 1`; logarithmic float ranges cannot use `step`.

A direct list of lists samples a complete list-valued field, such as
`input_columns`. For arbitrary mapping- or list-valued candidates, prefer the
explicit `choices` wrapper:

```yaml
overrides:
  model_spec:
    backbone:
      architecture:
        choices:
          - dim_model: 128
            max_context_length: 512
            num_layers: 4
            attention: {type: mha, n_heads: 8}
            feed_forward: {dim: 512, activation: swiglu}
          - dim_model: 256
            max_context_length: 512
            num_layers: 6
            attention: {type: gqa, n_heads: 16, n_kv_heads: 4}
            feed_forward: {dim: 1024, activation: swiglu}
```

Each choice replaces the complete overridden subtree, keeping coupled values
valid. The same mechanism can select complete ingestion or decoder components,
training phases, or other canonical subtrees.

Use `variants` when candidates should patch the base mapping and other sibling
fields should remain independent. A variant that changes a component's `type`
starts from that new component shape rather than retaining fields belonging to
the old type.

```yaml
overrides:
  model_spec:
    interfaces:
      event_prediction:
        ingestion:
          variants:
            - {type: direct_embed, output_dim: 128}
            - {type: pass_through, output_dim: 128}
          dropout: [0.0, 0.1]
```

## Search controls

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `base_config_path` | `str` | Yes | - | Canonical training config used as the base. |
| `overrides` | `mapping` | Yes | - | Recursive fixed values and search spaces. May be empty. |
| `project_root` | `str` | No | Base value | Root written to every generated training config. |
| `additional_config_paths` | `str`, `list[str]`, or `null` | No | `null` | Complementary direct fragments of the search config. Relative paths resolve against the entry config's `project_root`; recursive composition and duplicate fields are rejected. |
| `hp_search_name` | `str` | Yes | - | Study name and generated-run prefix. |
| `model_config_write_path` | `str` | Yes | - | Directory, under `project_root`, for generated trial configs. |
| `search_strategy` | `bayesian`, `sample`, or `grid` | No | `bayesian` | Optuna TPE, random, or exhaustive finite-grid sampling. |
| `n_samples` | `int` | Except grid | - | Target total number of completed or pruned runs in the persisted study. The runtime attribute is also named `n_trials`. |
| `global_seed` | `int` or `null` | No | `null` | Sampler seed. Training seeds belong in canonical `overrides`. |
| `prune_trials` | `bool` | No | `true` | Enables cooperative pruning. Distributed pruning remains experimental. |
| `pruning_warmup_epochs` | `int` or `null` | No | `null` | Complete epochs before pruning may begin. Mutually exclusive with batch warmup. |
| `pruning_warmup_batches` | `int` or `null` | No | `null` | Training batches before pruning may begin. Mutually exclusive with epoch warmup. |
| `evaluation_metrics` | `list[str]` or `null` | No | `null` | Metric names expected from a custom evaluation script. |
| `evaluation_metric_directions` | `list[minimize\|maximize]` or `null` | With metrics | `null` | One optimization direction per metric. |
| `evaluation_script` | `str` or `null` | With metrics | `null` | Script invoked with the best exported model's evaluation ID. |
| `evaluation_inference_config` | `str` or `null` | No | `null` | Inference config run before the custom evaluation script. |

For grid search, omitting `n_samples` runs the complete finite grid. If it is
provided, it must exactly equal the grid size. For Bayesian and random search,
`n_samples` is a target total across invocations of the persisted study. If an
identical completed or pruned parameter set is proposed again, it is recorded
as a failed duplicate and does not consume a generated run number or count
toward the target.

## Custom evaluation

Without custom metrics, Sequifier minimizes the best validation loss. With one
metric it performs single-objective optimization; with several it records the
Pareto front.

```yaml
evaluation_metrics: [accuracy, latency_ms]
evaluation_metric_directions: [maximize, minimize]
evaluation_inference_config: configs/infer-validation.yaml
evaluation_script: scripts/evaluate.py
```

The evaluation script receives the exported model's evaluation ID, formatted
as `<run-name>-best-<epoch>`, as its only
argument. It must write
`outputs/evaluations/[evaluation-id].json` under `project_root`, containing
exactly the configured metric names.

## CLI and outputs

The search definition comes from YAML. `--skip-metadata` is the only
configuration-related command flag: it validates the canonical base without
loading metadata-derived values, so all required authored fields must already
be present.

Generated canonical training configs are written below
`model_config_write_path`. The Optuna SQLite study is persisted at
`state/optuna/[hp_search_name].db` under `project_root`, allowing later
invocations to continue toward the configured total.
