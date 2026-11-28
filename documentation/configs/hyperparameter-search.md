# Hyperparameter Search Command Guide

`sequifier hyperparameter-search` searches over complete canonical training
configurations with Optuna. It supports Bayesian optimization, random sampling,
finite grid search, cooperative pruning, and custom single- or multi-objective
evaluation. Supervised, policy, and mixed plans use the final phase’s selected
monitor value and optimization direction by default. See
[Training artifacts](../training/artifacts.md).

```console
sequifier hyperparameter-search --config-path configs/hyperparameter-search.yaml
```

## Minimal search configuration

Every search starts from a canonical training config named by
`base_config_path`. The `parameters` tree describes fixed replacements and
search spaces using the same paths as the training schema.

```yaml
base_config_path: train.yaml
run:
  study_name: transformer-width-search
  outputs:
    trial_config_directory: configs/hp-search
method: bayesian
trials: 40
parameters:
  model:
    execution:
      context_length:
        $choices:
        - 64
        - 128
    backbone:
      architecture:
        num_layers:
          $range:
            low: 4
            high: 8
            step: 2
  phases:
    fit:
      batch_size:
        $choices:
        - 16
        - 32
      optimization:
        model:
          learning_rate:
            $range:
              low: 0.0001
              high: 0.001
              log: true
      budget:
        epochs:
          $choices:
          - 2
          - 4
```

`base_config_path` resolves relative to the hyperparameter-search entry file.
The base may itself be a composed training config. If the search config supplies
`run.project_root`, that value is used in every generated trial; otherwise the base
training config's root is inherited.

Use the same named paths as the training base: `model.interfaces.<name>`,
`datasets.<name>.sources.<name>`, and `phases.<name>`.

The historical self-contained search schema and historical flat training base
configs are not accepted. Every generated trial is validated as an authored
canonical `SequifierConfig` before training begins, so unknown paths, invalid
references, incompatible component types, and cross-field violations fail with
their canonical validation paths.

`model_name` and `project_root` cannot appear in `parameters`. Generated model
names use `[study_name]-run-[index]`, and `run.project_root` is controlled by the
search run field.

## Policy and mixed plans

Use the phase names and role names from the base training config. For example,
to tune a PPO phase named `online_control` after supervised pretraining:

```yaml
base_config_path: train-supervised-ppo.yaml
run:
  study_name: control-search
  outputs:
    trial_config_directory: configs/hp-search
method: bayesian
trials: 20
pruning_warmup_iterations: 3
parameters:
  phases:
    online_control:
      clip_epsilon:
        $choices: [0.1, 0.2, 0.3]
      optimization:
        actor:
          learning_rate:
            $range: {low: 0.00001, high: 0.001, log: true}
```

| Final phase | Default search objective |
| --- | --- |
| Supervised | Loss on the monitored evaluation source |
| Imitation or IQL | Minimize held-out action negative log-likelihood |
| DPO | Minimize held-out preference loss |
| PPO | Maximize mean completed return by default, or use the configured episode monitor |
| GRPO | Maximize mean score |

Intermediate values come from the final phase. Policy trials report at phase-local
iteration numbers, so changing the length of an earlier phase does not shift the
pruner’s comparison points. Unavailable evaluations, such as a return with no
completed episodes, supply no intermediate value. The default objective requires
a finite selected metric in the final phase’s report. The phase’s `best` or `last`
handoff determines which actor and score are selected.

All trials using the default objective must select the same metric and evaluation
source. Online trials use the same evaluation seeds, step and time limits, and
action selection. A GRPO score monitor also fixes the reward scale. Generated
evaluation partitions require the same numeric run seed across trials because
that seed determines the held-out episodes or preference pairs.
Search drivers for the same study run serially and share its total trial budget.
Resuming a study checks its recorded objective identity. Use a new study
name for a different objective or for an older study whose trials predate this
record. Policy trials have ten minutes to checkpoint after a pruning request
before the search terminates the worker.

Collection output manifests and their part directories are scoped to each trial.
Later phases can consume those outputs through the base plan’s `type: rollout`
source and producer phase name.

## Parameter expressions

Fields omitted from `parameters` retain their base values. Parameter expressions
have these forms:

| Form | Meaning |
| --- | --- |
| Primitive scalar | Fixed replacement. |
| Ordinary mapping | Recursive patch. |
| `{$range: {low, high, step?, log?, type?}}` | Numeric distribution; `type` is `int` or `float`. |
| `{$choices: [...]}` | Categorical selection of any scalar, list, or mapping value. |
| `{$fixed: value}` | Complete mapping or list replacement. |
| Integer keys under a base list | Recursive replacement of zero-based list entries. |
| `{$variants: [...]}` | Partial mapping variants, optionally with independently sampled sibling fields. |

Bare lists, unprefixed operators, and bare numeric distribution mappings are
rejected. `$` is reserved for search operators; their meaning does not depend on
the base value's type.

Integer ranges default to `step: 1`. Grid search requires a `step` for float
ranges because an unstepped float interval is infinite. Logarithmic integer
ranges require `step: 1`; logarithmic float ranges cannot use `step`.

Use `$choices` to sample complete scalar, mapping, or list values:

```yaml
parameters:
  model:
    backbone:
      architecture:
        "$choices":
        - model_dim: 128
          max_context_length: 512
          num_layers: 4
          attention:
            type: mha
            num_heads: 8
          feed_forward:
            activation: swiglu
            hidden_dim: 512
        - model_dim: 256
          max_context_length: 512
          num_layers: 6
          attention:
            type: gqa
            num_heads: 16
            num_kv_heads: 4
          feed_forward:
            activation: swiglu
            hidden_dim: 1024
```

Each choice replaces the complete overridden subtree, keeping coupled values
valid. The same mechanism can select complete ingestion or decoder components,
training phases, or other canonical subtrees.

Use `$variants` when candidates should patch the base mapping and other sibling
fields should remain independent. A variant that changes a component's `type`
starts from that new component shape rather than retaining fields belonging to
the old type.

```yaml
parameters:
  model:
    interfaces:
      event_prediction:
        ingestion:
          "$variants":
          - type: embedding
            output_dim: 128
          - type: passthrough
            output_dim: 128
          dropout:
            "$choices":
            - 0.0
            - 0.1
```

## Search controls

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `base_config_path` | `str` | Yes | - | Canonical training config used as the base. |
| `parameters` | `mapping` | Yes | - | Recursive fixed values and search spaces. May be empty. |
| `run.project_root` | `str` | No | Base value | Root used by the search and written to every generated training config. |
| `additional_config_paths` | `list[str]` or `null` | No | `null` | Complementary direct fragments of the search config. Relative paths resolve against the entry config's `run.project_root` (or `.` if omitted); recursive composition and duplicate fields are rejected. |
| `run.study_name` | `str` | Yes | - | Study name and generated-run prefix. |
| `run.outputs.trial_config_directory` | `str` | Yes | - | Directory, under `run.project_root`, for generated trial configs. |
| `method` | `bayesian`, `random`, or `grid` | No | `bayesian` | Optuna TPE, random, or exhaustive finite-grid sampling. |
| `trials` | `int` | Except grid | - | Target total number of completed or pruned runs in the persisted study. |
| `sampler_seed` | `int` or `null` | No | `null` | Sampler seed. Training seeds belong in canonical `parameters`. |
| `prune_trials` | `bool` | No | `true` | Enables cooperative pruning. Distributed pruning remains experimental. |
| `pruning_warmup_epochs` | `int` or `null` | No | `null` | Completed epochs before pruning a supervised or imitation objective phase. |
| `pruning_warmup_batches` | `int` or `null` | No | `null` | Global training steps before pruning a supervised objective phase. |
| `pruning_warmup_iterations` | `int` or `null` | No | `null` | Updates completed in the policy objective phase before pruning. Choose one warmup unit per search. |
| `evaluation_metrics` | `list[str]` or `null` | No | `null` | Metric names expected from a custom evaluation script. |
| `evaluation_metric_directions` | `list[min\|max]` or `null` | With metrics | `null` | One direction per custom metric. Without a script, an explicit direction must match the final phase monitor. |
| `evaluation_script_path` | `str` or `null` | With metrics | `null` | Script invoked with the selected model's evaluation ID. |
| `evaluation_inference_config_path` | `str` or `null` | No | `null` | Inference config run before the custom evaluation script. |

For grid search, omitting `trials` runs the complete finite grid. If it is
provided, it must exactly equal the grid size. For Bayesian and random search,
`trials` is a target total across invocations of the persisted study. If an
identical completed or pruned parameter set is proposed again, it is recorded
as a failed duplicate and does not consume a generated run number or count
toward the target.

## Custom evaluation

A custom evaluation script scores the published model with the metrics and
directions you specify. Multi-objective studies record a Pareto front.
Intermediate pruning uses the training monitor for a single objective with
a matching direction.

```yaml
evaluation_metrics:
- accuracy
- latency_ms
evaluation_metric_directions:
- max
- min
evaluation_inference_config_path: configs/infer-validation.yaml
evaluation_script_path: scripts/evaluate.py
```

The evaluation script receives the exported model's evaluation ID, formatted
as the filename stem of the run manifest’s `model_path`, as its only
argument. It must write
`outputs/evaluations/[evaluation-id].json` under `project_root`, containing all
configured metric names. Extra metrics are allowed but produce a warning and
are ignored by the optimization.

## CLI and outputs

The search definition comes from YAML. `--skip-metadata` is the only
configuration-related command flag: it validates the canonical base without
loading metadata-derived values, so all required authored fields must already
be present.

Generated canonical training configs are written below
`run.outputs.trial_config_directory`. The Optuna SQLite study is persisted at
`state/optuna/[study_name].db` under `run.project_root`, allowing later
invocations to continue toward the configured total.
