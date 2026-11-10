# Public configuration surface implementation plan

## Design goal and reference

Build an ergonomic policy-training and rollout extension of the public configuration at commit `2d01d407cd2134c8197b1a18f0034d01486f18a9`. Use that commit's ownership rules, singleton conveniences, source references, component specifications, and authored/resolved separation as the foundation.

Use commit `082225833e1d744e662d094008bcc44268ab868a` as the capability baseline. Every supported operation and meaningful control at that commit must remain expressible and executable through the new design. The earlier reference commit guides layout, not the set of features to retain. Record and assess any subsequent capability additions before implementation finishes.

Extend `SequifierConfig` and `ResolvedSequifierConfig` instead of introducing a parallel public run abstraction. Keep `training_plan` as the home of training phases, `dataset_training` as the home of offline data and dataset training policy, and `model` as the home of architecture. Keep the explicit inference selector `model_type`, extending its values to include `policy`.

This is one clean public contract. The implementation removes the competing policy/mixed-run schemas and routing mechanisms. No configuration-version switch, conversion tool, deprecation period, alternate-format loader, import wrapper, or checkpoint-format upgrade is part of the work.

The examples below describe the proposed design, not the schemas currently implemented.

## 1. Reuse the reference commit's layout and abstractions

The reference commit's train, infer, and hyperparameter-search guides and their corresponding configuration models establish these patterns:

| Reference pattern | Apply it to the new surface |
| --- | --- |
| One shared backbone and named model interfaces | Policies select ordinary interfaces and existing target heads. No separate policy architecture declaration. |
| `dataset_training.<dataset>.parts.<part>` | Put offline policy data in typed dataset parts. Reuse dataset/interface ownership and metadata resolution. |
| `sources: [{source: dataset.part}]` | Use the same reference object in supervised and offline policy phases and dataset evaluation. |
| `training_plan.phases` and direct single-phase fields | Add algorithm-specific phase variants in the same plan; support the same singleton convenience for policy phases. |
| Unique routes inferred; named routes required when ambiguous | Apply the rule to policy interfaces, sources, and bindings. Never select the first eligible mapping entry. |
| `model.interface`, `dataset.part`, and `evaluation: true` | Keep these intentional authoring conveniences and their deterministic expansion into named structures. |
| `ComponentSpec` and `OptimizerPlan` | Reuse flattened component arguments, parameter selectors, optimizer groups, and scheduler validation for policy optimization roles. |
| `global_training` owns runtime, optimization defaults, and recovery | Extend that owner rather than adding policy-only top-level runtime/recovery conventions. |
| Preprocessing metadata owns types, vocabularies, statistics, and storage facts | Resolve policy execution metadata from the same source of truth. |
| `SequifierConfig` / `ResolvedSequifierConfig`, `InferenceConfig` / `ResolvedInferenceConfig` | Validate authored choices first; perform resource loading and execution resolution separately. |
| Explicit required `model_type`, with `generative` and `embedding` operations | Add the `policy` operation; remove `execution_mode` and artifact-triggered operation selection. |
| Composed YAML and HPO trees use ordinary config paths | Extend the same composition, singleton expansion, and trial validation mechanisms. |

The distinction between singleton and named authoring is useful ergonomics. Both forms expand to one schema and one execution meaning. Neither selects a different run class or runtime path.

## 2. Training: extend `SequifierConfig` and `training_plan`

### Run ownership

Use the reference layout with a small number of deliberate extensions:

| Location | Ownership |
| --- | --- |
| Top-level `project_root`, `model_name`, `device`, `seed`, export fields | Run identity, execution target, and outputs, as in the reference. |
| `model.backbone`, `model.interfaces` | Architecture, input/target columns, static/image/depth components, parameter initialization. |
| `initialize_from` | Optional full-model artifact initialization; distinct from resuming a run or reusing only a backbone repository. |
| `global_training` | Shared runtime settings, window defaults, supervised optimization defaults, determinism, precision, checkpointing, and resume. |
| Top-level `model_path`, `report_path`, `phase_directory` | Optional destinations for the final model, run report, and phase artifacts; retain configurable outputs alongside existing export fields. |
| `dataset_training` | Named offline datasets, interface references, typed parts, supervised loss settings, and dataset freezing. |
| `policy_bindings` | Interface-keyed task declarations: named actions, action distributions, and explicit policy input bindings. |
| `training_plan` | Ordered phases, phase budgets, and supervised curriculum policy. |
| `evaluation` | Shared dataset evaluation specification, with phase-local evaluation where the algorithm needs a different contract. |

Do not add top-level `phases`, `run_type`, `policy_interface`, `execution_model`, `sources`, `preferences`, or a second `resume` block. Their responsibilities belong to the owners above.

Extend `TrainingPlanModel.phases` to a discriminated union keyed by `algorithm`: `supervised`, `policy_imitation`, `iql`, `ppo`, `dpo`, and `grpo`. An omitted algorithm means `supervised`, keeping a simple supervised phase as concise as the reference. Insert that default before discriminated validation.

Keep algorithm-specific budgets: supervised/imitation `epochs`, IQL/DPO `updates`, PPO `total_env_steps`, and GRPO `total_groups`. Do not make them fit an artificial universal duration field. Keep supervised `mode`, `selection`, weights, and batches per selection specific to supervised phases.

Direct single-phase fields under `training_plan` work for every algorithm. Expand them into one phase named `train`; fill `mode: sequential` only for supervised phases. Reject direct phase fields combined with `training_plan.phases`. Policy phases never receive supervised-only defaults.

### Global settings and phase optimization

Retain `global_training` as one shared owner, including in policy-only runs. Extend its validation so users do not have to invent supervised settings for an online policy phase:

- Require supervised objective/loss/data-loader settings only when selected supervised phases use them.
- Resolve window/model-contract facts from declared model metadata or `initialize_from`, checking explicit values as assertions.
- Keep policy-specific learning dynamics in their phase schema, including batch sizes that have algorithm-specific meanings.
- Reuse `ComponentSpec`, `OptimizerPlan`, parameter catalogs, and freezing selectors for each policy `optimization` role. A policy role uses its phase settings and documented algorithm defaults; do not silently inherit a supervised epoch scheduler as a policy update scheduler.
- Phase `optimizer_state: reset | continue` replaces the global reset switch. Continuing state requires matching algorithms, roles, parameter ownership, optimizer/scheduler definitions and a last handoff; mixed coordinator transitions are rejected.
- Keep `global_training.resume` using `ResumeConfig.policy` and `checkpoint_path`. Extend that block with an environment recovery `mode: exact | reset_environments`, defaulting to `reset_environments` when relevant. Validate snapshot capabilities for exact online recovery.
- Put automatic offline holdout sizing at `global_training.validation_fraction` and bounded episode caching at `global_training.episode_cache_size`. Keep both configurable; neither requires an offline preprocessing step.
- Retain phase-local `freeze`, `evaluation_interval`, and `checkpoint_interval`, alongside dataset-specific freezing. Resolve their combined parameter ownership with the existing catalogs/selectors and validate which roles may update each parameter.

Retain every applicable algorithm-specific control, including PPO partial-rollout handling, normalization, reward scaling, clipping/KL/entropy/value settings, optimization epochs, minibatching and accumulation; IQL expectile, temperature/weight limits, target updates and discounting; DPO beta and reference selection; and GRPO grouping, reference/old-policy KL settings and collection limits. Keep `reference.source: phase_start | artifact` and `reference.path` for DPO/GRPO. Keep `scoring_chunk_size`, `max_buffer_bytes`, `max_env_steps`, `archive_directory`, and PPO `archive_scores` on the applicable phase schemas. Their units and effects are algorithm-specific and must be documented rather than dropped during consolidation.

Use top-level `model_path`, `report_path`, and `phase_directory` overrides for run outputs and `global_training.resume.checkpoint_path` for the rolling run checkpoint destination, including when resume policy is `never`. Preserve timed/batch/epoch/latest checkpoint controls and supported export/dropout/embedding options. Resolve paths under `project_root` and validate input/output collisions. Document derived defaults once for the unified run; no output location should become unconfigurable merely because its former run class is removed.

Keep supervised phase transitions on continuing model weights. Add `handoff: last | best` to phases, defaulting to `last`; `best` requires a usable evaluation monitor and restores the complete selected model/distribution state. Handoff and optimizer-state continuation are separate decisions and must be validated together.

### One offline source system

Extend dataset parts with a discriminated `type`, following the component pattern:

| Part type | Authored data contract | Eligible uses |
| --- | --- | --- |
| `prepared` (default) | The reference `metadata_config_path`, media paths, and file ordering. | Supervised training; policy learning only when explicit action/episode mappings supply the needed task semantics. |
| `sequence` | Direct `paths`, a source `manifest`, or a metadata-selected split; optional schema/normalization metadata and typed column mappings. | Imitation and transition-based policy learning from tables, trajectories, or prepared data. |
| `preferences` | Preference manifest referencing sequence parts through the shared dataset catalog; metadata supplied by those parts or the resolved interface as needed. | DPO preference scoring. |

Reuse `SequenceSourceConfig`, `PolicyTableColumns`, trajectory readers, and preference readers as implementation building blocks. Refactor their public placement into dataset parts instead of exposing parallel top-level source registries. Keep raw/model encoding declarations with the part's data contract.

Sequence parts retain the current source controls: `paths`, `manifest`, `format`, `columns`, `encoding`, `metadata_config_path`, `split_index`, `static_data_path`, `episode_end`, `episode_namespace`, `units`, and `action_semantics`. Select exactly one data location: direct paths, a manifest, or `metadata_config_path` with `split_index`. Metadata may also accompany direct paths/manifests as a schema assertion; its presence alone must not replace their data location. Retain automatic format detection and explicit table/trajectory/prepared reading where currently supported.

Raw table and trajectory inputs do not require a generated preprocessing metadata file when the model/artifact, source declarations, and frozen normalization supply the necessary contract. Model-encoded/prepared inputs still require the metadata needed to interpret their values. Missing necessary information is a precise resolution error, not a requirement to preprocess every raw source. Preference manifests resolve their sequence references against the same dataset/part catalog, with no independent top-level registry.

Keep explicit `action_semantics: executed_predictions` for learning from forecast trajectories, separate from policy-action records. Retain episode identity namespaces, end-of-file completion declarations, previous-action/successor mappings, and bootstrap flags. These affect task meaning and leakage checks, not merely data-loading syntax.

`dataset_training.<name>.model_interface` still owns the dataset's interface. A reference to a whole dataset selects all its parts in declaration order; `dataset.part` selects one. Parts in one dataset must agree on the selected interface's codecs, normalization, and input semantics. Phase validation rejects incompatible part types or missing task fields.

Make `criterion` and supervised loss settings required when that dataset is selected by a supervised phase. Policy-only datasets do not need dummy criteria. Rewards, terminal/truncation flags, successors, action labels, and episode boundaries require explicit data semantics; a supervised window is not automatically an RL transition.

Use `TrainingSourceSpecModel` references throughout: `sources: [{source: demonstrations.recorded}]`, not a different list-of-strings syntax for policy phases. DPO's `preferences` field uses the same reference object and requires a preference part. Validate whole-dataset references for that algorithm rather than silently skipping incompatible parts.

Retain explicit offline `validation_sources` and DPO `validation_preferences` on phases, using the same source reference objects as their training counterparts. Resolve holdouts in this order: explicit phase validation references, declared metadata validation partitions where available, then a seeded automatic split using `global_training.validation_fraction`. Apply the automatic split only when no validation partition was supplied. Partition sequence data by episode and shared demonstration/group identity; split preference data by connected underlying episodes/spans rather than individual pairs. Detect overlap even when the same content is exposed through different source names. Keep explicit separate validation datasets and automatic splitting as first-class workflows.

Dataset evaluation uses the same source references as training. Its selected sources/metrics must agree with the effective validation partitions; an evaluation declaration must not silently re-split or replace phase validation data. Extend `EvaluationMonitorSpecModel` with registered algorithm metrics while retaining the reference spelling `mode: min | max`. Online evaluation retains its environment, seeds, episode budget, timeout, and deterministic/stochastic `selection`; expose its monitor with the same `metric`/`mode` vocabulary. A phase-local evaluation specification replaces the shared evaluation for that phase. `evaluation: true` expands only when one dataset evaluation source can be inferred; it never invents online evaluation seeds or an environment. Each phase must have a compatible effective evaluation specification when its algorithm or handoff requires one.

For example, a raw demonstration source and a separate holdout need no source metadata file when an initialization artifact supplies their model contract:

```yaml
dataset_training:
  demonstrations:
    model_interface: control
    parts:
      train: {type: sequence, paths: [data/train.csv], format: table}
      holdout: {type: sequence, paths: [data/holdout.csv], format: table}
training_plan:
  algorithm: policy_imitation
  sources: [{source: demonstrations.train}]
  validation_sources: [{source: demonstrations.holdout}]
  epochs: 3
```

This fragment assumes matching column names and disjoint episode/group identities. Omitting `validation_sources` permits the configured automatic holdout when the source has no declared validation partition.

### Minimal policy authoring

This proposed complete run declaration initializes from an ordinary model artifact, infers its unique interface, and reuses the reference singleton conventions:

```yaml
project_root: .
model_name: controller-imitation
device: cpu
initialize_from: models/controller.pt

global_training:
  resume: {policy: never}

policy_binding:
  actions:
    action: {type: categorical, values: [left, right]}

dataset:
  part:
    type: sequence
    metadata_config_path: configs/metadata_configs/demonstrations.json

training_plan:
  algorithm: policy_imitation
  epochs: 3

evaluation: true
export_pt: true
```

`policy_binding` is the singleton spelling of `policy_bindings.<unique-interface>`, following `model.interface`; it has exactly the same schema and meaning. Reject singular and plural declarations together, even if their contents match. A binding never lives under `model.interfaces.<name>.policy` in authored config.

For mixed training, the surrounding model, global settings, and datasets keep their locations. The phase list can contain:

```yaml
training_plan:
  phases:
    - name: supervised_warmup
      epochs: 5
      mode: sequential
      sources: [{source: control_data.train}]
    - name: imitate
      algorithm: policy_imitation
      sources: [{source: demonstrations.recorded}]
      epochs: 3
    - name: online_improvement
      algorithm: ppo
      model_interface: control
      total_env_steps: 4096
      rollout_length: 128
      environment:
        factory: my_control:make_environment
        implementation_id: control-v1
      evaluation:
        environment:
          factory: my_control:make_environment
          implementation_id: control-v1
        seeds: [17, 29]
        max_episode_steps: 100
        timeout_seconds: 30
        monitor: {metric: mean_completed_return, mode: max}
```

This second example is a fragment; it assumes the referenced datasets and binding for interface `control` are declared. Offline phases derive their interface from their dataset references. An explicit phase `model_interface` is an assertion when a source already fixes it, and selects the route for online phases. Infer it only when one eligible interface exists. A policy phase operates on one interface; reject offline references spanning several interfaces.

## 3. Policy bindings: task choices separated from execution facts

Introduce a small authored `PolicyBinding` and a separate `ResolvedPolicyBinding`. Use named `actions` for a single action and for several actions alike. Remove the `action_column`/`action_values` alternative and the runtime distinction selected by `is_joint`.

An authored binding contains:

- Named action distributions/domains, using `ActionFieldConfig` and existing categorical/real type vocabulary.
- `decision_position` when selecting a particular decoder output is necessary.
- `previous_action_inputs` and `initial_inputs` when ordinary declared model inputs require them.
- Optional return conditioning, defaulting to disabled, and its return definition/initial desired return when enabled.
- Task `units`, checked against source and environment declarations and retained in execution/trajectory identities.

Resolve action ordering from the selected interface's target/decoder contract. Validate that actions are a compatible subset of model targets, categorical support matches executable decoder IDs, real scales are finite and explicit, and conditional activity respects decoder dependencies. Leave unrelated prediction heads available to supervised phases.

The model and preprocessing metadata supply observation/static membership, types, class counts, ID maps, normalization, context geometry, and decoder semantics. Do not expose `alignment`, `input_schema`, `input_statistics`, class maps, `execution_model`, or phase `representation` in authored config. Do not insert previous-action/return inputs or replace target heads implicitly.

A fresh policy model uses the same `ModelSpecModel`, component initialization, and metadata-driven interface resolution as a fresh supervised model. For online-only models, add optional `global_training.metadata_config_path`: a string when the model interface is unique, or an interface-to-path map when named routes need different metadata. This reuses the reference's scalar-or-named-map pattern for media paths and supplies input metadata without a dummy offline dataset. If a selected dataset or initialization artifact already supplies the interface metadata, this declaration is unnecessary; simultaneous declarations must agree. Keep statistics/vocabularies in the metadata file. Reuse the explicit frozen-normalization contract for online data instead of deriving normalization from a changing rollout stream.

Full-model artifact initialization resolves architecture, codecs, and existing weights once. Bindings attach to existing targets; explicit architecture or metadata assertions must agree. Reuse parameter initialization/catalog abstractions for declared new distribution parameters. Do not overload backbone repository settings with full-policy initialization.

Artifact serialization retains resolved policy metadata for portable execution. Artifact readers and checkpoint validation consume the resolved type directly; they do not pass internal dictionaries through authored config validation. Use one ordinary model-input execution path for all named action configurations.

Removing the separate scalar-action branch must retain single-categorical behavior cloning and return-conditioned training. Provide explicit model inputs and binding declarations for previous-action history, initial/start values, and desired-return features; the resolver must support their encoding and update rules without depending on that branch. Cover string/numeric category labels, class IDs, first-step history, sequence fragments, padded windows, return-to-go/remaining-return calculations, sampled/deterministic action selection, likelihood, gradients, and collection/replay. Existing mixed/real distributions, learned scales, bounds, conditional activity, inactive values, and point-mass atoms remain available through `ActionFieldConfig`. Branch deletion is gated on capability coverage in the unified execution path.

## 4. Inference: extend the existing `model_type`

Keep `model_type` required and explicit, just as in the reference commit. Extend its allowed values; remove `execution_mode` and artifact-based auto-selection.

| `model_type` | Operation |
| --- | --- |
| `generative` | Ordinary prediction on the selected model interface, even when the artifact also has policy metadata. |
| `embedding` | Extract configured representations using the existing embedding path. |
| `policy` | Execute the selected interface's resolved policy binding. Fail if that capability is absent. |

There is no operation default to discover from an artifact. Missing `model_type` is a normal required-field error. Interface inference still follows the unique-route rule; unsupported formats and ambiguous routes produce explicit errors. Adding policy metadata cannot change what `model_type: generative` does.

Keep the reference field locations and defaults for `project_root`, `model_path`, `data_path`, `training_config_path`, `dataset`, `part`, `model_interface`, read/write formats, device, batching, sampling, and media inputs. Share common inference fields rather than maintaining independent policy-only defaults. Retain top-level `output_path` for an explicit policy prediction destination, along with output-format and probability controls and default derived paths.

For `model_type: policy`, keep `desired_return`, `desired_return_column`, named `previous_action_columns`, `sequence_id_column`, and `item_position_column`. A single previous action uses a one-entry named map. Retain independent-row and ordered-sequence inference, fragment-start previous actions, and recorded action history. Validate missing fragment context and incompatible return conditioning rather than silently resetting history or discarding the desired return.

Use policy `action_selection: deterministic | stochastic` for action choice, consistently with online evaluation; use `enforce_determinism` for runtime algorithm determinism. Preserve seed-controlled stochastic sampling, categorical probability outputs, and joint-action probability/log-probability outputs with their existing meanings. Generative sampling/dropout and embedding output options remain available. Consolidating defaults must not remove either action-selection mode or make stochastic sampling depend on disabling runtime determinism.

Refactor `load_inferer_config` around the reference resolution order:

1. Compose YAML and apply supported CLI overrides.
2. Validate the explicit operation and authored route/field shapes.
3. Load the selected training route and/or portable artifact through a shared cache keyed by normalized path, independent of the chosen operation.
4. Resolve interface and metadata; explicit values are assertions against those contracts.
5. Validate operation capabilities and build the resolved inference specification.

Fix the PT cache issue as part of this loader: prediction must not depend on a policy-routing helper having populated `artifacts[model_path]`. Require the declared policy operation to resolve a usable policy interface; no fallback to generative prediction. Extend the existing inference models with a typed operation-specific specification and share resolution services instead of dispatching between unrelated public constructors.

## 5. Rollout: explicit type and local collection budget

Use a discriminated `rollout.type: forecast | policy`, following the reference's `type` convention for components. Both variants share `adapter`, `trajectory_path`, `seed_index`, positive `max_steps`, and optional positive `max_episode_steps`.

```yaml
project_root: .
model_path: models/controller.pt
model_type: policy
data_path: data/initial_observations.csv
read_format: csv
device: cpu
rollout:
  type: policy
  adapter:
    factory: my_control:make_environment
    implementation_id: control-v1
    config: {}
  seed_index: 0
  max_steps: 200
  trajectory_path: outputs/control-trajectory.json
```

`model_type` selects the inference operation; `rollout.type` declares the interaction contract. Validate `forecast` with `generative` and `policy` with `policy`; reject embeddings with a rollout. Do not make either discriminator silently override the other.

Reuse a shared adapter-factory base with `factory` and `config`, plus distinct forecast/online schemas. Forecast adds input `bindings`; online policy adds implementation identity, units, and capabilities. The type fixes the protocol: forecast uses protocol 1, online policy uses protocol 3. Keep protocol validation internal; users need not repeat a fixed `protocol_version` in YAML. Use the same online adapter schema in rollout, training `environment`, and online evaluation.

Keep the two runtime protocols distinct: forecast reset receives prepared history and steps return external inputs; online policy reset/step supplies observations and exposes snapshot/group-start capabilities. Share loading, resource cleanup, limit validation, and meaningful common vocabulary.

Define the collection controls precisely:

- `max_steps` limits transitions collected by this invocation. Reaching it alone is a collection cutoff.
- `max_episode_steps` is an optional environment time limit with truncation semantics. True termination remains distinct; preserve correct bootstrapping when limits coincide.
- Forecast `seed_index` selects a prepared window in dataset/chunk order. Policy `seed_index` selects an initial observation row. Document the seed contract under each explicit rollout type.

An environment rollout requires no `autoregressive` or top-level `generation_steps`. Reject those fields when `rollout` is present, so there is only one collection budget. Standalone autoregressive prediction keeps the reference `autoregressive`/`generation_steps` controls.

Pass the local budget through forecast/policy inference, collectors, trajectory metadata, and public rollout APIs. Keep the collected data formats faithful to their forecast or policy semantics.

## 6. Resolution, execution, and public APIs

Reuse the existing pipeline and extend its typed results:

```text
compose YAML + supported CLI overrides
  -> expand singleton conveniences
  -> validate authored SequifierConfig / InferenceConfig
  -> resolve model, metadata, sources, bindings, and capabilities
  -> ResolvedSequifierConfig / ResolvedInferenceConfig
  -> phase coordinator and native algorithm runners
```

Singleton expansion is deterministic, non-mutating, and structural. It infers only unique routes, rejects singular/plural mixtures, and never performs artifact reads or chooses an execution implementation. Resolve an artifact-backed unique interface after resource loading, with the same ambiguity rule.

Retain `additional_config_paths`, project-root-relative paths, duplicate-field rejection for complementary fragments, list replacement for overlays, and explicit assertions against artifact/metadata contracts. Extend discriminator replacement to phase `algorithm`, dataset-part `type`, rollout `type`, and operation-specific inference fields, so stale fields cannot survive a variant change.

Extend HPO's same-path parameter tree and singleton expansion using the actual authored model. Validate each materialized trial; do not create another mirrored schema. Make algorithm/search capability limits explicit before trials start.

Use `LoadedTrainConfig`, `ResolvedTrainingPhase`, `SelectedInterfaceConfig`, dataset build views, and `ExecutionSchema` as foundations. Extract neutral shared definitions where necessary to avoid circular imports; do not add a new public run-envelope class merely to carry internal runner arguments.

Refactor the phase coordinator to consume the resolved `training_plan`. Preflight all source/interface/environment and execution-mode requirements before any optimizer update or output publication. Reuse native supervised and policy runners behind internal typed phase specifications; remove public `PolicyRunConfig`/`LearningRunConfig` dispatch. Retain supported supervised distributed execution, component initialization, exports, integrations, and optimizer plans as product capabilities. Unsupported mixed execution combinations fail with the relevant phase/capability named.

Preflight restrictions may describe combinations the capability baseline does not support; they cannot be used to remove a combination that already works. Retain direct Python operations for model/policy loading, attachment, training, evaluation, collection, trajectory import/export, preference scoring, and replay/inspection through the new shared types. Configuration cleanup must not make these operations CLI-only. Keep the sequence-import preprocessing operation, sharding controls, source readers, and forecast/policy trajectory save/load/validation facilities; refactor their authored contracts to reuse the same source and binding types.

Use one recovery contract owned by `global_training.resume`. Current-design checkpoints record phase position, committed optimizer clocks, RNG, role optimizers/schedulers/scalers, source/model/binding identities, selected handoff state, and necessary environment state. Validate exact versus reset-environment recovery and prevent duplicate phase execution. Test recovery of runs created by this design; no checkpoint conversion or older-format support is required.

Update CLI and Python entry points together. Expose the same authored and resolved types through the existing public API modules. Remove superseded config classes, alternate selectors, scalar-action branches, and source registries rather than wrapping them.

## 7. Implementation sequence and acceptance gates

Before implementation, create a field-by-field capability inventory against `082225833e1d744e662d094008bcc44268ab868a`. Inspect public config models, CLI commands/overrides, Python API exports, readers, runners, and examples. For each functional option, record its new authored location or derived execution owner, supported combinations, and an executable acceptance test. An internal fact may become resolved state; a user-controllable behavior needs a new authored home. Retired duplicate spellings are not separate capabilities. The matrix below establishes minimum coverage and must be expanded if the inventory finds additional functions.

| Current capability family | New owner / required expression | Completion evidence |
| --- | --- | --- |
| Raw tables, trajectories, source manifests, prepared splits, media/static inputs | Typed dataset parts with direct locations and optional/required metadata according to encoding. | Raw-source training without a generated metadata file; manifest and prepared-split execution; static/media resolution. |
| Episode/transition semantics and forecast-derived actions | Part column mappings, end declarations, namespaces, units, encoding, and `action_semantics`. | Correct history, terminal/bootstrap semantics, episode identities, and forecast-trajectory import. |
| Explicit and automatic validation | Phase `validation_sources` / `validation_preferences`, metadata partitions, `global_training.validation_fraction`. | Separate holdout execution; deterministic episode/component splits; overlap rejection. |
| Bounded episode loading | `global_training.episode_cache_size` and existing data-loader/memory controls. | Cache limit respected without changing episode content or partitions. |
| All supervised objectives/components and training policies | Existing model/global/dataset fields, phase selection/interleaving, curriculum, losses, hashing, initialization/freezing. | Supported model/objective fixtures, loss/gradient checks, and multi-dataset execution. |
| Imitation, IQL, PPO, DPO, GRPO and their optimization roles | Typed phases retaining every applicable budget, objective parameter, optimizer/scheduler and freeze control. | Each algorithm executes a meaningful update with representative non-default controls. |
| References, scoring, archives, and collection memory/step caps | Phase reference/scoring/archival/limit fields. | Phase-start/artifact references, chunked score agreement, archive score round trip, enforced limits. |
| Single categorical, return-conditioned, joint and real policies | Named bindings, explicit model inputs, existing action distribution fields. | History/return encoding, sampling, likelihood/gradients, update, and replay tests. |
| Policy inference controls | Explicit `model_type`, `selection`, desired-return/history/sequence columns, `output_path`, probabilities. | Independent rows and fragments, deterministic/stochastic actions, custom output and probability checks. |
| Prediction, embeddings, dropout and supported artifact/device routes | Existing inference/export fields and explicit operation. | PT/ONNX and supported device/mode tests; prediction independent of policy metadata. |
| Forecast and policy environment interaction | Typed rollouts/adapters, units/capabilities, local budgets, seed and episode controls. | Both protocols, snapshot/group-start capabilities, truncation/cutoff/cleanup checks. |
| Output and checkpoint control | Run output paths/export settings; global resume/checkpoint destinations and intervals; phase intervals/archives. | Custom destinations used, snapshot timing/rolling checkpoint behavior, report/phase artifact checks. |
| Recovery and mixed-phase handoff | Global resume mode, full phase state and identities, phase handoff/optimizer policy. | Same-design stop/resume, exact/reset environments, supervised → policy → supervised with no duplicate updates. |
| Public CLI/Python workflows and sequence import | Shared authored/resolved models and existing operation entry points. | Equivalent CLI/Python execution; import/sharding, direct evaluation/collection and trajectory operations. |
| Composition, HPO, integrations and supported distribution | Reference composition/search trees, runtime services, integration APIs. | Supported overrides/trials/integrations/distribution remain executable through the unified contract. |

Each inventory entry must link its acceptance test before its implementation slice is complete. The final release gate requires no uncovered or removed baseline capability. New field names, defaults, explicit selectors, and derived metadata ownership may differ; functional expressiveness and supported operations must remain.

| Slice | Implementation and principal locations | Acceptance gate |
| --- | --- | --- |
| 1. Reference contract and authored models | Complete the capability inventory; extend `SequifierConfig`, `TrainingPlanModel`, inference models, singleton expansion, and neutral shared definitions in `config/`. Write schema/examples first. | Every inventoried control has an owner; supervised/policy singleton and named configs use one model; public schemas contain no execution metadata or competing selectors. |
| 2. Dataset parts and shared references | Extend dataset parts/resolution, source and preference readers, validation partitioning, cache controls, source references and evaluation models. | Shared references resolve each data type; direct raw paths/manifests work without unnecessary metadata; explicit/automatic holdouts and bounded caching execute. |
| 3. Binding and execution resolution | Add authored/resolved bindings; update attachment, execution schemas, artifact config/export and shared materialization. Remove scalar-action execution selection only after coverage. | Single-categorical/return-conditioned and joint/real policies retain input, likelihood, update and replay capabilities; attachment and new artifact round trips work. |
| 4. Training lifecycle | Update training/coordinator/algorithm runners, all phase controls, optimizer/freezing services, output/checkpoint controls, global resume and API exports. | Each algorithm and references/archives/limits work; policy-only and mixed plans execute; custom outputs, handoff, role ownership and recovery pass their inventory tests. |
| 5. Inference and rollout | Extend explicit `model_type`, shared artifact caching, complete policy inference options, typed rollout/adapter schemas and public APIs. | Custom outputs, probabilities, return/fragment history and both action-selection modes work; generative routing is stable; both rollout contracts execute. |
| 6. Tooling, docs, and cleanup | Update composition, HPO, imports/CLI/Python operations, templates, examples/guides and CI; delete superseded public/runtime branches after coverage. | Every inventory entry has a passing acceptance test; all baseline capabilities remain expressible/executable; examples and shared CLI/Python contracts validate. |

Dependencies: establish the authored shape in slice 1 before implementing resolution. Dataset resolution and bindings converge before complete training integration. Inference/rollout can build on binding resolution while training lifecycle work proceeds. Include recovery identities and fixtures during slice 4, not after public APIs are finished.

## 8. Validation and completion criteria

Keep tests in the adjacent `../sequifier-test` repository: schema and validation tests under `local/unit`, execution tests under `local/integration`, and shared fixtures in `local/conftest.py`. Reuse its existing conventions and validate the intended behavior of the new design.

| Area | Required checks |
| --- | --- |
| Ergonomics and structure | Supervised singleton example from the reference; singleton policy run; named mixed run; independent singleton conveniences; unique-route inference; singular/plural conflicts; no extra routing/version flags. |
| Authored/resolved boundary | JSON schemas exclude internal fields; authored structural validation performs no file/environment loading; unknown fields and invalid algorithm-specific budgets fail at useful paths. |
| Dataset resolution | Raw table/trajectory paths without preprocessing metadata when resolvable; manifests/prepared splits; part eligibility; encoding/static/media/units; explicit versus automatic validation and configurable fractions; episode/preference leakage; episode-cache bounds; evaluation source/monitor validation. |
| Policy execution | Single-categorical behavior cloning and return conditioning; categorical IDs/start values; fragment/padded history; real/joint distributions and atoms; decoder order; fresh/initialized models; sampling, likelihood, gradients, update and replay after artifact reload. |
| Training | Each algorithm's full control inventory, budgets and roles; reference selection; chunked scoring; partial rollouts; archives/scores and buffer/step limits; dataset/phase freezing; interleaving/curriculum; handoff and optimization lifecycle. |
| Runtime capabilities | Supervised distribution/integrations and image/depth/static inputs; clear preflight errors for unsupported combinations; no source-shape-based runner selection. |
| Inference | Explicit operations and shared PT cache; independent-row/sequence/fragment inputs; desired-return/history column overrides; deterministic/stochastic selection independent of runtime determinism; custom output paths, probability outputs, dropout/embeddings and supported artifact/device routes. |
| Rollout | Forecast/online protocol validation; seed preparation for each type; budget 1 and invalid limits; termination/truncation/cutoff/bootstrap; adapter close on failure; no redundant generation controls. |
| Recovery | Uninterrupted versus resumed updates; phase-boundary commits; exact/reset-environment recovery; changed source/binding/adapter identities rejected; no repeated completed phase. |
| Outputs and checkpoints | Custom model/report/phase destinations; checkpoint destination with resume enabled/disabled; phase and timed/batch/epoch/latest intervals; archive paths; export/dropout settings and input/output collision checks. |
| Tooling | Composition/variant replacement; singleton/named HPO paths; CLI/Python operation coverage; sequence import/sharding; direct collection/evaluation and trajectory/preference inspection; all inventory entries linked to tests. |

Use deterministic tiny CPU models and toy adapters for integration tests. Include actual artifact-backed inference, one mixed phase run, and stop/resume checks. Test encoding, gradients/log probabilities, and observable outputs rather than relying on schema parsing alone.

Run focused pytest suites, relevant training/inference/rollout checks, Ruff and type checks. Update `README.md`, `documentation/configs/{train,infer,hyperparameter-search}.md`, policy/supervised-policy/rollout guides, templates and example configs; regenerate `src/sequifier/consolidated-docs.md` with `hooks/consolidate_docs.py`.

The design is complete when a user familiar with the reference commit can add policy data and phases using the same ownership, naming, metadata, source-reference, and singleton rules, and every capability inventoried from `082225833e1d744e662d094008bcc44268ab868a` remains expressible and passes its acceptance test. The additional concepts should express the task itself: action distributions, interaction contracts, and algorithm-specific budgets. They should not expose implementation paths or introduce another way to describe a run.
