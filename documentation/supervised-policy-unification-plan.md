# Supervised and policy learning unification plan

Status: proposed implementation plan, based on the current source and the two supplied analyses. No training implementation or experimental validation is included in this change.

## 1. Intended outcome

Unify around one model execution contract, one portable model artifact, and one run plan with algorithm-specific phase runners. Supervised objectives and policy objectives should consume the same model and prediction semantics. Episodes, preferences, critics, and online collection remain explicit algorithm-specific concepts.

The two completion requirements are:

1. **Architecture parity:** every architecture currently admitted by supervised learning can participate as an actor in policy learning, including arbitrary combinations of the currently supported scalar real and categorical targets, ordinary and autoregressive decoder branches, multiple interfaces, structured/depth/image ingestion, static conditioning, hashing, and nontrivial decoder support and prediction positions.
2. **Supervised-to-policy transfer:** every reconstructable supervised Sequifier model artifact can initialize a policy phase without first being trained or exported as a policy. Existing compatible prediction weights and preprocessing are retained. Necessary stochastic semantics are supplied explicitly for deterministic real heads.

“Every architecture” concerns the model's supported execution surface. It does not mean that a dataset automatically contains rewards, preferences, successor states, or observations that are available at action time. Every policy still needs an explicit task binding and the data or environment required by its algorithm. A model consuming information unavailable at decision time needs a declared input adaptation; there is no exact behavior-preserving transfer in that case.

This plan covers trainable Sequifier artifacts and supported run-checkpoint extraction. ONNX inference artifacts alone do not contain the complete training contract and are not a general source for further training. Retain existing supervised export behavior; specify policy deployment export capabilities separately rather than claiming arbitrary stochastic sampling exports automatically.

## 2. What the current code establishes

The supplied analyses correctly identify shared model construction and tensors, with separate policy configuration and execution. The following details affect the proposed work:

| Boundary | Current implementation | Consequence |
| --- | --- | --- |
| Actor | `rl/policy.py` wraps the ordinary network but synthesizes policy inputs, metadata, and one categorical target. | Replace categorical assumptions without introducing a second architecture resolver. |
| Configuration | `PolicyRunConfig.resolve` permits only the selected interface, one action target, scalar fields, and decoder `support=prediction_length=1`. | Typed actions alone will not deliver architecture parity. |
| Transfer | `rl/learner.py::_initialize_actor` calls `load_policy`, then validates identical policy representation before loading weights. | A shared artifact file format currently does not imply supervised-to-policy compatibility. |
| Real targets | `TargetDecoderBranch` and the autoregressive branch already emit scalar real predictions. | Reuse those predictions as location parameters; do not discard and relearn the heads. |
| Autoregression | `AutoregressiveTransformerDecoderBranch` already supports teacher forcing and greedy generation across ordered targets at one temporal position. | Add stochastic generation to this execution path and score actual prefixes consistently. |
| IQL | `HistoryCritic` already evaluates a supplied categorical action through `nn.Embedding` and emits a scalar Q-value. | Analysis 2 overstates this particular redesign: generalize the action encoder, not an all-actions Q-vector architecture. |
| Environment | `OnlineEnvironmentAdapter.step` already accepts a mapping of action names to domain scalars. | Its action container can carry multiple scalar fields; collection, validation, history, and storage still assume one field. Observation transport must expand for non-scalar model inputs. |
| Trajectories | Records already contain named predictions, but policy validation restricts them to one action head and nonpositive log probabilities. | Version the policy record contract; continuous log densities can be positive. |
| Execution metadata | `artifacts/model_config.py` reconstructs execution using training-objective fields; the factory derives attention behavior from an objective. | Separate persisted forward semantics from the objective used to update parameters. |

Primary implementation references: [configuration](../src/sequifier/config/train_config.py), [actor](../src/sequifier/rl/policy.py), [distributions](../src/sequifier/rl/distributions.py), [decoders](../src/sequifier/model/decoders.py), [network](../src/sequifier/model/network.py), [critics](../src/sequifier/rl/models.py), [artifact restoration](../src/sequifier/artifacts/model_config.py), and [trajectory validation](../src/sequifier/trajectory/schema.py).

## 3. The shared contract

Use this separation:

```text
Authored model + persisted execution metadata
                   |
          Shared model resolver/factory
                   |
           Named prediction parameters
                   |
       Shared prediction/distribution layer
            /                     \
 Supervised objective        Policy task binding
                                  |
                    Imitation / IQL / PPO / DPO / GRPO

One run coordinator owns phase transitions, artifacts, and recovery.
Each runner owns its algorithm's batches, collection, and update rules.
```

### 3.1 Model execution stays independent of the learner

Persist the forward contract: named inputs, execution schema, normalization and codecs, attention policy, context geometry, decoder branches, target order, output positions, target transforms, parameter dtypes, and parameter sharing. A phase chooses an objective; it does not implicitly replace those forward semantics.

Derive legacy forward behavior from existing objective metadata during artifact migration. In particular, do not silently turn every imported model into the current `training_objective="policy"` execution configuration. A bidirectional encoder over an entirely observed history is usable as an actor. Reconstructing old actions with later observations in that encoder would leak information; evaluate each historical decision using its actual observation boundary. BERT-specific masking and next-occurrence losses remain supervised data/objective behavior, not mandatory policy behavior.

Keep `ComposableTransformerNetwork`, `ModelOutput`, and the existing named tensor boundary. Preserve the public `logits` field for compatibility even though its real-valued entries are prediction parameters. Add optional distribution metadata and parameter access without an immediate naming migration.

### 3.2 Prediction distributions are reusable model capabilities

Resolve a prediction specification for each target from existing interface metadata plus optional distribution settings. Types and categorical maps remain authoritative in the existing model metadata; policy configuration references targets and adds execution semantics rather than redefining them.

For ordinary categorical heads, use categorical logits over persisted decoder IDs. For real heads, the existing scalar output is the location of a Gaussian by default. Supply an explicit positive exploration scale when attaching a distribution to a legacy point predictor; there is no universally appropriate scale across physical units. Initially support fixed per-target scales. Add learned global or state-dependent scales as optional model modules, with positive floors and stable parameterization. They must be owned and serialized by the network, not hidden inside an optimizer or inference wrapper.

Preserve MSE, MAE, and other existing supervised losses. Adding a policy distribution must not reinterpret all historical supervised training as likelihood training. Gaussian NLL and other distribution-aware supervised losses become additional objectives; imitation is a weighted joint NLL objective using the same prediction contract. Class weights and supervised loss weights are loss reductions, not action-probability definitions. PPO ratios and DPO sequence scores use the true joint likelihood rather than a weighted average of head scores.

Real domains are explicit:

- Unbounded real: Gaussian with identity physical transform.
- Positive or finitely bounded real: a declared invertible transform or a support-aware family, with stable inverse and log-Jacobian calculations. Never clip a Gaussian sample and score it as an unclipped Gaussian.
- Exact endpoint actions or deterministic conditional fields: a specified atom/mixture or conditional point mass. An open-interval squashing transform cannot represent exact endpoints. Do not silently move recorded actions off a boundary.

Freeze model-to-physical transforms across collection, replay, actor, and reference. Persist units and whether a scale is specified in model or physical coordinates. Newly imposed bounds may be incompatible with the pretrained point function; disclose this as adaptation rather than claiming exact preservation.

### 3.3 Policy binding describes a task, not another model

Introduce a `PolicyBinding` associated with a named model interface. It specifies:

- Which targets and output positions form one environment decision.
- Input producers: observed fields, static values, named structured/media payloads, explicitly lagged action fields, and optional desired returns.
- Categorical executable support, real domains, and physical transforms.
- Joint factorization, target order, conditional activity, and legality rules.
- How predictions or sampled actions are fed back into future observations.
- How a decision or explicitly executed action block maps to environment time and rewards.

Default transfer behavior preserves the source input schema. Previous actions and desired returns are optional bindings; do not append them automatically. If a source already contains the prior target as an input, bind that existing input with the correct lag and codec. For newly authored policies, the current observation/previous-action convention remains a convenient explicit preset.

Retain all model interfaces. Select a policy interface per phase and retain inactive interfaces and tied parameters in the artifact. Optimizers follow explicit trainable/frozen parameter selection. Updating a shared backbone can change predictions through other interfaces; freezing only their decoders does not preserve their outputs. Offer supervised replay or regularization when retaining those behaviors matters.

### 3.4 Typed actions and joint probability

Represent actions internally as named tensors with dtypes, event shape, and explicit validity/activity masks. Preserve field order in the resolved schema; never depend on unordered dictionary iteration for likelihood factorization. Support arbitrary numbers of the real and categorical scalar heads currently available to supervised learning. Preserve event axes in the contract so adding true vector heads later does not require another buffer redesign.

Provide a distribution execution interface along these lines:

```python
sample(inputs, generator) -> ActionSample
deterministic(inputs) -> ActionSample
log_prob(inputs, recorded_action, context) -> DecisionScores
entropy(inputs, estimator_config) -> EntropyEstimate
kl(inputs, reference, estimator_config) -> KLEstimate
```

`ActionSample` carries model-space values, executed values, activity/support context, joint log probability, and optional latent values for numerically stable transformed-density replay. `DecisionScores` exposes one log probability per decision and optional per-field diagnostics. Keep the batch/decision axes; reduce action event axes exactly once. Use explicit capabilities for exact versus estimated entropy and KL.

For independent branches, joint log probability is the sum over active fields. For an autoregressive branch, it is the sum of conditional log probabilities given the actual earlier sampled or recorded fields. Composite decoders retain their existing branch factorization. Do not flatten their architecture into one independent output layer.

### 3.5 Autoregressive and hashed targets are required for parity

Extend the existing autoregressive decoder with distribution-driven stochastic generation. Feed each sampled real or categorical value through the same value embedding used for teacher forcing. Scoring a recorded tuple must teacher-force its true prefix. Using logits conditioned on greedy prefixes produces the wrong joint likelihood. Share transformer, projection, embeddings, tied weights, and target ordering with supervised execution.

When a selected action field depends on a non-action predecessor in an autoregressive branch, require a valid observed predecessor binding, or include the predecessor in the sampled probabilistic decision. Do not omit its likelihood while pretending a marginal probability has been computed. A general marginal over unobserved predecessors is additional inference work.

For hashed categorical targets, reuse the existing collision-checked codebook and candidate scores. Normalize over the legal canonical action candidates with `logsumexp`. The product of hash-component probabilities is generally not already normalized over that candidate subset. Sample and score the same canonical candidates; do not sample arbitrary hash tuples and resolve them to a different action. Exact chunked normalization can control memory without changing semantics; approximate retrieval is a separate, explicitly limited mode. Keep supervised component losses intact.

For dependent distributions, joint entropy requires an expectation over prefixes. Summing conditional entropies at an arbitrary recorded prefix is not generally joint entropy. Specify and validate any sampling-based estimator and its gradient, including real sampled prefixes, before enabling an entropy bonus. Preserve the distinction between a sampled reference penalty and exact KL; do not relabel the existing GRPO penalty as universally exact.

## 4. Supervised artifact ingestion

Add a policy attachment path to the common artifact loader:

1. Restore the complete source network and resolved execution contract using the ordinary artifact machinery, without requiring `interface.policy`.
2. Select the source interface and resolve explicit action/input bindings against it. The artifact is authoritative for architecture, codecs, target ordering, preprocessing, and existing weights. If a run also supplies these declarations, validate agreement instead of building an unrelated new model.
3. Attach distributions to existing target outputs. Reuse categorical heads and real location heads. Initialize only declared additions, such as scale parameters, with reproducible seeds.
4. Load all original model parameters and buffers strictly. Permit only an enumerated set of newly introduced keys; do not use broad `strict=False` loading to hide mismatches. Verify tied/shared parameter identity as well as tensor values.
5. Emit a transfer report listing reused and new parameters, bindings, transforms, support restrictions, precision conversions, and any changed execution semantics. Compare source and attached-model prediction parameters on fixed inputs before training.
6. Export a normal model artifact containing the optional binding and distribution contract. Ordinary forecast inference and policy inference select their respective views of the same network.

Distinguish two transfer modes:

| Mode | Guarantee | Examples |
| --- | --- | --- |
| Attach | Original prediction function is retained, within declared numerical tolerance. Stochastic sampling is newly defined where necessary. | Existing real output becomes a Gaussian location with fixed scale; existing logits parameterize a categorical policy. |
| Adapt | Compatible weights are retained, but function equivalence is not promised. Every changed module or semantic choice is reported. | Adding action-history inputs, replacing an unusable input, changing attention visibility, restricting executable categories, imposing incompatible bounds, or changing context geometry. |

Exact categorical transfer requires the same executable decoder support. Filtering special tokens changes the distribution even if logits are unchanged. For unbounded Gaussian heads with the original physical transform, the deterministic location reproduces the old point prediction. For an autoregressive policy, deterministic sequential predictions reproduce the old greedy path when support and transforms are unchanged; this does not imply that the greedy tuple is the global joint mode.

Preserve source normalization by default. Never refit it merely because new policy episodes were supplied. Real action normalization must be reversible and shared by likelihood evaluation, critic action encoding, execution, and history. Sampling and scoring must remain consistent when dtypes are converted; replace the present unconditional FP32-weights rejection with an explicit precision policy while retaining stable probability arithmetic and deterministic compatibility checks.

Initialization is distinct from exact resume. Fresh policy phases create new optimizer, critic, reference, sampler, and collector state according to the phase plan. Exact resume requires the complete run checkpoint and matching task/data/environment/distribution identities. Existing extraction of model state from supported run checkpoints should feed the same initialization path.

## 5. Input, data, and time semantics

Build a shared input materializer from the supervised preprocessing, execution schema, and window resolver. It accepts named raw values or explicitly model-encoded tensors and applies each transform exactly once. Policy and supervised source adapters call it instead of maintaining separate normalization or scalar-only encoding rules.

Generalize history and episode representations to carry named previous-action fields and the full input payload required by the selected interface. Reuse existing depth/image/static tensor and mask contracts. Extend environment observations, snapshots, datasets, collation, memory accounting, and archival records accordingly. A symbolic media reference is sufficient only if replay can recover the identical content and transforms; persist payload identities or immutable materializations. Do not push images or depth arrays through `DomainValue` scalar validators.

Keep source adapters distinct for supervised windows, episodes/transitions, preferences, and online collection. Share codecs, feature materialization, masks, and batching helpers. Rewards and terminal semantics cannot be inferred from arbitrary supervised windows. Require an explicit source mapping for them and preserve episode-level train/validation separation.

Separate three masks: temporal padding, observed training labels, and semantically active action fields. Missing labels are not inactive actions. A missing predecessor in an autoregressive tuple cannot simply be masked out while treating subsequent terms as a complete joint likelihood. For likelihood-based phases, require complete relevant actions unless a supported marginalization method is explicitly selected.

Remove the assumption that every policy decision is the last position of a `support=1` decoder. Preserve the source decoder's receptive field, padding requirements, and output-position semantics. The default binding selects the declared current decision position and leaves other outputs available for supervised losses. Existing `prediction_length` identifies output positions; it must not automatically become a future action chunk.

Action blocks are an explicit extension: define each slot's execution time, whether intermediate observations are consumed, joint likelihood, interruption semantics, and reward/bootstrap aggregation before supporting them. For IQL, a multi-step action block needs matching offline transitions and duration-aware discounting. Selecting one current action from a model that emits several positions does not require block execution.

The same availability check applies to bidirectional models and overlapping offline windows: each scored decision sees only what the policy would actually receive then, including any supplied masked-query token, static payload, or return-conditioning value. Return-conditioned actors remain loadable, but non-imitation phases need an explicit rule supplying and updating that input; do not silently remove the trained return feature.

## 6. Algorithm changes

| Algorithm | Required work | Acceptance condition |
| --- | --- | --- |
| Imitation | Use shared joint NLL over typed actions, with explicit decision reductions and activity masks. Reuse the objective in supervised training when likelihood training is requested. | Independent mixed-head NLL matches a manual sum; autoregressive NLL matches recorded-prefix scoring. Existing MSE/CE training remains valid. |
| PPO | Replace direct `CategoricalActionDistribution` construction in collection/update with distribution execution. Store exact action values, old joint log probability, support/activity context, and required transform latents. Keep V state-only. | Re-scoring a frozen collected actor gives ratio 1 within tolerance for every supported factorization and domain. No sampling/scoring mismatch at bounds. |
| GRPO | Apply the same action and score contract throughout group collection, buffers, reference scoring, and replay. Preserve group-start capabilities and trajectory-versus-decision weighting. | Old and reference scores replay consistently; equal actor/reference produces zero reference penalty; mixed actions execute in the environment. |
| DPO | Sum decision joint log likelihoods on preferred/rejected spans for actor and frozen reference. Persist identical data transforms, binding, support, and conditional masks. | Actor/reference sequence log ratios match manual mixed categorical/density calculations; no category-only or nonpositive-score assumptions remain. |
| IQL | Replace the single categorical action embedding with a schema-driven action encoder: categorical embeddings, transformed real projections, and explicit activity indicators, fused with the state representation. Retain scalar twin Q(h,a), V(h), target networks, and weighted actor log likelihood. Generalize state inputs and pooling. | Every action field reaches Q with finite gradients; Bellman/expectile/actor updates work on mixed actions; actor loss uses the same likelihood as imitation. |

Critics can continue to own independent copies of the supported input/backbone architecture initially. Choose pooling through a declared state-representation adapter rather than hard-coding the last unmasked slot for every layout. Reusing the actor encoder is a later efficiency choice with explicit gradient and optimizer ownership; it is not necessary for unification.

Begin continuous imitation, DPO, and weighted IQL actor fitting with fixed scales. Unrestricted learned scale can distort preference or weighted-likelihood fitting, and variance can collapse or expand without improving the intended behavior. Introduce learned-scale modes with explicit bounds/regularization and diagnostic acceptance criteria. PPO/GRPO can support controlled learned exploration once probability replay is established. Reward improvement is a benchmark goal, not a universal mathematical guarantee.

## 7. Unified configuration and lifecycle

Introduce one canonical run model with model/artifact initialization, named typed sources, policy bindings, execution settings, recovery settings, and ordered discriminated phases. Preserve algorithm-specific settings and budgets: supervised/imitation epochs, offline optimizer updates, PPO environment steps, and GRPO groups. References, critics, and rollout state remain phase-local.

Migrate existing supervised and policy YAML/API forms into that canonical representation with compatibility adapters. A legacy single-categorical policy expands into a one-field binding with its existing previous-action and optional return inputs. Preserve its categorical IDs, start token, transforms, phase defaults, and scoring behavior. Avoid simultaneous wholesale renaming of public configuration keys.

The coordinator performs whole-plan validation, resolves the model once, preflights task/data/environment capabilities, executes runners, selects best/last handoff state, and owns the common checkpoint envelope. Shared services cover parameter catalogs/freezing, optimizer transactions, scheduling, precision, metrics, RNG, atomic artifacts, and callbacks. Adapt the existing supervised and policy services incrementally; do not force RL collection and multiple optimizer roles into an epoch loop.

Use an explicit capability matrix for execution: model architecture parity must not be confused with immediate parity across DDP/FSDP, compilation, integrations, or every device/dtype combination. Preserve existing supervised execution modes while adding policy capabilities with targeted validation. Distribute offline learners before attempting distributed environment collection if distributed policy execution is in scope.

Each phase declares trainable modules and optimizer roles, sources, evaluation and monitor direction, budget units, and handoff behavior. A supervised phase can interleave its current datasets as before. Sequential supervised → RL → supervised phases are part of this plan; arbitrary simultaneous PPO plus supervised replay requires an explicit auxiliary-objective/update policy and is a subsequent extension, not an implicit meaning of interleaving phases.

Default to fresh optimizer state at algorithm boundaries. Any continuation must name a compatible parameter/state mapping. Fresh policy references are captured at the specified phase boundary or loaded from a compatible artifact. Best/last selection restores associated actor distribution parameters and transforms, not only backbone weights.

The checkpoint envelope contains model and binding identity, phase index, committed optimizer clocks, per-role optimizer/scheduler/scaler state, references/targets, sampler/RNG state, source fingerprints, rollout state, environment snapshot identity, and collection generation. Exact resume is guaranteed only where the environment exposes the necessary snapshot capabilities; reset-environment recovery remains a distinct mode. Rejected/stale buffers must not be consumed after a handoff or distribution change.

## 8. Delivery sequence and completion gates

These are implementation slices, not duration estimates. Several are substantial changes and should be split further when the first prototype exposes integration costs.

| Slice | Deliverable and principal locations | Completion gate |
| --- | --- | --- |
| 1. Contracts and baselines | Inventory the currently admitted model combinations; specify forward semantics, prediction distributions, action bindings, and transfer reports. Start in `config`, `model`, and `artifacts`. | Representative supervised and legacy-policy fixtures establish current behavior; every supervised feature has a parity acceptance case. |
| 2. Minimal end-to-end transfer | Shared artifact attachment and a thin policy runtime for an unchanged ordinary supervised interface, one categorical target or one unbounded real location with fixed scale. Touch `rl/checkpoint.py`, `learner.py`, `policy.py`, and distribution resolution. | A real-only supervised artifact with no policy metadata completes a policy-imitation and PPO update without replacing its input or location head. A categorical artifact does the same. |
| 3. Mixed actions throughout the pipeline | Named action tensors, independent multiple heads, reversible transforms, datasets, history, trajectory schema, collectors, buffers, inference, and scalar environment execution. Add joint NLL. | A categorical-plus-two-real actor completes imitation and PPO collection/update, persists, reloads, and reproduces collection scores. Legacy single-categorical behavior still matches. |
| 4. Existing algorithm migration | Generalize GRPO and DPO sequence scoring and IQL action encoding against the established action contract. | All five algorithms train mixed-action fixtures; no algorithm directly instantiates a categorical-only distribution. |
| 5. Decoder architecture parity | Add stochastic autoregressive generation, recorded-prefix scoring, branch composition, tied heads, hashed candidate normalization, and conditional activity. | Sampled and teacher-forced tuple scores agree, normalization checks pass, and original decoder sharing survives reload. |
| 6. Input and geometry parity | Reuse the full supervised materializer and execution schema across policy data, environment observations, history, critics, and recovery. Remove scalar-only, one-interface, and unit-support restrictions after their acceptance cases pass. | Structured/depth/image/static/composite input fixtures, hashed inputs, multiple interfaces, nontrivial support, prediction-position selection, and observed-history bidirectional models train as actors. |
| 7. Common run plan | Compatibility loaders and coordinator around existing runners; then consolidate optimizer/metrics/checkpoint services. Touch `train.py`, `training/engine.py`, `rl/learner.py`, runtime and recovery modules. | A single supervised → PPO → supervised plan works, and interrupted phase transitions resume without duplicate updates or reinitialization. |
| 8. Compatibility and release | Versioned artifacts, trajectories and environment payload capabilities; public API/templates/docs; targeted execution-mode validation. | Legacy artifacts/configurations/records load through tested adapters; unsupported recovery versions fail explicitly; all mandatory parity cases pass. |

Dependencies: slices 2–4 establish a usable supervised-to-RL path before broad lifecycle work. Decoder and input parity both build on the shared contracts and mixed-action representation. Slice 7 can begin with the compatibility schema earlier, but it is not a prerequisite for exporting a supervised artifact and post-training it in a separate run. The release is not architecture-complete until slices 5 and 6 are finished; “multiple independent scalar heads” is an intermediate milestone.

## 9. Validation required for completion

Use small deterministic fixtures for probability and recovery invariants, then a few controlled learning tasks. Do not rely solely on whether training losses decrease.

1. **Transfer invariants:** strict reuse of source parameters/buffers and sharing; fixed-input predictions match before updates; explicit scale additions; unchanged normalization and categorical maps; artifact save/reload preserves both inference views. Include a model with no policy metadata and a model extracted from a supported supervised run checkpoint.
2. **Probability correctness:** manual independent sums; tiny enumerated autoregressive categorical joint normalization; numerical continuous density/transform checks; positive log-density acceptance; Gaussian location/scale gradients; hashed candidate normalization; boundary and conditional-atom behavior; padding/activity masks and missing-label rejection.
3. **Collection/replay equivalence:** seeded sample reproducibility on supported devices; sampled-prefix versus recorded-prefix scores; PPO/GRPO initial ratio 1; reference equality; immutable support and preprocessing; action-history lag; full-precision archived action/latent round trips.
4. **Architecture matrix:** every currently supported ingestion and decoder family plus feature combinations that change execution, including static prefixes, depth/media masks, parameter tying, multiple interfaces, support > 1, and multiple prediction positions. Test full likelihood and a gradient update, not just network construction. Cover ordinary/tabular and non-scalar inputs in critic execution as well.
5. **Algorithm behavior:** mixed imitation improves held-out likelihood; a deterministic toy mixed-action control task demonstrates PPO/GRPO updates and executable actions; DPO improves preferred relative log ratios; IQL has finite meaningful Q/V and actor updates without an action-count vector assumption. Keep reward and likelihood metrics separate.
6. **Lifecycle and compatibility:** supervised → RL → supervised handoff; stop/resume at committed updates and phase boundaries; best/last selection including scales; stale-buffer rejection; source/environment identity changes; legacy categorical config/artifact/trajectory adapters; explicit limits on exact recovery of old online checkpoints.

Baseline performance checks should measure autoregressive sampling cost, hashed candidate scoring cost, and buffer bytes for structured inputs. Correctness is mandatory; use those results to prioritize caching or packing without changing the probability contract.

## 10. Recommended scope decisions

Proceed with these defaults unless a task explicitly requires something else:

- Preserve pretrained interfaces and preprocessing on attachment; do not inject new history features by default.
- Use existing real heads as locations with explicitly configured fixed scales first. Add learnable scales without changing existing point-prediction APIs.
- Deliver independent mixed heads as the first broad action-space milestone, followed by the existing autoregressive and composite architecture surface before claiming parity.
- Support current-decision output selection first while preserving larger decoder support/prediction geometry. Add true action-block execution under a separate temporal contract.
- Represent conditional activity and deterministic values in the action contract now; implement their explicit probability semantics before advertising dependent constrained actions. General user-defined constraint solvers and arbitrary rejection/projection sampling are deferred.
- Keep algorithms and source adapters distinct. Consolidate their shared execution and lifecycle services around the contracts they actually share.
- Require a usable task binding and rewards/preferences/interaction source. Artifact ingestion makes a model trainable; it cannot manufacture the learning task.

The first implementation should prove an unchanged supervised real-valued actor can be attached, sampled, scored, and updated by PPO. That exercises both overarching requirements immediately and prevents the work from stopping at a more generic policy schema that still cannot ingest an ordinary supervised checkpoint.
