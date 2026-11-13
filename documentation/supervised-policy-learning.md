# Supervised models and policy learning

A policy binding attaches a task to existing model targets. It preserves interfaces,
weights, codecs, normalization, forward attention and decoder geometry; it does
not automatically add inputs or replace a pretrained decoder. Start from an
ordinary supervised PT artifact or supported run checkpoint:

```yaml
initialize_from: models/supervised/model.pt
policy_bindings:
  default:
    actions:
      mode:
        type: categorical
        values:
        - 'off'
        - 'on'
      price:
        type: real
        scale:
          fixed: 0.5
```

Add datasets/phases as in the [policy guide](policy-learning.md). A separately
declared model must agree with the artifact. Omitted categorical `values` select
the whole decoder vocabulary; an explicit subset is reported as an adaptation.
Real `scale` explicitly selects `model` or `{fixed: ...}` in physical units.
Learned scales belong to the interface's `prediction_distributions`, where
`scale` declares `initial`, `trainable`, `min`, and `max`. `lower`/`upper` use bijections;
`atoms` assigns exact point masses (positive probabilities summing to less than one).
Bounds change the deterministic prediction and count as adaptation. Samples retain
transform latents; external endpoints need declared atoms. `active_when` references
earlier action fields; inactive fields use `inactive_value` with zero likelihood cost.

Ordinary branches factor independently; autoregressive branches retain target
order and score recorded prefixes. Selecting a later autoregressive action requires
all preceding decoder targets. Composite branches and tied weights are retained.
Likelihoods sum active fields; supervised loss weights never redefine probabilities.
Independent categorical/unbounded Gaussian entropy is exact; dependent/transformed
entropy uses a score-function estimator. Deterministic selection follows sequential
argmax/locations, without claiming a global joint mode.

Inputs retain the complete scalar/static/depth/image/mask execution schema.
Prepared sources reuse depth masks and image stores; tables may contain nested
Parquet payloads. Explicit `previous_action_inputs` maps existing input names to
actions; `initial_inputs` supplies the first lagged value using its persisted codec.
`decision_position` defaults to `-1` and selects one existing decoder position,
not an action block. Historical likelihoods use only observations available then.
Return conditioning requires an existing `__policy_return` input and known offline
tails or online `initial_desired_return`.

Supervised and policy phases share `training_plan.phases`; see the
[complete mixed template](../src/sequifier/templates/train-supervised-to-policy.yaml).
Prepared and sequence/preference parts belong to `dataset_training`. Each phase owns
its batching and role optimization (`model` for supervised; policy roles in the guide).
All interfaces survive handoff, though shared backbone updates can affect inactive
interfaces. The coordinator preflights the plan, commits incoming artifacts, and uses
native checkpoints plus atomic completion receipts. Mixed plans execute locally and
reject optimizer continuation; compatible native plans follow the
[continuation rules](configs/train.md#optimization-across-phases).

Existing supervised losses retain their meanings. Real `prediction_distributions`
and `GaussianNLLLoss` opt into the shared distribution implementation; learned scales
belong to the model artifact. `task: forecast` selects point prediction from a
policy-bound artifact; `task: policy` selects policy execution. See
[inference routing](configs/infer.md#policy-model-interfaces).
Model artifacts write version 3 and read versions 1/2/3; joint trajectories and PPO
collection records use version 2. ONNX remains a prediction export.
