# Integrated policy training and inference (unreleased)

`sequifier train` accepts one trajectory-backed `return_conditioned` phase;
`return_conditioned: false` selects behavior cloning. Policies now use the
standard configurable ingestion, backbone, decoder, and network implementation.
Encoded categorical observations and matching forecast-backbone transfer are
supported. PyTorch execution supports CPU, CUDA, and MPS.

The selected best model is an ordinary `sequifier_model` artifact, usable with
`sequifier infer`, Python policy APIs, and generic model-loading tools. Inference
supports raw CSV/Parquet decisions and environment execution through the shared
rollout coordinator. Each training epoch writes a standard resumable run
checkpoint, including optimizer/scheduler/scaler, random/data-order state,
training progress, and best weights. The older standalone actor format requires
retraining. The default model path is now `models/<name>/<name>.pt`.

File-backed training uses a bounded episode cache, incremental normalization,
and context-sized window construction. Behavior cloning retains observed
state/action pairs from truncated and cutoff episodes without return labels.
Grouped episode splitting and training-only normalization remain in place.

`sequifier make` includes policy train and infer templates. Mixed algorithm
phases, critics, IQL/PPO, distributed policy learning, multimodal trajectories,
and CLI ONNX policy execution remain future work.

No tests or training runs were executed, as requested. Validation used static
lint, formatting, and type checks.
