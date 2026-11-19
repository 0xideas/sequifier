# Consolidated Sequifier contract versions

Sequifier manages compatibility for generated artifacts, metadata, and adapter
protocols internally. Rollout configs select the adapter by factory.

- Re-run preprocessing from raw data to rebuild dataset metadata, PT batches,
  stored-window metadata, preprocessing manifests, depth ledgers, and video
  frame indexes.
- Start a fresh training run. Pre-cutover checkpoints cannot be resumed. Use the
  previous Sequifier release for any run or portable model you still need to
  load from that release. Produce fresh portable models, backbone repository
  revisions, reports, metrics, and ONNX exports for the current release.
- Re-export sequence data and regenerate trajectories and collection archives.
  Authored preference-pair and trajectory-source manifests use a format tag.
- Custom adapters can inherit protocol metadata from
  `ForecastRolloutAdapterBase` or `OnlineEnvironmentAdapterBase`.
- Initialization and policy collection use new seed namespaces. The same inputs
  are deterministic within the new version and can produce different seeds from
  earlier releases. Versioned identities change fingerprints and export
  preflight cache keys.
