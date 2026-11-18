# Consolidated Sequifier contract versions

[ContractVersions](../src/sequifier/contract_versions.py) is the source of current
IDs for Sequifier artifacts, metadata, adapter protocols, and implementation
identifiers. Checkpoint and collection archive envelopes have their own IDs, as
do the payloads they contain.

- Re-run preprocessing from raw data to rebuild dataset metadata, PT batches,
  stored-window metadata, preprocessing manifests, depth ledgers, and video
  frame indexes.
- Start a fresh training run. Pre-cutover checkpoints cannot be resumed. Use the
  previous Sequifier release for any run or portable model you still need to
  load from that release. Produce fresh portable models, backbone repository
  revisions, reports, metrics, and ONNX exports for the current release.
- Re-export sequence data and regenerate trajectories and collection archives.
  Update authored preference-pair and trajectory-source manifests to declare
  the current version and point to current source artifacts.
- Update external forecast adapters to declare
  `ContractVersions.FORECAST_ROLLOUT_ADAPTER` and external online policy
  adapters to declare `ContractVersions.ONLINE_POLICY_ADAPTER`. Use the
  current implementation IDs in configurations that select the built-in or toy
  control.
- Initialization and policy collection use new seed namespaces. The same inputs
  are deterministic within the new version and can produce different seeds from
  earlier releases. Versioned identities change fingerprints and export
  preflight cache keys. The ONNX operator set remains 18.

To change a contract, allocate an ID above all active and retired IDs on the
target branch, retire the previous ID in the same change, and update the affected
boundary checks. For a pure rename, record the old and new names in
`tools/check_contract_versions.py`. Bump an envelope only when its own contract
changes. Require the Contract versions check on an up-to-date merge result,
using the merge queue when enabled.
