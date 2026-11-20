"""Current versions scoped to individual Sequifier contracts."""


class ContractVersions:
    # Checkpoint envelope and runner-owned payloads.
    RUN_CHECKPOINT = 100
    SUPERVISED_RECOVERY = 108
    POLICY_RECOVERY = 103
    POLICY_PHASE_STATE = 100
    LEARNING_RUN_STATE = 104

    # Portable artifacts and training telemetry.
    MODEL_ARTIFACT = 101
    BACKBONE_MANIFEST = 100
    PHASE_REPORT = 100
    RUN_MANIFEST = 100
    METRICS_SCHEMA = 100

    INITIALIZATION_POLICY = 100
    INITIALIZATION_SEED_DERIVATION = 100

    # Dataset storage and preprocessing resume records.
    PT_TENSOR_BATCH = 100
    DATASET_TENSOR_METADATA = 100
    DEPTH_TENSOR_METADATA = 100
    STORED_WINDOW_LAYOUT = 100
    INDEXED_SEQUENCES = 101
    PREPROCESSING_MANIFEST = 101
    VIDEO_FRAME_INDEX = 100
    SERIAL_DEPTH_LEDGER = 101
    PARALLEL_DEPTH_LEDGER = 101

    # Rollout records, algorithm archives, and offline source manifests.
    FORECAST_TRAJECTORY = 100
    JOINT_POLICY_TRAJECTORY = 101
    COLLECTION_ARCHIVE = 100
    PPO_COLLECTION = 101
    GRPO_ARCHIVE = 100
    SEQUENCE_DATA_MANIFEST = 101
    SCALAR_POLICY_TRAJECTORY = 100

    FORECAST_ROLLOUT_ADAPTER = 100
    ONLINE_POLICY_ADAPTER = 100
    MODEL_EXECUTION_SCHEMA = 100
    EXPORT_PREFLIGHT = 100

    # Algorithm and implementation identities used in fingerprints and seeds.
    HASH_CODEC = 100
    RESERVED_CODE_SPECIAL_TOKENS = 100
    INJECTIVE_SPECIAL_TOKENS = 100
    POLICY_COLLECTION_SEED = 100
    EXPORT_ATTENTION_LOWERING = 100
    CATEGORICAL_CONTROL = 100

    PHASE_COMMIT = 101
    PHASE_BOUNDARY = 100

    # External ONNX operator set.
    ONNX_OPSET = 18


FORECAST_ROLLOUT_PROTOCOL = "sequifier.forecast-rollout-adapter"
ONLINE_POLICY_PROTOCOL = "sequifier.online-policy-adapter"


def require_contract_version(contract: str, actual, expected: int) -> None:
    if not isinstance(actual, int) or isinstance(actual, bool) or actual != expected:
        raise ValueError(f"Unsupported {contract} version")
