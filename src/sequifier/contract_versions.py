"""Current versions scoped to individual Sequifier contracts."""


class ContractVersions:
    # Checkpoint envelope and runner-owned payloads.
    RUN_CHECKPOINT = 100
    SUPERVISED_RECOVERY = 111
    POLICY_RECOVERY = 106
    POLICY_PHASE_STATE = 103
    LEARNING_RUN_STATE = 104

    # Portable artifacts and training telemetry.
    MODEL_ARTIFACT = 103
    BACKBONE_MANIFEST = 100
    PHASE_REPORT = 100
    RUN_MANIFEST = 100
    METRICS_SCHEMA = 100
    CONDENSED_METRICS_SCHEMA = 101
    ITERATION_METRICS_SCHEMA = 102

    INITIALIZATION_POLICY = 100
    INITIALIZATION_SEED_DERIVATION = 100

    # Dataset storage and preprocessing resume records.
    PT_TENSOR_BATCH = 100
    DATASET_TENSOR_METADATA = 101
    DEPTH_TENSOR_METADATA = 101
    STORED_WINDOW_LAYOUT = 100
    INDEXED_SEQUENCES = 104
    PREPROCESSING_MANIFEST = 105
    VIDEO_FRAME_INDEX = 100
    SERIAL_DEPTH_LEDGER = 101
    PARALLEL_DEPTH_LEDGER = 101

    # Rollout records and offline source manifests.
    SEQUENCE_DATA_MANIFEST = 106

    ROLLOUT_SEED_LAYOUT = 101
    FORECAST_ROLLOUT_ADAPTER = 100
    ONLINE_POLICY_ADAPTER = 101
    MODEL_EXECUTION_SCHEMA = 100
    POLICY_ONNX = 100
    EXPORT_PREFLIGHT = 101
    INFERENCE_WORKSPACE = 101
    POLICY_INPUT_LAYOUT = 104
    PROBABILITY_TABLE = 100
    PREDICTION_TABLE = 106

    # Algorithm and implementation identities used in fingerprints and seeds.
    HASH_CODEC = 100
    RESERVED_CODE_SPECIAL_TOKENS = 100
    INJECTIVE_SPECIAL_TOKENS = 100
    POLICY_COLLECTION_SEED = 100
    EXPORT_ATTENTION_LOWERING = 101
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
