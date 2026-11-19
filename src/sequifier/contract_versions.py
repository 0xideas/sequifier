"""Current versions scoped to individual Sequifier contracts."""


class ContractVersions:
    # Checkpoint envelope and runner-owned payloads.
    RUN_CHECKPOINT = 100
    SUPERVISED_RECOVERY = 140
    POLICY_RECOVERY = 141
    POLICY_PHASE_STATE = 103
    LEARNING_RUN_STATE = 142

    # Portable artifacts and training telemetry.
    MODEL_ARTIFACT = 105
    BACKBONE_MANIFEST = 106
    PHASE_REPORT = 107
    RUN_MANIFEST = 108
    METRICS_SCHEMA = 109

    INITIALIZATION_POLICY = 110
    INITIALIZATION_SEED_DERIVATION = 111

    # Dataset storage and preprocessing resume records.
    PT_TENSOR_BATCH = 112
    DATASET_TENSOR_METADATA = 114
    DEPTH_TENSOR_METADATA = 115
    STORED_WINDOW_LAYOUT = 115
    PREPROCESSING_MANIFEST = 116
    VIDEO_FRAME_INDEX = 117
    SERIAL_DEPTH_LEDGER = 119
    PARALLEL_DEPTH_LEDGER = 120

    # Rollout records, algorithm archives, and offline source manifests.
    FORECAST_TRAJECTORY = 120
    JOINT_POLICY_TRAJECTORY = 121
    COLLECTION_ARCHIVE = 122
    PPO_COLLECTION = 123
    GRPO_ARCHIVE = 124
    SEQUENCE_DATA_MANIFEST = 145
    PREFERENCE_PAIR_MANIFEST = 126
    SCALAR_POLICY_TRAJECTORY = 138
    TRAJECTORY_SOURCE_MANIFEST = 140  # Authored manifest of episode files.

    FORECAST_ROLLOUT_ADAPTER = 128
    ONLINE_POLICY_ADAPTER = 129
    MODEL_EXECUTION_SCHEMA = 129
    EXPORT_PREFLIGHT = 130

    # Algorithm and implementation identities used in fingerprints and seeds.
    HASH_CODEC = 131
    RESERVED_CODE_SPECIAL_TOKENS = 132
    INJECTIVE_SPECIAL_TOKENS = 133
    POLICY_COLLECTION_SEED = 134
    EXPORT_ATTENTION_LOWERING = 135
    CATEGORICAL_CONTROL = 136
    TOY_CONTROL = 137

    PHASE_COMMIT = 143
    PHASE_BOUNDARY = 144

    # External ONNX operator set.
    ONNX_OPSET = 18


FORECAST_ROLLOUT_PROTOCOL = "sequifier.forecast-rollout-adapter"
ONLINE_POLICY_PROTOCOL = "sequifier.online-policy-adapter"


def require_contract_version(contract: str, actual, expected: int) -> None:
    if not isinstance(actual, int) or isinstance(actual, bool) or actual != expected:
        raise ValueError(
            f"Unsupported {contract} version {actual!r}, expected {expected}"
        )
