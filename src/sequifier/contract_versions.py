"""Sequifier's global allocation of persisted contract and protocol versions."""


class ContractVersions:
    HISTORICAL_MIN = 0
    HISTORICAL_MAX = 99
    RETIRED = ()

    # Checkpoint envelope and runner-owned payloads.
    RUN_CHECKPOINT = 100
    SUPERVISED_RECOVERY = 101
    POLICY_RECOVERY = 102
    POLICY_PHASE_STATE = 103
    LEARNING_RUN_STATE = 104

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
    DATASET_TENSOR_METADATA = 113
    DEPTH_TENSOR_METADATA = 114
    STORED_WINDOW_LAYOUT = 115
    PREPROCESSING_MANIFEST = 116
    VIDEO_FRAME_INDEX = 117
    SERIAL_DEPTH_LEDGER = 118
    PARALLEL_DEPTH_LEDGER = 119

    # Rollout records, algorithm archives, and offline source manifests.
    FORECAST_TRAJECTORY = 120
    JOINT_POLICY_TRAJECTORY = 121
    COLLECTION_ARCHIVE = 122
    PPO_COLLECTION = 123
    GRPO_ARCHIVE = 124
    SEQUENCE_DATA_MANIFEST = 125
    PREFERENCE_PAIR_MANIFEST = 126
    SCALAR_POLICY_TRAJECTORY = 138
    TRAJECTORY_SOURCE_MANIFEST = 139  # Authored manifest of episode files.

    FORECAST_ROLLOUT_ADAPTER = 127
    ONLINE_POLICY_ADAPTER = 128
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

    # External ONNX operator set, outside the Sequifier allocation.
    ONNX_OPSET = 18


def require_contract_version(
    contract: str, actual, expected: int | tuple[int, ...]
) -> None:
    allowed = (expected,) if isinstance(expected, int) else expected
    if not isinstance(actual, int) or isinstance(actual, bool) or actual not in allowed:
        raise ValueError(
            f"Unsupported {contract} version {actual!r}, expected {expected}"
        )
