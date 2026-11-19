"""Collection archive envelopes with algorithm-owned payloads."""

import json
from pathlib import Path

from sequifier.contract_versions import ContractVersions, require_contract_version


def collection_archive(algorithm, payload):
    return {
        "format": "sequifier.collection-archive",
        "version": ContractVersions.COLLECTION_ARCHIVE,
        "algorithm": algorithm,
        "collection": payload,
    }


def read_collection_archive(path, algorithm=None):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    require_contract_version(
        "collection archive envelope",
        payload.get("version") if isinstance(payload, dict) else None,
        ContractVersions.COLLECTION_ARCHIVE,
    )
    if payload.get("format") != "sequifier.collection-archive":
        raise ValueError("Expected a Sequifier collection archive envelope")
    actual = payload.get("algorithm")
    if algorithm is not None and actual != algorithm:
        raise ValueError(f"Expected {algorithm} collection archive")
    contracts = {
        "ppo": ("sequifier.ppo-collection", ContractVersions.PPO_COLLECTION),
        "grpo": ("sequifier.grpo-archive", ContractVersions.GRPO_ARCHIVE),
    }
    collection = payload.get("collection")
    if (
        not isinstance(actual, str)
        or actual not in contracts
        or not isinstance(collection, dict)
    ):
        raise ValueError(
            "Collection archive requires a known algorithm and collection mapping"
        )
    kind, version = contracts[actual]
    require_contract_version(kind, collection.get("version"), version)
    if collection.get("format") != kind:
        raise ValueError(f"Expected {kind} collection payload")
    return collection
