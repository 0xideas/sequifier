"""Collection archive envelopes with algorithm-owned payloads."""

import json
from pathlib import Path


def collection_archive(algorithm, payload):
    return {
        "format": "sequifier.collection-archive",
        "version": 1,
        "algorithm": algorithm,
        "collection": payload,
    }


def read_collection_archive(path, algorithm=None):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    legacy_formats = {
        "ppo": "sequifier.ppo-collection",
        "grpo": "sequifier.grpo-archive",
    }
    if isinstance(payload, dict) and payload.get("format") in legacy_formats.values():
        if algorithm is not None and payload["format"] != legacy_formats.get(algorithm):
            raise ValueError(f"Expected {algorithm} collection archive")
        return payload
    if payload.get("format") == "sequifier.collection-archive":
        if payload.get("version") != 1:
            raise ValueError("Unsupported collection archive envelope version")
        if algorithm is not None and payload.get("algorithm") != algorithm:
            raise ValueError(f"Expected {algorithm} collection archive")
        if payload.get("algorithm") not in {"ppo", "grpo"} or not isinstance(
            payload.get("collection"), dict
        ):
            raise ValueError(
                "Collection archive requires a known algorithm and collection mapping"
            )
        return payload["collection"]
    raise ValueError("Expected a Sequifier collection archive envelope")
