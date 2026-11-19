"""Deterministic identities for JSON-compatible content."""

import hashlib
import json


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
