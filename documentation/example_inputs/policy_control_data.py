"""Generate curated sequence preferences for the multi-phase control examples.

Usage: python documentation/example_inputs/policy_control_data.py data
Preferences are supplied by a sign-matching rule, independent of reward sums.
This script only creates example data; it does not train or evaluate a model.
"""

import csv
import json
import sys
from pathlib import Path


def generate(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pairs = []
    with (directory / "control-decisions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "sequenceId",
            "episodeId",
            "contextId",
            "itemPosition",
            "cue",
            "action",
            "reward",
            "terminated",
            "truncated",
            "bootstrap_allowed",
            "next_cue",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for context in range(40):
            initial = (-1 if context % 2 else 1) * (1 + context / 100)
            alternatives = {}
            for kind, length in (("chosen", 8), ("rejected", 6)):
                identity = f"context-{context}-{kind}"
                cue = initial
                for step in range(length):
                    matched = "right" if cue > 0 else "left"
                    action = (
                        matched
                        if kind == "chosen"
                        else ("left" if matched == "right" else "right")
                    )
                    writer.writerow(
                        {
                            "sequenceId": identity,
                            "episodeId": identity,
                            "contextId": f"context-{context}",
                            "itemPosition": step,
                            "cue": cue,
                            "action": action,
                            "reward": 1 if action == matched else -1,
                            "terminated": step + 1 == 8,
                            "truncated": step + 1 == length and length != 8,
                            "bootstrap_allowed": step + 1 != 8,
                            "next_cue": -cue,
                        }
                    )
                    cue = -cue
                alternatives[kind] = {
                    "source": "decisions.default",
                    "episode_id": identity,
                    "start": 0,
                    "end": length,
                }
            pairs.append(
                {
                    "pair_id": f"pair-{context}",
                    "context_id": f"context-{context}",
                    **alternatives,
                }
            )
    (directory / "control-preferences.json").write_text(
        json.dumps(
            {"format": "sequifier.preference-pairs", "version": 1, "pairs": pairs},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    generate(sys.argv[1] if len(sys.argv) > 1 else "data")
