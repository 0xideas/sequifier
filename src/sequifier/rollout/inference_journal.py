"""Write-ahead reset and step records for a single CLI inference episode."""

import json
from copy import deepcopy
from dataclasses import asdict, is_dataclass

from sequifier.io.atomic import write_json_atomic
from sequifier.io.values import portable_value


class InferenceJournal:
    def __init__(self, directory, episode_id):
        self.directory = directory
        self.episode_id = episode_id
        self.resuming = (directory / "reset.json").exists()

    def decision(self, step, select):
        path = self.directory / f"decision-{step}.json"
        if path.exists():
            return json.loads(path.read_text())
        decision = select()
        write_json_atomic(path, decision)
        return decision

    def execute(self, adapter, step, payload, invoke, result_type):
        """Adapters with external effects durably deduplicate (episode_id, step).

        'reset' and integer step indices have disjoint keys. An opt-in local
        simulator may instead save and restore its entirely private state.
        """
        path = self.directory / f"{step}.json"
        payload = portable_value(payload)
        isolated = getattr(adapter, "inference_isolated_snapshot", False) is True
        recover = getattr(adapter, "inference_operation", None)
        mode = "keyed" if callable(recover) else "snapshot" if isolated else "manual"
        if isolated and not all(
            callable(getattr(adapter, name, None))
            for name in ("snapshot_state", "restore_state")
        ):
            raise ValueError(
                "Isolated inference recovery requires snapshot_state and restore_state"
            )
        if path.exists():
            entry = json.loads(path.read_text())
            if entry["payload"] != payload or entry["recovery"] != mode:
                raise ValueError(
                    f"Inference operation {self.episode_id}/{step} changed"
                )
            if "result" in entry:
                if isolated:
                    adapter.restore_state(entry["snapshot"])
                return result_type(**entry["result"]) if result_type else None
        else:
            entry = {"payload": payload, "recovery": mode}
        if self.resuming and not isolated and not callable(recover):
            raise RuntimeError(
                f"Interrupted inference episode {self.episode_id} requires adapter "
                "recovery before further environment operations"
            )
        write_json_atomic(path, entry)
        if callable(recover):
            result = recover(self.episode_id, step, deepcopy(payload))
        else:
            result = invoke()
        entry["result"] = None
        if result_type is not None:
            if (
                not isinstance(result, result_type)
                or not is_dataclass(result)
                or isinstance(result, type)
            ):
                raise ValueError(
                    f"Expected {result_type.__name__} for inference operation {step}"
                )
            entry["result"] = portable_value(asdict(result))
        if isolated:
            entry["snapshot"] = portable_value(adapter.snapshot_state())
        write_json_atomic(path, entry)
        return result
