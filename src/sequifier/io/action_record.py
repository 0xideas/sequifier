"""Flat record fields shared by policy decisions, trajectory logs and tables."""

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class ActionRecordField:
    action: str
    key: str
    kind: str

    @property
    def missing(self):
        if self.kind == "latent":
            return float("nan")
        if self.kind == "component":
            return -2
        return None


class ActionRecordLayout:
    def __init__(self, actions):
        used = {name for name, kind in actions}
        real = [name for name, kind in actions if kind == "real"]

        def auxiliary_keys(prefix):
            keys = {}
            # Reserve the longest names first when an action resembles a metadata key.
            for name in sorted(real, key=len, reverse=True):
                key = prefix + name
                while key in used:
                    key = prefix + key
                keys[name] = key
                used.add(key)
            return keys

        self.latents = auxiliary_keys("__policy_latent:")
        self.components = auxiliary_keys("__policy_component:")
        self.fields = (
            *(ActionRecordField(name, name, kind) for name, kind in actions),
            *(ActionRecordField(name, self.latents[name], "latent") for name in real),
            *(
                ActionRecordField(name, self.components[name], "component")
                for name in real
            ),
        )
        self.keys = frozenset(used)

    def pack(self, values, latents, components):
        return {
            **values,
            **{self.latents[name]: value for name, value in latents.items()},
            **{self.components[name]: value for name, value in components.items()},
        }

    def unpack_auxiliary(self, record):
        return (
            {name: record[key] for name, key in self.latents.items() if key in record},
            {
                name: record[key]
                for name, key in self.components.items()
                if key in record
            },
        )


@lru_cache(maxsize=128)
def _layout(actions):
    return ActionRecordLayout(actions)


def action_record_layout(config):
    return _layout(tuple((name, field.type) for name, field in config.actions.items()))
