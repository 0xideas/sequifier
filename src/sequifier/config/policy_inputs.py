"""Internal policy feature layout shared by fitting and model attachment."""

from dataclasses import dataclass

from sequifier.config.column_names import INCOMING_ACTION_PRESENT
from sequifier.config.metadata import RawInputMetadata
from sequifier.config.policy_config import PolicyBinding, ResolvedPolicyBinding


def observation_columns(temporal, binding):
    return [
        name
        for name in temporal
        if name != INCOMING_ACTION_PRESENT
        and name not in binding.previous_action_inputs
        and (name != "__policy_return" or not binding.return_conditioned)
    ]


@dataclass(frozen=True, kw_only=True)
class PolicyFittingSpec:
    binding: PolicyBinding
    context_length: int
    interface_name: str
    observation_columns: list[str]
    static_columns: list[str]
    raw_inputs: RawInputMetadata | None = None
    raw_mask_column: str | None = None
    categorical_columns: list[str]
    categorical_n_classes: dict[str, int]
    id_maps: dict[str, dict[str | int, int]]

    @classmethod
    def from_metadata(
        cls, interface_name, temporal, static, binding, metadata, context_length
    ):
        observations = observation_columns(temporal, binding)
        features = observations + static
        return cls(
            binding=binding,
            context_length=context_length,
            interface_name=interface_name,
            observation_columns=observations,
            static_columns=static,
            raw_inputs=metadata,
            raw_mask_column=metadata.raw_mask_column,
            categorical_columns=[
                k for k in features if "int" in metadata.column_data_types[k].lower()
            ],
            categorical_n_classes={
                k: metadata.n_classes[k] for k in features if k in metadata.n_classes
            },
            id_maps={k: metadata.id_maps[k] for k in features if k in metadata.id_maps},
        )

    def validate_source_units(self, units):
        self.binding.validate_source_units(units)

    @property
    def previous_action_inputs(self):
        return self.binding.previous_action_inputs

    @property
    def actions(self):
        return self.binding.actions

    @property
    def action_columns(self):
        return list(self.actions)

    @property
    def return_conditioned(self):
        return self.binding.return_conditioned

    @property
    def returns(self):
        return self.binding.returns

    @property
    def input_encoding(self):
        return "raw"


PolicyEpisodeSpec = ResolvedPolicyBinding | PolicyFittingSpec
