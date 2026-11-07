"""The explicit sequence import operation of sequifier preprocess."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from sequifier.config.policy_config import PolicyConfigBase, PolicySchemaConfig
from sequifier.config.policy_train_config import SequenceSourceConfig


class SequenceImportConfig(PolicyConfigBase):
    operation: Literal["import_sequences"]
    project_root: str = "."
    source: SequenceSourceConfig
    policy: PolicySchemaConfig
    output_path: str
    episodes_per_shard: int = Field(default=64, gt=0)

    def resolved_source(self) -> SequenceSourceConfig:
        from sequifier.config.policy_train_config import resolve_sequence_source

        return resolve_sequence_source(self.source, Path(self.project_root).resolve())
