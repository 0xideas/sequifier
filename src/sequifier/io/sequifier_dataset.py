# In ./src/sequifier/train.py, near the top with other imports
import polars as pl
import torch
from torch.utils.data import Dataset

from sequifier.helpers import PANDAS_TO_TORCH_TYPES  # noqa: E402
from sequifier.helpers import normalize_path  # noqa: E402
from sequifier.helpers import read_data  # noqa: E402
from sequifier.helpers import subset_to_selected_columns  # noqa: E402


class SequifierDataset(Dataset):
    """Custom PyTorch Dataset for Sequifier data using pre-aggregation."""

    def __init__(self, data_path, read_format, config):
        self.config = config
        self.column_types = {
            col: PANDAS_TO_TORCH_TYPES[config.column_types[col]]
            for col in config.column_types
        }

        # Load the initial data
        data = read_data(normalize_path(data_path, config.project_path), read_format)
        if config.selected_columns is not None:
            data = subset_to_selected_columns(data, config.selected_columns)

        joint_seq_cols = [str(c) for c in range(self.config.seq_length, -1, -1)]

        merged_cols = []
        for col in self.config.selected_columns + self.config.target_columns:
            if col not in merged_cols:
                merged_cols.append(col)

        aggs = []
        for col_name in merged_cols:
            aggs.append(
                pl.col(joint_seq_cols)
                .filter(pl.col("inputCol") == col_name)
                .flatten()
                .alias(col_name)
            )

        self.precomputed_data = (
            data.group_by(["sequenceId", "subsequenceId"])
            .agg(aggs)
            .sort(["sequenceId", "subsequenceId"])
        )
        self.n_samples = len(self.precomputed_data)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample_row = self.precomputed_data[idx]

        sequence = {}
        for col_name in self.config.selected_columns:
            sequence_data = sample_row.get_column(col_name).to_list()[0][:-1]
            sequence[col_name] = torch.tensor(
                sequence_data, dtype=self.column_types[col_name]
            )

        targets = {}
        for col_name in self.config.target_columns:
            target_data = sample_row.get_column(col_name).to_list()[0][1:]
            targets[col_name] = torch.tensor(
                target_data, dtype=self.column_types[col_name]
            )

        return sequence, targets
