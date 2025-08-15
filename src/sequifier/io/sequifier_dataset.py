# In ./src/sequifier/train.py, near the top with other imports
import polars as pl
import torch
from torch.utils.data import Dataset

from sequifier.helpers import PANDAS_TO_TORCH_TYPES  # noqa: E402
from sequifier.helpers import normalize_path  # noqa: E402
from sequifier.helpers import read_data  # noqa: E402
from sequifier.helpers import subset_to_selected_columns  # noqa: E402


class SequifierDataset(Dataset):
    """Custom PyTorch Dataset for Sequifier data."""

    def __init__(self, data_path, read_format, config):
        self.config = config
        self.column_types = {
            col: PANDAS_TO_TORCH_TYPES[config.column_types[col]]
            for col in config.column_types
        }

        # Load the entire dataset but keep it in efficient Polars format
        self.data = read_data(
            normalize_path(data_path, config.project_path), read_format
        )
        if config.selected_columns is not None:
            self.data = subset_to_selected_columns(self.data, config.selected_columns)

        # Pre-calculate the unique samples to determine the dataset length
        self.samples = (
            self.data.group_by(["sequenceId", "subsequenceId"])
            .agg(pl.count())
            .sort(["sequenceId", "subsequenceId"])
        )
        self.n_samples = len(self.samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Get the identifiers for the i-th sample
        sample_ids = self.samples[idx]
        seq_id = sample_ids.get_column("sequenceId").item()
        subseq_id = sample_ids.get_column("subsequenceId").item()

        # Filter the full dataframe to get only the rows for this specific sample
        sample_df = self.data.filter(
            (pl.col("sequenceId") == seq_id) & (pl.col("subsequenceId") == subseq_id)
        )

        # Use a modified version of the original numpy_to_pytorch logic here for a single sample
        X, y = self._convert_sample_to_tensors(sample_df)
        return X, y

    def _convert_sample_to_tensors(self, sample_df: pl.DataFrame):
        """Converts a small Polars DataFrame for one sample into tensors."""
        # This logic is adapted from helpers.numpy_to_pytorch
        targets = {}
        target_seq_cols = [str(c) for c in range(self.config.seq_length - 1, -1, -1)]
        for col in self.config.target_columns:
            targets[col] = torch.tensor(
                sample_df.filter(pl.col("inputCol") == col)
                .select(target_seq_cols)
                .to_numpy(),
                dtype=self.column_types[col],
            ).squeeze(0)  # Squeeze the batch dimension of 1

        sequence = {}
        input_seq_cols = [str(c) for c in range(self.config.seq_length, 0, -1)]
        for col in self.config.selected_columns:
            sequence[col] = torch.tensor(
                sample_df.filter(pl.col("inputCol") == col)
                .select(input_seq_cols)
                .to_numpy(),
                dtype=self.column_types[col],
            ).squeeze(0)  # Squeeze the batch dimension of 1

        return sequence, targets
