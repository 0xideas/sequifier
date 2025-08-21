from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset

from sequifier.config.train_config import TrainModel
from sequifier.helpers import PANDAS_TO_TORCH_TYPES, numpy_to_pytorch, read_data


class SequifierDatasetFromFile(Dataset):
    """
    A PyTorch Dataset that pre-loads and pre-converts the entire dataset into
    large PyTorch tensors in CPU RAM during initialization.

    This approach mirrors the original data loading logic but encapsulates it
    within a standard Dataset class. It offers the fastest possible per-epoch
    batch creation by simply slicing the already-prepared tensors.

    **Trade-offs:**
    - 👍 **Pros:** Maximum training loop speed, as there is virtually no CPU
              overhead for batch creation.
    - 👎 **Cons:** Very high peak RAM usage (~2x dataset size) and a long,
              blocking initialization time before training can begin.
    """

    def __init__(self, data_path: str, config: TrainModel):
        """
        Initializes the dataset by performing the entire data conversion upfront.

        Args:
            data_path (str): The path to the data file (e.g., '.parquet').
            config (TrainModel): The training configuration object.
        """
        # --- Initialization: Perform the entire data conversion now ---
        print(
            f"🚀 [Dataset] Pre-loading and converting entire dataset from '{data_path}' to tensors..."
        )

        # 1. Load the raw data file into a Polars DataFrame in RAM.
        data_df = read_data(data_path, config.read_format)

        column_types = {
            col: PANDAS_TO_TORCH_TYPES[config.column_types[col]]
            for col in config.column_types
        }
        self.sequences, self.targets = numpy_to_pytorch(
            data=data_df,
            column_types=column_types,
            selected_columns=config.selected_columns,
            target_columns=config.target_columns,
            seq_length=config.seq_length,
            device="cpu",  # Ensure master tensors are created on CPU
            to_device=False,
        )

        # 3. Discard the now-redundant DataFrame to free up half of the peak RAM.
        del data_df

        # Determine the total number of samples from the first tensor.
        first_col = config.selected_columns[0]
        self.n_samples = self.sequences[first_col].shape[0]

        print(
            f"✅ [Dataset] Ready. {self.n_samples} samples fully loaded into CPU RAM."
        )

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.n_samples

    def __getitem__(
        self, idx: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Retrieves a single sample by slicing the pre-loaded tensors.

        This operation is extremely fast as it's a simple indexing operation
        on tensors that are already in memory.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            A tuple containing two dictionaries: one for the input sequences
            and one for the target sequences.
        """
        # Create a dictionary for the input sequences of the requested sample.
        sequence_item = {key: tensor[idx] for key, tensor in self.sequences.items()}

        # Create a dictionary for the target sequences of the requested sample.
        target_item = {key: tensor[idx] for key, tensor in self.targets.items()}

        return sequence_item, target_item
