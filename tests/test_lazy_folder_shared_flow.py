"""PT and Parquet lazy folders use the same rank, worker, and resume order."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
import pytest
import torch

from sequifier.config.depth_layout import DepthLayoutRegistryModel
from sequifier.helpers import ModelWindowView, StoredWindowLayout
from sequifier.io.sequifier_dataset_from_folder_parquet_lazy import (
    SequifierDatasetFromFolderParquetLazy,
)
from sequifier.io.sequifier_dataset_from_folder_pt_lazy import (
    SequifierDatasetFromFolderPtLazy,
)


@pytest.mark.parametrize(
    ("dataset_class", "extension", "worker_info_module"),
    [
        (
            SequifierDatasetFromFolderPtLazy,
            "pt",
            "sequifier.io.sequifier_dataset_from_folder_pt_lazy",
        ),
        (
            SequifierDatasetFromFolderParquetLazy,
            "parquet",
            "sequifier.io.sequifier_dataset_from_folder_parquet_lazy",
        ),
    ],
)
@pytest.mark.parametrize(
    ("rank", "world_size", "expected"),
    [
        (
            1,
            2,
            [
                ([100.0, 101.0], [True, True]),
                ([101.0, 102.0], [False, False]),
                ([102.0, 100.0], [True, False]),
                ([100.0], [False]),
            ],
        ),
        (
            2,
            3,
            [
                ([0.0, 1.0], [False, False]),
                ([4.0, 5.0], [False, False]),
                ([2.0, 3.0], [False, False]),
                ([6.0], [False]),
            ],
        ),
    ],
)
def test_short_rank_worker_resume_preserves_batches_and_masks(
    tmp_path, dataset_class, extension, worker_info_module, rank, world_size, expected
):
    files = []
    for file_id, count in enumerate((7, 3)):
        markers = torch.arange(file_id * 100, file_id * 100 + count)
        path = tmp_path / f"{file_id}.{extension}"
        if extension == "pt":
            zeros = torch.zeros(count, dtype=torch.int64)
            torch.save(
                {
                    "format": "sequifier_tensor_batch",
                    "version": 5,
                    "sequences": {
                        "item": markers.float()[:, None].expand(count, 3).clone()
                    },
                    "metadata": {
                        "depth_valid_masks": {},
                        "sample_positions": None,
                        "curriculum_columns": [],
                    },
                    "sequence_ids": markers,
                    "subsequence_ids": zeros,
                    "start_item_positions": zeros,
                    "left_pad_lengths": zeros,
                    "split_start_item_positions": zeros,
                    "split_end_item_positions": torch.full((count,), 3),
                },
                path,
            )
        else:
            pl.DataFrame(
                {
                    "sequenceId": markers.tolist(),
                    "subsequenceId": [0] * count,
                    "startItemPosition": [0] * count,
                    "leftPadLength": [0] * count,
                    "splitStartItemPosition": [0] * count,
                    "splitEndItemPosition": [3] * count,
                    "inputCol": ["item"] * count,
                    **{str(i): markers.float().tolist() for i in range(2, -1, -1)},
                }
            ).write_parquet(path)
        files.append({"path": path.name, "samples": count})
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "batch_files": files,
                "total_samples": 10,
                "window_length": 3,
                "max_target_offset": 1,
                "stored_window_layout_version": 2,
                "depth_layouts": {},
                "n_classes": {},
            }
        )
    )
    config = SimpleNamespace(
        project_root=str(tmp_path),
        training_spec=SimpleNamespace(batch_size=2, num_workers=2),
        seed=42,
        storage_layout=StoredWindowLayout(
            window_length=3, max_target_offset=1, version=2
        ),
        window_view=ModelWindowView(
            context_length=2, objective="causal", target_offset=1
        ),
        depth_layouts=DepthLayoutRegistryModel(),
        n_classes={},
        column_data_types={"item": "Float32"},
        input_columns=["item"],
        target_columns=["item"],
        training_objective="causal",
    )
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=world_size),
        patch("torch.distributed.get_rank", return_value=rank),
        patch(
            "sequifier.io.folder_dataset.shared_int",
            side_effect=lambda value: torch.tensor(value, dtype=torch.int64),
        ),
    ):
        dataset = dataset_class(str(tmp_path), config, shuffle=False)

    def collect(start_batch):
        dataset.set_start_batch(start_batch)
        per_worker = []
        for worker_id in range(2):
            worker_info = SimpleNamespace(id=worker_id, num_workers=2)
            with patch(
                f"{worker_info_module}.get_worker_info", return_value=worker_info
            ):
                per_worker.append(list(dataset))
        ordered = [
            batch
            for round_index in range(2)
            for worker_batches in per_worker
            for batch in worker_batches[round_index : round_index + 1]
        ]
        return [
            (
                batch.inputs["item"][:, 0].tolist(),
                batch.metadata["sample_valid_mask"].tolist(),
            )
            for batch in ordered
        ]

    full = collect(0)
    assert full == expected
    assert collect(1) == full[1:]
