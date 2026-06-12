import pytest
import torch

from sequifier.io.batch import SequifierBatch


def _metadata():
    valid_mask = torch.ones(1, 2, dtype=torch.bool)
    return {
        "attention_valid_mask": valid_mask,
        "target_valid_mask": valid_mask.clone(),
    }


def test_sequifier_batch_requires_metadata_masks():
    with pytest.raises(ValueError, match="target_valid_mask"):
        SequifierBatch(
            inputs={"item": torch.ones(1, 2)},
            targets={"item": torch.ones(1, 2)},
            metadata={"attention_valid_mask": torch.ones(1, 2, dtype=torch.bool)},
        )


def test_sequifier_batch_does_not_support_tuple_unpacking():
    batch = SequifierBatch(
        inputs={"item": torch.ones(1, 2)},
        targets={"item": torch.ones(1, 2)},
        metadata=_metadata(),
    )

    with pytest.raises(TypeError):
        tuple(batch)  # type: ignore
