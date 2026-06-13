from types import SimpleNamespace

import polars as pl
import torch

from sequifier.helpers import (
    ModelWindowView,
    StoredWindowLayout,
    apply_bert_masking,
    build_valid_mask,
    construct_index_maps,
    numpy_to_pytorch,
    resolve_window_view,
)

# ==========================================
# 1. Test Index Map Construction
# ==========================================


def test_construct_index_maps_string():
    """Tests reversing a string-to-int map, ensuring 0 maps to '[unknown]'."""
    id_maps: dict[str, dict[str | int, int]] = {"itemId": {"apple": 3, "banana": 4}}
    target_cols = ["itemId"]

    result = construct_index_maps(id_maps, target_cols, map_to_id=True)

    # Check standard reversal
    assert result["itemId"][3] == "apple"
    assert result["itemId"][4] == "banana"

    # Check the special 0 index for strings
    assert result["itemId"][0] == "[unknown]"
    assert result["itemId"][1] == "[other]"
    assert result["itemId"][2] == "[mask]"


def test_construct_index_maps_integer():
    """Tests reversing an int-to-int map with special IDs as sentinels."""
    id_maps: dict[str, dict[str | int, int]] = {"storeId": {100: 3, 101: 4}}
    target_cols = ["storeId"]

    result = construct_index_maps(id_maps, target_cols, map_to_id=True)

    assert result["storeId"][3] == 100
    assert result["storeId"][4] == 101

    assert result["storeId"][0] == "[unknown]"
    assert result["storeId"][1] == "[other]"
    assert result["storeId"][2] == "[mask]"


def test_construct_index_maps_flag_false():
    """Tests that nothing is returned if map_to_id is False."""
    id_maps: dict[str, dict[str | int, int]] = {"itemId": {"a": 1}}
    result = construct_index_maps(id_maps, ["itemId"], map_to_id=False)
    assert result == {}


# ==========================================
# 2. Test Numpy to PyTorch Conversion
# ==========================================


def test_numpy_to_pytorch_shapes_and_shifting():
    """
    Tests that DataFrames are correctly converted to input and target Tensors.

    Logic to test:
    If context_length = 3:
    - Input cols:  ['3', '2', '1']
    - Target cols: ['2', '1', '0']
    """

    # Setup: 2 sequences for feature "A"
    # Seq 1: [10, 20, 30, 40] (where 40 is t=0, 30 is t=1, etc.)
    data = pl.DataFrame(
        {
            "sequenceId": [1, 2],
            "subsequenceId": [0, 0],
            "startItemPosition": [0, 0],
            "leftPadLength": [0, 0],
            "inputCol": ["A", "A"],
            "3": [10, 11],
            "2": [20, 21],
            "1": [30, 31],
            "0": [40, 41],
        }
    )

    column_types = {"A": torch.float32}
    all_columns = ["A"]
    resolved_view = resolve_window_view(
        StoredWindowLayout(stored_context_width=4, max_target_offset=1, version=2),
        ModelWindowView(context_length=3, objective="causal", target_offset=1),
    )

    tensors, metadata = numpy_to_pytorch(
        data,
        column_types,
        all_columns,
        resolved_view,
    )

    # 1. Check Keys
    assert "A" in tensors
    assert "A_target" in tensors
    assert torch.equal(
        metadata["attention_valid_mask"],
        torch.tensor([[True, True, True], [True, True, True]]),
    )
    assert torch.equal(
        metadata["target_valid_mask"],
        torch.tensor([[True, True, True], [True, True, True]]),
    )

    # 2. Check Input Tensor (Cols 3, 2, 1)
    # Row 0: [10, 20, 30]
    expected_input = torch.tensor([[10, 20, 30], [11, 21, 31]], dtype=torch.float32)
    assert torch.equal(tensors["A"], expected_input)

    # 3. Check Target Tensor (Cols 2, 1, 0) -> Shifted by 1 step into future
    # Row 0: [20, 30, 40]
    expected_target = torch.tensor([[20, 30, 40], [21, 31, 41]], dtype=torch.float32)
    assert torch.equal(tensors["A_target"], expected_target)


def test_numpy_to_pytorch_dtypes():
    """Tests that column_types dict is respected for tensor dtypes."""

    # Note: numpy_to_pytorch filters by inputCol, so we need to handle how
    # it selects rows. It assumes all rows matching "inputCol" have valid data
    # for that type.

    # We'll test one type at a time to avoid schema conflicts in the Polars DF creation
    # (Polars columns usually have a single type).

    # Case 1: Integer
    data_int = pl.DataFrame(
        {
            "sequenceId": [1],
            "subsequenceId": [0],
            "startItemPosition": [0],
            "leftPadLength": [0],
            "inputCol": ["int_col"],
            "1": [10],
            "0": [20],
        }
    )
    tensors_int, _ = numpy_to_pytorch(
        data_int,
        {"int_col": torch.int64},
        ["int_col"],
        resolve_window_view(
            StoredWindowLayout(stored_context_width=2, max_target_offset=1, version=2),
            ModelWindowView(context_length=1, objective="causal", target_offset=1),
        ),
    )
    assert tensors_int["int_col"].dtype == torch.int64

    # Case 2: Float
    data_float = pl.DataFrame(
        {
            "sequenceId": [1],
            "subsequenceId": [0],
            "startItemPosition": [0],
            "leftPadLength": [0],
            "inputCol": ["float_col"],
            "1": [10.5],
            "0": [20.5],
        }
    )
    tensors_float, _ = numpy_to_pytorch(
        data_float,
        {"float_col": torch.float32},
        ["float_col"],
        resolve_window_view(
            StoredWindowLayout(stored_context_width=2, max_target_offset=1, version=2),
            ModelWindowView(context_length=1, objective="causal", target_offset=1),
        ),
    )
    assert tensors_float["float_col"].dtype == torch.float32


def test_build_valid_mask_from_left_pad_lengths():
    left_pad_lengths = torch.tensor([0, 2, 5], dtype=torch.int64)

    resolved_view = resolve_window_view(
        StoredWindowLayout(stored_context_width=6, max_target_offset=1, version=2),
        ModelWindowView(context_length=5, objective="causal", target_offset=1),
    )
    input_mask = build_valid_mask(
        left_pad_lengths, full_length=6, view_slice=resolved_view.input_slice
    )
    target_mask = build_valid_mask(
        left_pad_lengths, full_length=6, view_slice=resolved_view.target_slice
    )

    assert torch.equal(
        input_mask,
        torch.tensor(
            [
                [True, True, True, True, True],
                [False, False, True, True, True],
                [False, False, False, False, False],
            ]
        ),
    )
    assert torch.equal(
        target_mask,
        torch.tensor(
            [
                [True, True, True, True, True],
                [False, True, True, True, True],
                [False, False, False, False, True],
            ]
        ),
    )


def test_resolve_window_view_slices_tensors():
    tensor = torch.arange(8).reshape(2, 4)
    resolved_view = resolve_window_view(
        StoredWindowLayout(stored_context_width=4, max_target_offset=1, version=2),
        ModelWindowView(context_length=3, objective="causal", target_offset=1),
    )

    assert torch.equal(
        tensor[:, resolved_view.input_slice],
        torch.tensor([[0, 1, 2], [4, 5, 6]]),
    )
    assert torch.equal(
        tensor[:, resolved_view.target_slice],
        torch.tensor([[1, 2, 3], [5, 6, 7]]),
    )


def test_numpy_to_pytorch_includes_explicit_padding_masks():
    data = pl.DataFrame(
        {
            "sequenceId": [1, 2],
            "subsequenceId": [0, 0],
            "startItemPosition": [0, 0],
            "leftPadLength": [0, 2],
            "inputCol": ["A", "A"],
            "3": [0.0, 0.0],
            "2": [0.0, 0.0],
            "1": [1.0, 1.0],
            "0": [2.0, 2.0],
        }
    )

    _, metadata = numpy_to_pytorch(
        data,
        {"A": torch.float32},
        ["A"],
        resolve_window_view(
            StoredWindowLayout(stored_context_width=4, max_target_offset=1, version=2),
            ModelWindowView(context_length=3, objective="causal", target_offset=1),
        ),
    )

    assert torch.equal(
        metadata["attention_valid_mask"],
        torch.tensor([[True, True, True], [False, False, True]]),
    )
    assert torch.equal(
        metadata["target_valid_mask"],
        torch.tensor([[True, True, True], [False, True, True]]),
    )


class _OnesSpanMasking:
    def sample(self, shape, device):
        return torch.ones(shape, dtype=torch.long, device=device)


def test_apply_bert_masking_uses_explicit_valid_mask_for_zero_values():
    config = SimpleNamespace(
        categorical_columns=[],
        real_columns=["real_col"],
        n_classes={},
        context_length=4,
        training_spec=SimpleNamespace(
            batch_size=1,
            bert_spec=SimpleNamespace(
                masking_probability=1.0,
                span_masking=_OnesSpanMasking(),
                replacement_distribution=SimpleNamespace(
                    masked=1.0,
                    random=0.0,
                    identical=0.0,
                ),
            ),
        ),
    )
    data_batch = {
        "real_col": torch.tensor([[99.0, 0.0, 1.0, 2.0]]),
    }
    targets_batch = {
        "real_col": torch.tensor([[99.0, 0.0, 1.0, 2.0]]),
    }
    metadata = {
        "attention_valid_mask": torch.tensor([[False, True, True, True]]),
        "target_valid_mask": torch.tensor([[False, True, True, True]]),
    }

    _, _, masked_metadata = apply_bert_masking(
        data_batch, targets_batch, metadata, config
    )

    assert torch.equal(
        masked_metadata["bert_mask"],
        torch.tensor([[False, True, True, True]]),
    )
