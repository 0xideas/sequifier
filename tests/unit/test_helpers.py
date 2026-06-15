from types import SimpleNamespace

import polars as pl
import pytest
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
from sequifier.special_tokens import SPECIAL_TOKEN_IDS


def test_construct_index_maps_string():
    """Reverse string ID map."""
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
    """Reverse int ID map with sentinels."""
    id_maps: dict[str, dict[str | int, int]] = {"storeId": {100: 3, 101: 4}}
    target_cols = ["storeId"]

    result = construct_index_maps(id_maps, target_cols, map_to_id=True)

    assert result["storeId"][3] == 100
    assert result["storeId"][4] == 101

    assert result["storeId"][0] == "[unknown]"
    assert result["storeId"][1] == "[other]"
    assert result["storeId"][2] == "[mask]"


def test_construct_index_maps_flag_false():
    """Empty maps when map_to_id is false."""
    id_maps: dict[str, dict[str | int, int]] = {"itemId": {"a": 1}}
    result = construct_index_maps(id_maps, ["itemId"], map_to_id=False)
    assert result == {}


def test_numpy_to_pytorch_shapes_and_shifting():
    """Check causal input and shifted target tensors."""

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

    expected_input = torch.tensor([[10, 20, 30], [11, 21, 31]], dtype=torch.float32)
    assert torch.equal(tensors["A"], expected_input)

    expected_target = torch.tensor([[20, 30, 40], [21, 31, 41]], dtype=torch.float32)
    assert torch.equal(tensors["A_target"], expected_target)


def test_numpy_to_pytorch_dtypes():
    """Check column_types controls tensor dtypes."""

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
    def sample(self, shape, device, generator=None):
        return torch.ones(shape, dtype=torch.long, device=device)


class _VariableSpanMasking:
    def sample(self, shape, device, generator=None):
        lengths = torch.tensor([3, 2, 4, 1, 5], dtype=torch.long, device=device)
        repeats = (shape[1] + lengths.numel() - 1) // lengths.numel()
        return lengths.repeat(repeats)[: shape[1]].repeat(shape[0], 1)


def _bert_masking_config(
    masking_probability=0.5,
    *,
    categorical_columns=None,
    real_columns=None,
    n_classes=None,
    replacement_distribution=None,
    span_masking=None,
):
    return SimpleNamespace(
        categorical_columns=categorical_columns or [],
        real_columns=real_columns or [],
        n_classes=n_classes or {},
        context_length=6,
        training_spec=SimpleNamespace(
            batch_size=4,
            bert_spec=SimpleNamespace(
                masking_probability=masking_probability,
                span_masking=span_masking or _OnesSpanMasking(),
                replacement_distribution=replacement_distribution
                or SimpleNamespace(masked=0.0, random=0.0, identical=1.0),
            ),
        ),
    )


def test_apply_bert_masking_builds_exact_valid_masks():
    valid_mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [False, False, True, True, True, True],
            [False, False, False, False, False, False],
            [False, True, False, True, True, True],
        ]
    )
    config = _bert_masking_config(masking_probability=0.5)
    data_batch = {"passthrough": torch.arange(24).reshape(4, 6)}
    targets_batch = {"passthrough": torch.arange(24).reshape(4, 6)}
    metadata = {
        "attention_valid_mask": valid_mask,
        "target_valid_mask": valid_mask.clone(),
    }

    _, _, masked_metadata = apply_bert_masking(
        data_batch,
        targets_batch,
        metadata,
        config,
        eval_seed=11,
    )

    bert_mask = masked_metadata["bert_mask"]
    expected_budget = (valid_mask.sum(dim=1) * 0.5).long()

    assert bert_mask.dtype == torch.bool
    assert bert_mask.device == valid_mask.device
    assert torch.equal(bert_mask.sum(dim=1), expected_budget)
    assert not torch.any(bert_mask & ~valid_mask)


def test_apply_bert_masking_eval_seed_is_reproducible_without_global_rng_change():
    valid_mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [False, True, True, True, True, True],
        ]
    )
    config = _bert_masking_config(masking_probability=0.5)
    data_batch = {"passthrough": torch.arange(12).reshape(2, 6)}
    targets_batch = {"passthrough": torch.arange(12).reshape(2, 6)}
    metadata = {
        "attention_valid_mask": valid_mask,
        "target_valid_mask": valid_mask.clone(),
    }

    torch.manual_seed(123)
    rng_state = torch.get_rng_state().clone()
    _, _, first_metadata = apply_bert_masking(
        data_batch,
        targets_batch,
        metadata,
        config,
        eval_seed=99,
    )
    after_first = torch.get_rng_state().clone()

    _, _, second_metadata = apply_bert_masking(
        data_batch,
        targets_batch,
        metadata,
        config,
        eval_seed=99,
    )
    after_second = torch.get_rng_state().clone()

    assert torch.equal(first_metadata["bert_mask"], second_metadata["bert_mask"])
    assert torch.equal(after_first, rng_state)
    assert torch.equal(after_second, rng_state)


def test_apply_bert_masking_handles_variable_length_spans_and_sparse_masks():
    valid_mask = torch.tensor(
        [
            [True, True, True, True, True, True, True, True],
            [False, True, False, True, True, False, True, True],
        ]
    )
    config = _bert_masking_config(
        masking_probability=0.75,
        span_masking=_VariableSpanMasking(),
    )
    data_batch = {"passthrough": torch.arange(16).reshape(2, 8)}
    targets_batch = {"passthrough": torch.arange(16).reshape(2, 8)}
    metadata = {
        "attention_valid_mask": valid_mask,
        "target_valid_mask": valid_mask.clone(),
    }

    _, _, masked_metadata = apply_bert_masking(
        data_batch,
        targets_batch,
        metadata,
        config,
        eval_seed=99,
    )

    assert torch.equal(
        masked_metadata["bert_mask"],
        torch.tensor(
            [
                [False, True, True, True, True, True, True, False],
                [False, True, False, True, True, False, False, False],
            ]
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_bert_masking_cuda_stays_on_device_and_preserves_rng():
    device = torch.device("cuda")
    valid_mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [False, True, True, True, True, True],
        ],
        device=device,
    )
    config = _bert_masking_config(masking_probability=0.5)
    data_batch = {"passthrough": torch.arange(12, device=device).reshape(2, 6)}
    targets_batch = {"passthrough": torch.arange(12, device=device).reshape(2, 6)}
    metadata = {
        "attention_valid_mask": valid_mask,
        "target_valid_mask": valid_mask.clone(),
    }

    torch.manual_seed(123)
    cpu_state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state(device).clone()

    _, _, masked_metadata = apply_bert_masking(
        data_batch,
        targets_batch,
        metadata,
        config,
        eval_seed=99,
    )

    assert masked_metadata["bert_mask"].device == device
    assert torch.equal(torch.get_rng_state(), cpu_state)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_state)


def test_apply_bert_masking_replaces_categorical_with_mask_token_without_mutation():
    valid_mask = torch.tensor([[True, False, True, True]])
    config = _bert_masking_config(
        masking_probability=1.0,
        categorical_columns=["cat_col"],
        n_classes={"cat_col": 8},
        replacement_distribution=SimpleNamespace(
            masked=1.0,
            random=0.0,
            identical=0.0,
        ),
    )
    original = torch.tensor([[3, 4, 5, 6]])
    data_batch = {"cat_col": original.clone()}
    targets_batch = {"cat_col": original.clone()}
    metadata = {
        "attention_valid_mask": valid_mask,
        "target_valid_mask": valid_mask.clone(),
    }

    masked_data, _, _ = apply_bert_masking(
        data_batch,
        targets_batch,
        metadata,
        config,
        eval_seed=7,
    )

    assert torch.equal(
        masked_data["cat_col"],
        torch.tensor(
            [
                [
                    SPECIAL_TOKEN_IDS.mask,
                    4,
                    SPECIAL_TOKEN_IDS.mask,
                    SPECIAL_TOKEN_IDS.mask,
                ]
            ]
        ),
    )
    assert torch.equal(data_batch["cat_col"], original)


def test_apply_bert_masking_replaces_categorical_with_random_tokens_without_mutation():
    valid_mask = torch.ones((1, 5), dtype=torch.bool)
    config = _bert_masking_config(
        masking_probability=1.0,
        categorical_columns=["cat_col"],
        n_classes={"cat_col": 12},
        replacement_distribution=SimpleNamespace(
            masked=0.0,
            random=1.0,
            identical=0.0,
        ),
    )
    original = torch.tensor([[3, 4, 5, 6, 7]])
    data_batch = {"cat_col": original.clone()}
    targets_batch = {"cat_col": original.clone()}
    metadata = {
        "attention_valid_mask": valid_mask,
        "target_valid_mask": valid_mask.clone(),
    }

    masked_data, _, _ = apply_bert_masking(
        data_batch,
        targets_batch,
        metadata,
        config,
        eval_seed=7,
    )

    assert torch.all(masked_data["cat_col"] >= SPECIAL_TOKEN_IDS.user_start)
    assert torch.all(masked_data["cat_col"] < config.n_classes["cat_col"])
    assert not torch.equal(masked_data["cat_col"], original)
    assert torch.equal(data_batch["cat_col"], original)


def test_apply_bert_masking_replaces_real_with_zero_without_mutation():
    valid_mask = torch.tensor([[True, False, True, True]])
    config = _bert_masking_config(
        masking_probability=1.0,
        real_columns=["real_col"],
        replacement_distribution=SimpleNamespace(
            masked=1.0,
            random=0.0,
            identical=0.0,
        ),
    )
    original = torch.tensor([[1.5, 2.5, 3.5, 4.5]])
    data_batch = {"real_col": original.clone()}
    targets_batch = {"real_col": original.clone()}
    metadata = {
        "attention_valid_mask": valid_mask,
        "target_valid_mask": valid_mask.clone(),
    }

    masked_data, _, _ = apply_bert_masking(
        data_batch,
        targets_batch,
        metadata,
        config,
        eval_seed=7,
    )

    assert torch.equal(
        masked_data["real_col"],
        torch.tensor([[0.0, 2.5, 0.0, 0.0]]),
    )
    assert torch.equal(data_batch["real_col"], original)


def test_apply_bert_masking_replaces_real_with_random_noise_without_mutation():
    valid_mask = torch.tensor([[True, True, False, True]])
    config = _bert_masking_config(
        masking_probability=1.0,
        real_columns=["real_col"],
        replacement_distribution=SimpleNamespace(
            masked=0.0,
            random=1.0,
            identical=0.0,
        ),
    )
    original = torch.tensor([[10.0, 11.0, 12.0, 13.0]])
    data_batch = {"real_col": original.clone()}
    targets_batch = {"real_col": original.clone()}
    metadata = {
        "attention_valid_mask": valid_mask,
        "target_valid_mask": valid_mask.clone(),
    }

    masked_data, _, _ = apply_bert_masking(
        data_batch,
        targets_batch,
        metadata,
        config,
        eval_seed=7,
    )

    assert torch.equal(masked_data["real_col"][~valid_mask], original[~valid_mask])
    assert torch.all(masked_data["real_col"][valid_mask] != original[valid_mask])
    assert torch.equal(data_batch["real_col"], original)


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
