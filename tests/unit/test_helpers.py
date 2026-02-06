import polars as pl
import torch

from sequifier.helpers import construct_index_maps, numpy_to_pytorch

# ==========================================
# 1. Test Index Map Construction
# ==========================================


def test_construct_index_maps_string():
    """Tests reversing a string-to-int map, ensuring 0 maps to 'unknown'."""
    id_maps: dict[str, dict[str | int, int]] = {"itemId": {"apple": 2, "banana": 3}}
    target_cols = ["itemId"]

    result = construct_index_maps(id_maps, target_cols, map_to_id=True)

    # Check standard reversal
    assert result["itemId"][2] == "apple"
    assert result["itemId"][3] == "banana"

    # Check the special 0 index for strings
    assert result["itemId"][0] == "unknown"
    assert result["itemId"][1] == "other"


def test_construct_index_maps_integer():
    """Tests reversing an int-to-int map, ensuring 0 maps to min_id - 1."""
    id_maps: dict[str, dict[str | int, int]] = {"storeId": {100: 2, 101: 3}}
    target_cols = ["storeId"]

    result = construct_index_maps(id_maps, target_cols, map_to_id=True)

    assert result["storeId"][2] == 100
    assert result["storeId"][3] == 101

    # Check the special 0 index for integers (min value - 2)
    # Min value is 100, so 2 -> 98, 1 -> 99
    assert result["storeId"][0] == 98
    assert result["storeId"][1] == 99


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
    If seq_length = 3:
    - Input cols:  ['3', '2', '1']
    - Target cols: ['2', '1', '0']
    """

    # Setup: 2 sequences for feature "A"
    # Seq 1: [10, 20, 30, 40] (where 40 is t=0, 30 is t=1, etc.)
    data = pl.DataFrame(
        {
            "inputCol": ["A", "A"],
            "3": [10, 11],
            "2": [20, 21],
            "1": [30, 31],
            "0": [40, 41],
        }
    )

    column_types = {"A": torch.float32}
    all_columns = ["A"]
    seq_length = 3

    tensors = numpy_to_pytorch(data, column_types, all_columns, seq_length)

    # 1. Check Keys
    assert "A" in tensors
    assert "A_target" in tensors

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
    data_int = pl.DataFrame({"inputCol": ["int_col"], "1": [10], "0": [20]})
    tensors_int = numpy_to_pytorch(
        data_int, {"int_col": torch.int64}, ["int_col"], seq_length=1
    )
    assert tensors_int["int_col"].dtype == torch.int64

    # Case 2: Float
    data_float = pl.DataFrame({"inputCol": ["float_col"], "1": [10.5], "0": [20.5]})
    tensors_float = numpy_to_pytorch(
        data_float, {"float_col": torch.float32}, ["float_col"], seq_length=1
    )
    assert tensors_float["float_col"].dtype == torch.float32
