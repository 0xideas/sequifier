import json

import pytest

from sequifier.config.infer_config import _assert_folder_metadata_matches
from sequifier.config.metadata import DatasetMetadata


@pytest.fixture
def metadata_values():
    return {
        "window_length": 5,
        "column_data_types": {"category": "Int64", "value": "Float64"},
        "n_classes": {"category": 5},
        "id_maps": {"category": {"first": 3, "second": 4}},
        "selected_columns_statistics": {"value": {"mean": 2.0, "std": 0.5}},
        "normalize_real_columns": True,
    }


def _check_folder(tmp_path, selected_values, folder_values):
    (tmp_path / "metadata.json").write_text(json.dumps(folder_values))
    return _assert_folder_metadata_matches(
        str(tmp_path),
        DatasetMetadata.model_validate(selected_values),
        ["category"],
        target_columns=["value"],
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("id_maps", {"category": {"first": 4, "second": 3}}),
        ("selected_columns_statistics", {"value": {"mean": 3.0, "std": 0.5}}),
        ("normalize_real_columns", False),
    ],
)
def test_inference_rejects_different_folder_meanings(
    tmp_path, metadata_values, field, replacement
):
    folder_values = dict(metadata_values)
    folder_values[field] = replacement

    with pytest.raises(ValueError, match=field):
        _check_folder(tmp_path, metadata_values, folder_values)


def test_inference_requires_folder_semantic_fields(tmp_path, metadata_values):
    folder_values = dict(metadata_values)
    del folder_values["id_maps"]

    with pytest.raises(ValueError, match="missing semantic fields.*id_maps.*reprocess"):
        _check_folder(tmp_path, metadata_values, folder_values)


def test_inference_accepts_matching_folder_meanings(tmp_path, metadata_values):
    assert _check_folder(tmp_path, metadata_values, metadata_values) is not None
