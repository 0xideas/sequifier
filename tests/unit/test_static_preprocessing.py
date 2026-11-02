"""Regression coverage for reused static preprocessing metadata."""

import json

import polars as pl

from sequifier.preprocess import Preprocessor


def test_reused_integer_static_categories_keep_their_ids(tmp_path):
    metadata_path = tmp_path / "configs" / "metadata_configs" / "events.json"
    metadata_path.parent.mkdir(parents=True)
    (tmp_path / "data").mkdir()
    existing = {
        "static_columns": ["category"],
        "n_classes": {"category": 5},
        "column_data_types": {"category": "Int64"},
        "id_maps": {"category": {10: 3, 20: 4}},
        "selected_columns_statistics": {},
    }
    metadata_path.write_text(json.dumps(existing))

    preprocessor = Preprocessor.__new__(Preprocessor)
    preprocessor._static_source = pl.DataFrame(
        {"sequenceId": [1, 2], "category": [10, 20]}
    )
    preprocessor._preexisting_static_metadata = json.loads(metadata_path.read_text())
    preprocessor.static_categorical_columns = ["category"]
    preprocessor.static_real_columns = []
    preprocessor.normalize_real_columns = True
    preprocessor.project_root = str(tmp_path)
    preprocessor.data_name_root = "events"
    preprocessor.merge_output = True
    preprocessor.output_n_classes = {}
    preprocessor.output_column_data_types = {}
    preprocessor.output_id_maps = {}
    preprocessor.output_selected_columns_statistics = {}

    preprocessor._finalize_static_source()

    processed = pl.read_parquet(tmp_path / "data" / "events-static.parquet")
    assert processed["category"].to_list() == [3, 4]
    assert json.loads(metadata_path.read_text())["id_maps"]["category"] == {
        "10": 3,
        "20": 4,
    }
