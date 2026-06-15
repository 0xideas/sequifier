import json
import os
import re
from collections import Counter

import numpy as np
import polars as pl
import torch

from sequifier.helpers import (
    ModelWindowView,
    StoredWindowLayout,
    resolve_window_view,
    stored_window_layout_from_metadata,
)

TARGET_VARIABLE_DICT = {"categorical": "itemId", "real": "itemValue"}
BERT_SEQ_LENGTH = 8
BERT_EMBEDDING_DIM = 16
BERT_DATA_NAME = "test-data-categorical-1-lookahead-0"
CAUSAL_CONTEXT_LENGTH = 8


def _project_path(project_root, path):
    if os.path.isabs(path) or path.startswith(project_root):
        return path
    return os.path.join(project_root, path)


def _causal_storage_layout(data_path):
    metadata_path = (
        os.path.join(data_path, "metadata.json") if os.path.isdir(data_path) else None
    )
    if metadata_path is not None and os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return stored_window_layout_from_metadata(json.load(f))

    return StoredWindowLayout(
        stored_context_width=CAUSAL_CONTEXT_LENGTH + 1,
        max_target_offset=1,
        version=2,
    )


def _target_valid_mask(left_pad_lengths, storage_layout, prediction_length):
    resolved_view = resolve_window_view(
        storage_layout,
        ModelWindowView(
            context_length=CAUSAL_CONTEXT_LENGTH,
            objective="causal",
            target_offset=1,
        ),
    )
    masks = resolved_view.build_masks(torch.tensor(left_pad_lengths, dtype=torch.int64))
    return masks["target_valid_mask"][:, -prediction_length:].reshape(-1).numpy()


def _read_pt_window_metadata(data_path):
    contents = []
    for root, _, files in os.walk(data_path):
        for file in sorted(files):
            if not file.endswith(".pt"):
                continue

            loaded = torch.load(os.path.join(root, file), weights_only=False)
            if len(loaded) == 5:
                _, sequence_ids, _, start_positions, left_pad_lengths = loaded
            else:
                _, sequence_ids, _, start_positions = loaded
                left_pad_lengths = torch.zeros_like(sequence_ids)

            contents.append(
                pl.DataFrame(
                    {
                        "sequenceId": sequence_ids.detach().cpu().numpy(),
                        "startItemPosition": start_positions.detach().cpu().numpy(),
                        "leftPadLength": left_pad_lengths.detach().cpu().numpy(),
                    }
                )
            )

    assert len(contents) > 0, f"no files found for {data_path}"
    return pl.concat(contents, how="vertical")


def _read_long_window_metadata(data_path):
    paths = []
    if os.path.isdir(data_path):
        for root, _, files in os.walk(data_path):
            paths.extend(
                os.path.join(root, file)
                for file in sorted(files)
                if file.endswith((".csv", ".parquet"))
            )
    else:
        paths = [data_path]

    contents = []
    for path in paths:
        data = pl.read_csv(path) if path.endswith(".csv") else pl.read_parquet(path)
        if "leftPadLength" not in data.columns:
            data = data.with_columns(pl.lit(0).alias("leftPadLength"))
        contents.append(
            data.group_by(["sequenceId", "subsequenceId"], maintain_order=True).agg(
                pl.col("startItemPosition").first(),
                pl.col("leftPadLength").first(),
            )
        )

    assert len(contents) > 0, f"no files found for {data_path}"
    return pl.concat(contents, how="vertical")


def _prediction_source_spec(project_root, model_name):
    if model_name == "model-categorical-multitarget-5-best-3":
        return {
            "path": _project_path(
                project_root, "data/test-data-categorical-multitarget-5-split2"
            ),
            "format": "parquet",
            "prediction_length": 1,
            "autoregression_total_steps": None,
        }
    if model_name == "model-categorical-multitarget-5-last-3":
        return {
            "path": _project_path(
                project_root, "data/test-data-categorical-multitarget-5-split2"
            ),
            "format": "parquet",
            "prediction_length": 1,
            "autoregression_total_steps": None,
        }
    if model_name == "model-categorical-distributed-best-3":
        return {
            "path": _project_path(project_root, "data/test-data-categorical-3-split2"),
            "format": "pt",
            "prediction_length": 1,
            "autoregression_total_steps": None,
        }
    if model_name == "model-categorical-lazy-best-3":
        return {
            "path": _project_path(project_root, "data/test-data-categorical-3-split2"),
            "format": "pt",
            "prediction_length": 1,
            "autoregression_total_steps": None,
        }
    if model_name == "model-categorical-1-best-3-autoregression":
        return {
            "path": _project_path(project_root, "data/test-data-categorical-1-split2"),
            "format": "pt",
            "prediction_length": 1,
            "autoregression_total_steps": 20,
        }
    if model_name == "model-real-1-best-3-autoregression":
        return {
            "path": _project_path(
                project_root, "data/test-data-real-1-split1-autoregression.csv"
            ),
            "format": "csv",
            "prediction_length": 1,
            "autoregression_total_steps": 20,
            "csv_autoregression": True,
        }

    match = re.fullmatch(r"model-categorical-(\d+)-inf-size-best-3", model_name)
    if match is not None:
        data_number = int(match.group(1))
        return {
            "path": _project_path(
                project_root, f"data/test-data-categorical-{data_number}-split2"
            ),
            "format": "pt",
            "prediction_length": 3,
            "autoregression_total_steps": None,
        }

    match = re.fullmatch(r"model-categorical-(\d+)-best-3", model_name)
    if match is not None:
        data_number = int(match.group(1))
        return {
            "path": _project_path(
                project_root, f"data/test-data-categorical-{data_number}-split2"
            ),
            "format": "pt",
            "prediction_length": 1,
            "autoregression_total_steps": None,
        }

    match = re.fullmatch(r"model-real-(\d+)-best-3", model_name)
    if match is not None:
        data_number = int(match.group(1))
        return {
            "path": _project_path(
                project_root, f"data/test-data-real-{data_number}-split1.parquet"
            ),
            "format": "parquet",
            "prediction_length": 1,
            "autoregression_total_steps": None,
        }

    raise AssertionError(f"No source metadata mapping for {model_name}")


def _expected_prediction_positions(project_root, model_name):
    spec = _prediction_source_spec(project_root, model_name)
    metadata = (
        _read_pt_window_metadata(spec["path"])
        if spec["format"] == "pt"
        else _read_long_window_metadata(spec["path"])
    )
    storage_layout = _causal_storage_layout(spec["path"])
    prediction_length = spec["prediction_length"]

    if spec["autoregression_total_steps"] is not None:
        total_steps = spec["autoregression_total_steps"]
        if spec.get("csv_autoregression", False):
            metadata = metadata.group_by("sequenceId", maintain_order=True).head(1)

        sequence_ids = np.repeat(metadata["sequenceId"].to_numpy(), total_steps)
        item_positions = np.concatenate(
            [
                np.arange(start, start + total_steps)
                for start in (
                    metadata["startItemPosition"].to_numpy() + CAUSAL_CONTEXT_LENGTH
                )
            ],
            axis=0,
        )
        valid_mask = np.repeat(
            _target_valid_mask(
                metadata["leftPadLength"].to_numpy(), storage_layout, prediction_length
            ),
            total_steps,
        )
    else:
        starts = metadata["startItemPosition"].to_numpy()
        offsets = np.arange(-prediction_length + 1, 1)
        sequence_ids = np.repeat(metadata["sequenceId"].to_numpy(), prediction_length)
        item_positions = np.repeat(
            starts + CAUSAL_CONTEXT_LENGTH, prediction_length
        ) + np.tile(offsets, len(starts))
        valid_mask = _target_valid_mask(
            metadata["leftPadLength"].to_numpy(), storage_layout, prediction_length
        )

    return pl.DataFrame(
        {
            "sequenceId": sequence_ids[valid_mask],
            "itemPosition": item_positions[valid_mask],
        }
    )


def _categorical_metadata(project_root):
    metadata_path = os.path.join(
        project_root, "configs", "metadata_configs", f"{BERT_DATA_NAME}.json"
    )
    with open(metadata_path, "r") as f:
        return json.load(f)


def _bert_inference_metadata(project_root):
    data_path = os.path.join(project_root, "data", f"{BERT_DATA_NAME}-split2")
    contents = []
    for root, _, files in os.walk(data_path):
        for file in sorted(files):
            if not file.endswith(".pt"):
                continue

            loaded = torch.load(os.path.join(root, file), weights_only=False)
            if len(loaded) == 5:
                _, sequence_ids, subsequence_ids, _, left_pad_lengths = loaded
            else:
                _, sequence_ids, subsequence_ids, _ = loaded
                left_pad_lengths = None

            if left_pad_lengths is None:
                left_pad_lengths = torch.zeros_like(sequence_ids)

            valid_counts = torch.clamp(
                BERT_SEQ_LENGTH - left_pad_lengths, min=0, max=BERT_SEQ_LENGTH
            )
            contents.append(
                pl.DataFrame(
                    {
                        "sequenceId": sequence_ids.detach().cpu().numpy(),
                        "subsequenceId": subsequence_ids.detach().cpu().numpy(),
                        "valid_count": valid_counts.detach().cpu().numpy(),
                    }
                )
            )

    assert len(contents) > 0, f"no files found for {data_path}"
    return pl.concat(contents, how="vertical")


def _expected_bert_valid_counts(project_root, group_columns):
    metadata = _bert_inference_metadata(project_root)

    return (
        metadata.group_by(group_columns)
        .agg(pl.col("valid_count").sum().alias("expected_len"))
        .filter(pl.col("expected_len") > 0)
    )


def _assert_counts_match_expected(actual, expected, group_columns):
    actual_keys = set(actual.select(group_columns).iter_rows())
    expected_keys = set(expected.select(group_columns).iter_rows())

    assert actual_keys == expected_keys

    comparison = actual.join(expected, on=group_columns)
    assert comparison.select(
        (pl.col("len") == pl.col("expected_len")).all()
    ).item(), comparison


def test_predictions_real(predictions):
    for model_name, model_predictions in predictions.items():
        if "categorical" not in model_name or "multitarget" in model_name:
            if "multitarget" not in model_name:
                assert np.all(
                    [
                        v > -10.0 and v < 10.0
                        for v in model_predictions[
                            TARGET_VARIABLE_DICT["real"]
                        ].to_numpy()
                    ]
                )
            else:
                assert np.all(
                    [
                        v > -10.0 and v < 10.0
                        for v in model_predictions["supReal3"].to_numpy()
                    ]
                ), model_predictions


def test_predictions_cat(predictions):
    valid_values = [str(x) for x in np.arange(100, 130)] + [
        "[unknown]",
        "[other]",
        "[mask]",
    ]
    for model_name, model_predictions in predictions.items():
        if "categorical" in model_name or "multitarget" in model_name:
            assert np.all(
                [
                    v in valid_values
                    for v in model_predictions[
                        TARGET_VARIABLE_DICT["categorical"]
                    ].to_numpy()
                ]
            ), model_predictions

            if "multitarget" in model_name:
                admssible_vals = [str(x) for x in np.arange(0, 10)] + [
                    "[unknown]",
                    "[other]",
                    "[mask]",
                ]
                assert np.all(
                    [
                        v in admssible_vals
                        for v in model_predictions["supCat1"].to_numpy()
                    ]
                ), model_predictions

            if "inf" in model_name:
                prediction_length = 3
                n_test_rows = model_predictions.height
                baseline_preds = predictions["model-categorical-1-best-3"]
                n_baseline_rows = baseline_preds.height

                assert n_test_rows == n_baseline_rows * prediction_length, (
                    f"Expected {n_baseline_rows * prediction_length} rows for prediction_length={prediction_length}, "
                    f"but found {n_test_rows} rows."
                )

                baseline_rows_per_seq = (
                    baseline_preds.group_by("sequenceId").len().height
                )
                test_rows_per_seq_groups = model_predictions.group_by(
                    "sequenceId"
                ).len()

                assert baseline_rows_per_seq == test_rows_per_seq_groups.height
                assert (
                    test_rows_per_seq_groups["len"] == prediction_length
                ).all(), (
                    f"Test should have {prediction_length} predictions per sequence"
                )


def test_probabilities(probabilities):
    for model_name, model_probabilities in probabilities.items():
        if "itemId" in model_name:
            assert model_probabilities.shape[1] == 33
        elif "supCat1" in model_name:
            assert model_probabilities.shape[1] == 13

        np.testing.assert_almost_equal(
            model_probabilities.sum_horizontal(),
            np.ones(model_probabilities.shape[0]),
            decimal=5,
        )


def test_bert_generative_predictions_default_to_context_length(
    bert_predictions, project_root
):
    metadata = _categorical_metadata(project_root)
    valid_values = {str(v) for v in metadata["id_maps"]["itemId"].keys()}.union(
        {"[unknown]", "[other]"}
    )

    assert bert_predictions.height > 0
    assert set(bert_predictions["itemId"].to_list()).issubset(valid_values)

    rows_per_sequence = bert_predictions.group_by("sequenceId").len()
    expected_rows_per_sequence = _expected_bert_valid_counts(
        project_root, ["sequenceId"]
    )
    _assert_counts_match_expected(
        rows_per_sequence, expected_rows_per_sequence, ["sequenceId"]
    )


def test_bert_probabilities(bert_predictions, bert_probabilities, project_root):
    metadata = _categorical_metadata(project_root)
    expected_class_count = metadata["n_classes"]["itemId"]

    assert bert_probabilities.height == bert_predictions.height
    assert bert_probabilities.shape[1] == expected_class_count
    np.testing.assert_almost_equal(
        bert_probabilities.sum_horizontal(),
        np.ones(bert_probabilities.shape[0]),
        decimal=5,
    )


def test_multi_pred(predictions):
    multitarget_models = [name for name in predictions.keys() if "multitarget" in name]

    for model_name in multitarget_models:
        preds = predictions[model_name]

        assert preds.shape[0] > 0, f"{model_name} has no predictions"
        assert preds.shape[1] == 5, f"{model_name} should have 5 columns"

        admssible_vals = [str(x) for x in np.arange(0, 10)] + [
            "[unknown]",
            "[other]",
            "[mask]",
        ]

        assert np.all(
            [v in admssible_vals for v in preds["supCat1"]]
        ), f"Invalid supCat1 values in {model_name}"
        assert np.all(preds["supReal3"].to_numpy() > -4.0) and np.all(
            preds["supReal3"].to_numpy() < 4.0
        ), f"supReal3 out of bounds in {model_name}"


def test_embeddings(embeddings):
    for model_name, model_embeddings in embeddings.items():
        if "categorical-1" in model_name:
            assert model_embeddings.shape[0] == 10
            assert model_embeddings.shape[1] == 203
            assert np.abs(model_embeddings[:, 1:].to_numpy().mean()) < 0.3
        if "categorical-3" in model_name:
            assert model_embeddings.shape[0] == 30
            assert model_embeddings.shape[1] == 203
            assert np.abs(model_embeddings[:, 1:].to_numpy().mean()) < 0.3


def test_bert_embeddings(bert_predictions, bert_embeddings, project_root):
    expected_embedding_cols = [str(i) for i in range(BERT_EMBEDDING_DIM)]
    expected_columns = ["sequenceId", "subsequenceId", "itemPosition"]

    assert bert_embeddings.height == bert_predictions.height
    assert bert_embeddings.shape[1] == len(expected_columns) + BERT_EMBEDDING_DIM
    assert all(col in bert_embeddings.columns for col in expected_columns)
    assert all(col in bert_embeddings.columns for col in expected_embedding_cols)

    embedding_values = bert_embeddings.select(expected_embedding_cols).to_numpy()
    assert np.isfinite(embedding_values).all()

    rows_per_subsequence = bert_embeddings.group_by(
        ["sequenceId", "subsequenceId"]
    ).len()
    expected_rows_per_subsequence = _expected_bert_valid_counts(
        project_root, ["sequenceId", "subsequenceId"]
    )
    _assert_counts_match_expected(
        rows_per_subsequence,
        expected_rows_per_subsequence,
        ["sequenceId", "subsequenceId"],
    )


def test_predictions_item_position(predictions, project_root):
    """Check itemPosition values against source preprocessing metadata."""
    for model_name, preds_df in predictions.items():
        expected_positions = _expected_prediction_positions(project_root, model_name)
        actual_pairs = list(preds_df.select(["sequenceId", "itemPosition"]).iter_rows())
        expected_pairs = list(expected_positions.iter_rows())

        duplicate_pairs = [
            pair for pair, count in Counter(actual_pairs).items() if count > 1
        ]
        assert duplicate_pairs == [], (
            f"Model '{model_name}': Found duplicate sequence-position pairs:\n"
            f"{duplicate_pairs[:10]}"
        )

        assert Counter(actual_pairs) == Counter(
            expected_pairs
        ), f"Model '{model_name}': itemPosition values do not match source metadata."


def test_embeddings_subsequence_id(embeddings):
    """Check subsequenceId increments within each sequence."""
    for model_name, embeds_df in embeddings.items():
        for sequence_id, sequence_df in embeds_df.group_by(
            "sequenceId", maintain_order=True
        ):
            subsequence_ids = sorted(
                sequence_df.get_column("subsequenceId").unique().to_list()
            )
            expected_ids = list(range(len(subsequence_ids)))
            assert subsequence_ids == expected_ids, (
                f"Model '{model_name}', sequenceId {sequence_id}: expected "
                f"subsequenceIds {expected_ids}, found {subsequence_ids}."
            )
