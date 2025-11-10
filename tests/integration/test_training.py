import os

import numpy as np


def test_checkpoint_files_exists(run_training, project_path):
    found_items = np.array(
        sorted(list(os.listdir(os.path.join(project_path, "checkpoints"))))
    )
    expected_items = np.array(
        sorted(
            [
                f"model-{model_type}-{j}-epoch-{i}.pt"
                for model_type in ["categorical", "real"]
                for j in [1, 3, 5, 50]
                for i in range(1, 4)
            ]
            + [
                f"model-categorical-{j}-inf-size-epoch-{i}.pt"
                for j in [1, 3]
                for i in range(1, 4)
            ]
            + [f"model-categorical-multitarget-5-epoch-{i}.pt" for i in range(1, 4)]
        )
    )

    print(f"{expected_items = }")
    print(f"{found_items = }")

    assert np.all(
        found_items == expected_items
    ), f"{found_items = } != {expected_items = }"


def test_model_files_exists(run_training, project_path):
    model_type_formats = {"categorical": ["onnx", "pt"], "real": ["onnx", "pt"]}
    found_items = np.array(
        sorted(list(os.listdir(os.path.join(project_path, "models"))))
    )

    expected_items = np.array(
        sorted(
            [
                f"sequifier-model-{model_type2}-{j}-{kind}{model_type}-3.{model_type_format}"
                for model_type2 in ["categorical", "real"]
                for model_type in ["", "-embedding"]
                for model_type_format in model_type_formats[model_type2]
                for j in [1, 3, 5, 50]
                for kind in ["best", "last"]
            ]
            + [
                "sequifier-model-categorical-multitarget-5-best-3.onnx",
                "sequifier-model-categorical-multitarget-5-last-3.onnx",
                "sequifier-model-categorical-multitarget-5-best-embedding-3.onnx",
                "sequifier-model-categorical-multitarget-5-last-embedding-3.onnx",
                "sequifier-model-real-1-best-3-autoregression.pt",
                "sequifier-model-categorical-1-best-3-autoregression.onnx",
                "sequifier-model-categorical-1-inf-size-best-3.onnx",
                "sequifier-model-categorical-1-inf-size-last-3.onnx",
                "sequifier-model-categorical-1-inf-size-best-embedding-3.onnx",
                "sequifier-model-categorical-1-inf-size-last-embedding-3.onnx",
                "sequifier-model-categorical-3-inf-size-best-3.pt",
                "sequifier-model-categorical-3-inf-size-last-3.pt",
                "sequifier-model-categorical-3-inf-size-best-embedding-3.pt",
                "sequifier-model-categorical-3-inf-size-last-embedding-3.pt",
            ]
        )
    )

    print(f"{expected_items = }")
    print(f"{found_items = }")
    assert np.all(
        found_items == expected_items
    ), f"{found_items = } != {expected_items = }"
