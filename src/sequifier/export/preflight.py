"""Disposable-process capability preflight with versioned, complete cache keys."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from sequifier.artifacts.manifests import write_manifest
from sequifier.artifacts.model_export import model_execution_config

PROTOCOL_VERSION = 1
PREFLIGHT_SEED = 20260910


def preflight_description(config):
    import onnxruntime
    import torch

    versions = {}
    for name in ("torch", "onnx", "onnxscript", "onnxruntime", "sequifier"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            if name != "sequifier":
                raise
            versions[name] = "source-checkout"
    root = Path(__file__).resolve().parents[1]
    implementation = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        implementation.update(str(path.relative_to(root)).encode())
        implementation.update(path.read_bytes())
    execution = model_execution_config(config)

    # Initial weights and their provenance do not affect graph capability.
    def graph_only(value):
        if isinstance(value, dict):
            return {
                key: graph_only(item)
                for key, item in value.items()
                if key
                not in {
                    "initialization",
                    "initialization_seed",
                    "id_maps",
                    "selected_columns_statistics",
                    "storage_layout",
                    "tensor_payload_version",
                }
            }
        if isinstance(value, (tuple, list)):
            return [graph_only(item) for item in value]
        return value

    return {
        "protocol_version": PROTOCOL_VERSION,
        "representative_batch_policy": "publication_clamped_2_to_4_preflight_2",
        "execution": graph_only(execution),
        "generative": config.export_generative_model,
        "embedding": config.export_embedding_model,
        "embedding_sites": config.embedding_layer_names,
        "dropout_mode": "stochastic" if config.export_with_dropout else "evaluation",
        "export_policy": {
            "dynamo": True,
            "opset": 18,
            "dtype": "float32",
            "external_data": "unique_sidecars_graph_committed_last",
            "optimize": False,
            "attention_lowering": "explicit_matmul_softmax_dropout_v1",
            "dynamic_shapes": "shared_batch_fixed_capacities",
        },
        "versions": versions,
        "implementation_sha256": implementation.hexdigest(),
        "provider": "CPUExecutionProvider",
        "session_optimization": "disabled" if config.export_with_dropout else "all",
        "available_providers": onnxruntime.get_available_providers(),
        "device": {
            "machine": platform.machine(),
            "system": platform.platform(),
            "torch_build": torch.__config__.show(),
        },
        "seed_policy": {
            "seed": PREFLIGHT_SEED,
            "boundary": "disposable_process_once",
            "per_batch_reseed": False,
        },
        "validation": {
            "graph_checker": True,
            "session_creation": True,
            "finite_outputs": True,
            "evaluation_rtol": 2e-4,
            "evaluation_atol": 2e-5,
            "dropout_inventory": "per_executed_module_count_and_scope",
            "focused_stochastic_acceptance": "pending_external_validation",
        },
    }


def ensure_export_preflight(config, distributed):
    if not config.export_onnx or not any(
        dataset.interface.depth_layouts for dataset in config.dataset_training.values()
    ):
        return
    failure = None
    if distributed.rank == 0:
        try:
            description = preflight_description(config)
            encoded = json.dumps(
                description, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
            fingerprint = hashlib.sha256(encoded).hexdigest()
            cache = Path(config.project_root) / ".sequifier" / "export-preflight"
            cache.mkdir(parents=True, exist_ok=True)
            record_path = cache / f"{fingerprint}.json"
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if (
                    record.get("success") is True
                    and record.get("description") == description
                ):
                    return_record = True
                else:
                    return_record = False
            else:
                return_record = False
            if not return_record:
                with tempfile.TemporaryDirectory(
                    prefix="worker-", dir=cache
                ) as directory:
                    request = Path(directory) / "request.json"
                    request.write_text(
                        json.dumps(
                            {
                                "execution": model_execution_config(config),
                                "generative": config.export_generative_model,
                                "embedding": config.export_embedding_model,
                                "dropout": config.export_with_dropout,
                                "seed": PREFLIGHT_SEED,
                            }
                        )
                    )
                    environment = dict(os.environ)
                    source_root = str(Path(__file__).resolve().parents[2])
                    environment["PYTHONPATH"] = (
                        source_root + os.pathsep + environment.get("PYTHONPATH", "")
                    )
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "sequifier.export.preflight",
                            str(request),
                        ],
                        capture_output=True,
                        text=True,
                        env=environment,
                        timeout=900,
                    )
                    if result.returncode:
                        raise RuntimeError((result.stderr or result.stdout)[-16000:])
                    record = {
                        "success": True,
                        "fingerprint": fingerprint,
                        "description": description,
                        "diagnostics": result.stdout[-16000:],
                    }
                    write_manifest(record_path, record)
        except Exception as error:
            failure = f"Depth ONNX capability preflight failed: {error}"
    failures = distributed.gather_objects(failure)
    if any(failures):
        raise RuntimeError(next(message for message in failures if message))


def _worker(request_path):
    import onnxruntime
    import torch

    from sequifier.artifacts.model_config import resolved_config_from_model_config
    from sequifier.export.embedding import EmbeddingModelExporter
    from sequifier.export.onnx import OnnxModelExporter
    from sequifier.helpers import configure_determinism
    from sequifier.model.factory import build_transformer_network

    request_path = Path(request_path)
    request = json.loads(request_path.read_text())
    configure_determinism(request["seed"])
    onnxruntime.set_seed(request["seed"])
    config, _ = resolved_config_from_model_config(request["execution"], device="cpu")
    network = build_transformer_network(
        config, device=torch.device("cpu")
    ).network.float()
    exporter = OnnxModelExporter()
    for index, dataset in enumerate(config.dataset_training.values()):
        if request["generative"]:
            exporter.export(
                network,
                dataset.interface,
                request_path.parent / f"{index}-generative.onnx",
                interface_name=dataset.model_interface,
                batch_size=2,
                context_length=config.global_training.context_length,
                training=request["dropout"],
            )
        if request["embedding"]:
            embedding = EmbeddingModelExporter().build(
                network, dataset.model_interface, config
            )
            exporter.export(
                embedding,
                dataset.interface,
                request_path.parent / f"{index}-embedding.onnx",
                interface_name=None,
                batch_size=2,
                context_length=config.global_training.context_length,
                training=request["dropout"],
            )
    print(
        "Graph, session, input/output contract, and publication output validation completed."
    )


if __name__ == "__main__":
    _worker(sys.argv[1])
