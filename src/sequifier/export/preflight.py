"""Disposable-process capability preflight with explicit cache versions."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

from sequifier.artifacts.manifests import write_manifest
from sequifier.artifacts.model_export import model_execution_config
from sequifier.contract_versions import ContractVersions
from sequifier.model.execution import ModelExecution

PREFLIGHT_SEED = 20260910


def preflight_description(config, execution):
    import torch

    versions = {
        name: version(name) for name in ("torch", "onnx", "onnxscript", "onnxruntime")
    }

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
                    "split_paths",
                    "storage_layout",
                    "tensor_payload_kind",
                    "tensor_payload_version",
                }
            }
        if isinstance(value, (tuple, list)):
            return [graph_only(item) for item in value]
        return value

    export = config.run.export
    return {
        "protocol_version": ContractVersions.EXPORT_PREFLIGHT,
        "representative_batch_policy": "publication_clamped_2_to_4_preflight_2",
        "execution": graph_only(execution),
        "generative": export.generative_model,
        "embedding": export.embedding_model,
        "embedding_sites": config.embedding_layer_names,
        "dropout_mode": "stochastic" if export.with_dropout else "evaluation",
        "export_policy": {
            "dynamo": True,
            "opset": ContractVersions.ONNX_OPSET,
            "dtype": "float32",
            "external_data": "unique_external_files_graph_committed_last",
            "optimize": False,
            "attention_lowering": f"explicit_matmul_softmax_dropout_v{ContractVersions.EXPORT_ATTENTION_LOWERING}",
            "dynamic_shapes": "shared_batch_fixed_capacities",
        },
        "versions": versions,
        "provider": "CPUExecutionProvider",
        "session_optimization": "disabled" if export.with_dropout else "all",
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
    export = config.run.export
    if config.run_mode != "supervised" or not export.onnx:
        return
    execution = ModelExecution.from_config(config)
    interfaces = {
        dataset.model_interface: execution.interface(dataset.model_interface)
        for dataset in config.dataset_training.values()
    }
    if not any(
        interface.depth_layouts
        or interface.image_sources
        or interface.preconditioning_image is not None
        for interface in interfaces.values()
    ):
        return
    failure = None
    if distributed.rank == 0:
        try:
            contract = model_execution_config(replace(execution, interfaces=interfaces))
            description = preflight_description(config, contract)
            encoded = json.dumps(
                description, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
            fingerprint = hashlib.sha256(encoded).hexdigest()
            cache = Path(config.project_root) / ".sequifier" / "export-preflight"
            cache.mkdir(parents=True, exist_ok=True)
            record_path = cache / f"{fingerprint}.json"
            if record_path.exists():
                record = json.loads(record_path.read_text())
                cached_description = record.get("description")
                cached_protocol = (
                    cached_description.get("protocol_version")
                    if isinstance(cached_description, dict)
                    else None
                )
                if (
                    record.get("success") is True
                    and isinstance(cached_protocol, int)
                    and not isinstance(cached_protocol, bool)
                    and cached_description == description
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
                                "execution": contract,
                                "generative": export.generative_model,
                                "embedding": export.embedding_model,
                                "dropout": export.with_dropout,
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
            failure = f"Media ONNX capability preflight failed: {error}"
    failures = distributed.gather_objects(failure)
    if any(failures):
        raise RuntimeError(next(message for message in failures if message))


def _worker(request_path):
    import onnxruntime
    import torch

    from sequifier.artifacts.model_config import model_execution_from_model_config
    from sequifier.export.embedding import EmbeddingModelExporter
    from sequifier.export.onnx import OnnxModelExporter
    from sequifier.helpers import configure_determinism
    from sequifier.model.factory import build_transformer_network

    request_path = Path(request_path)
    request = json.loads(request_path.read_text())
    configure_determinism(request["seed"])
    onnxruntime.set_seed(request["seed"])
    config, _ = model_execution_from_model_config(request["execution"])
    network = build_transformer_network(config, device=torch.device("cpu")).float()
    exporter = OnnxModelExporter()
    for index, (name, interface) in enumerate(config.interfaces.items()):
        if request["generative"]:
            exporter.export(
                network,
                interface,
                request_path.parent / f"{index}-generative.onnx",
                interface_name=name,
                batch_size=2,
                context_length=config.options.context_length,
                training=request["dropout"],
            )
        if request["embedding"]:
            embedding = EmbeddingModelExporter().build(network, name, config)
            exporter.export(
                embedding,
                interface,
                request_path.parent / f"{index}-embedding.onnx",
                interface_name=None,
                batch_size=2,
                context_length=config.options.context_length,
                training=request["dropout"],
            )
    print(
        "Graph, session, input/output contract, and publication output validation completed."
    )


if __name__ == "__main__":
    _worker(sys.argv[1])
