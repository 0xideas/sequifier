"""Resolved scoring contracts for supervised best-model selection."""

from pathlib import Path

from sequifier.fingerprints import fingerprint


def evaluation_batch_size(phase):
    evaluation = phase.evaluation
    if evaluation is not None and evaluation.batch_size is not None:
        return evaluation.batch_size
    return phase.batch_size


def evaluation_fingerprint(config, phase_index):
    """Exclude learning mechanics while retaining everything used for scoring.

    Contracts are compared within one run, whose model architecture and execution
    precision are fixed. Dataset interfaces and validation bindings are included
    explicitly so different routes or held-out sources cannot share a history.
    """
    phase = config.training_plan[phase_index]
    if phase.algorithm != "supervised" or phase.evaluation_monitor is None:
        return None
    sources = []
    for source in phase.evaluation_sources:
        dataset = config.dataset_training[source.dataset]
        learning = phase.learning.datasets.get(source.dataset)
        if learning is None:
            return None
        parts = (source.part,) if source.part is not None else tuple(dataset.parts)
        sources.append(
            {
                "source": source.source,
                "interface": dataset.model_interface,
                "interface_contract": dataset.interface.model_dump(mode="json"),
                "parts": [
                    {
                        "name": name,
                        "validation_path": str(
                            Path(dataset.parts[name].validation_data_path).resolve()
                        )
                        if dataset.parts[name].validation_data_path is not None
                        else None,
                        "read_format": dataset.parts[name].read_format,
                        "storage_form": dataset.parts[name].storage_form,
                        "file_order": dataset.parts[name].file_order,
                        "metadata": dataset.parts[name].metadata.model_dump(
                            mode="json"
                        ),
                        "image_paths": dataset.parts[name].image_paths,
                        "preconditioning_image_paths": dataset.parts[
                            name
                        ].preconditioning_image_paths,
                    }
                    for name in parts
                ],
                "criterion": learning.criterion,
                "class_weights": learning.class_weights,
                "loss_weights": {
                    target: (learning.loss_weights or {}).get(target, 1.0)
                    for target in dataset.interface.target_columns
                },
            }
        )
    runtime = config.runtime_settings
    return fingerprint(
        {
            "model": config.model.model_dump(mode="json"),
            "monitor": phase.evaluation_monitor.model_dump(mode="json"),
            "sources": sources,
            "objective": runtime.training_objective,
            "context_length": runtime.context_length,
            "target_offset": runtime.target_offset,
            "window_stride": phase.learning.window_stride,
            "bert_spec": phase.learning.bert_spec.model_dump(mode="json")
            if phase.learning.bert_spec is not None
            else None,
            # EvaluationService seeds corruption by phase, epoch and batch index.
            "mask_seed_phase": phase_index
            if runtime.training_objective == "bert"
            else None,
            "next_occurrence_config": phase.learning.next_occurrence_config.model_dump(
                mode="json"
            )
            if phase.learning.next_occurrence_config is not None
            else None,
            "batch_size": evaluation_batch_size(phase),
            "device": config.device,
            "layer_type_dtypes": runtime.layer_type_dtypes,
            "layer_autocast": runtime.layer_autocast,
            "float32_matmul_precision": runtime.float32_matmul_precision,
            "distributed": runtime.distributed,
            "data_parallelism": runtime.data_parallelism,
            "world_size": runtime.world_size,
            "num_workers": runtime.num_workers,
        }
    )


def compatible_supervised_phases(config, previous_index, current_index):
    current = config.training_plan[current_index]
    if current.model_selection.scope == "phase":
        return False
    previous = evaluation_fingerprint(config, previous_index)
    return previous is not None and previous == evaluation_fingerprint(
        config, current_index
    )
