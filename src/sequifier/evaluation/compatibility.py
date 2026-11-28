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
        interface = config.dataset_interface(source.dataset)
        learning = phase.learning.datasets.get(source.dataset)
        if learning is None:
            return None
        sources.append(
            {
                "source": source.source,
                "interface": dataset.model_interface,
                "interface_contract": {
                    **interface.model_dump(mode="json"),
                    **config.dataset_storage(source.dataset).model_dump(mode="json"),
                },
                "parts": [
                    {
                        "name": name,
                        "validation_path": str(
                            Path(
                                dataset.parts[name].split_paths["validation"]
                            ).resolve()
                        ),
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
                    for name in source.parts
                ],
                "criterion": learning.criterion,
                "class_weights": learning.class_weights,
                "loss_weights": {
                    target: (learning.loss_weights or {}).get(target, 1.0)
                    for target in interface.target_columns
                },
            }
        )
    options = config.execution.options
    execution = config.run.execution.supervised_settings
    precision = config.run.execution.precision
    return fingerprint(
        {
            "model": config.model.model_dump(mode="json"),
            "monitor": phase.evaluation_monitor.model_dump(mode="json"),
            "sources": sources,
            "objective": phase.learning.objective_name(options.training_objective),
            "context_length": options.context_length,
            "target_offset": options.target_offset,
            "window_stride": phase.learning.window_stride,
            "bert_spec": phase.learning.bert_spec.model_dump(mode="json")
            if phase.learning.bert_spec is not None
            else None,
            # EvaluationService seeds corruption by phase, epoch and batch index.
            "mask_seed_phase": phase_index
            if options.training_objective == "bert"
            else None,
            "next_occurrence_config": phase.learning.next_occurrence_config.model_dump(
                mode="json"
            )
            if phase.learning.next_occurrence_config is not None
            else None,
            "batch_size": evaluation_batch_size(phase),
            "device": config.device,
            "layer_type_dtypes": options.layer_type_dtypes,
            "layer_autocast": precision.layer_autocast,
            "float32_matmul_precision": precision.float32_matmul_precision,
            "data_parallelism": execution.distribution.strategy,
            "world_size": execution.distribution.world_size,
            "num_workers": execution.data_loading.num_workers,
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
