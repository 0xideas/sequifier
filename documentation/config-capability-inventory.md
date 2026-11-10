# Configuration capability inventory

Capability baseline: `082225833e1d744e662d094008bcc44268ab868a`. Layout reference: `2d01d407`.

The phase/route redesign supersedes baseline placements below. Training batching
and optimization belong to phases, inference validates the selected model route,
and rollout declarations use `environment`. See the current
[training](configs/train.md), [inference](configs/infer.md), and
[phase/route tests](../../sequifier-test/local/unit/test_phase_route_contracts.py).

This inventory records every directly declared field in the baseline run, policy, inference, source, rollout and recovery models. Shared component behavior remains owned by the existing component models. Internal facts have resolved owners; duplicate routing spellings have one authored owner. Test references below are acceptance coverage, not support for old config syntax.

## `train_config.GlobalTrainingSpecModel`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `read_format` | global_training.read_format | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `training_objective` | global_training.training_objective | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `forward_attention` | global_training.forward_attention | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `context_length` | global_training.context_length | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `target_offset` | global_training.target_offset | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `window_stride` | global_training.window_stride | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `inference_batch_size` | global_training.inference_batch_size | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `batch_size` | training_plan.phases[].batch_size | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `accumulation_steps` | training_plan.phases[].accumulation_steps | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `learning_rate` | training_plan.phases[].optimization.<role>.learning_rate | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `optimizer` | training_plan.phases[].optimization.<role>.optimizer | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `scheduler` | training_plan.phases[].optimization.<role>.scheduler | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `scheduler_step_on` | training_plan.phases[].optimization.<role>.scheduler_step_on | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `reset_optimization_on_phase` | training_plan.phases[].optimizer_state (reset by default; validated continuation) | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `gradient_clip` | training_plan.phases[].optimization.<role>.gradient_clip | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `bert_spec` | global_training.bert_spec | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `next_occurrence_config` | global_training.next_occurrence_config | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `device_max_concat_length` | global_training.device_max_concat_length | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `log_interval` | global_training.log_interval | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `early_stopping_epochs` | global_training.early_stopping_epochs | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `save_interval_epochs` | global_training.save_interval_epochs | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `save_latest_interval_minutes` | global_training.save_latest_interval_minutes | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `save_interval_minutes` | global_training.save_interval_minutes | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `save_interval_batches` | global_training.save_interval_batches | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `save_interval_val_loss` | global_training.save_interval_val_loss | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `calculate_validation_loss_on_initialization` | global_training.calculate_validation_loss_on_initialization | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `resume` | global_training.resume | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `enforce_determinism` | global_training.enforce_determinism | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `distributed` | global_training.distributed | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `load_full_data_to_ram` | global_training.load_full_data_to_ram | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `max_ram_gb` | global_training.max_ram_gb | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `world_size` | global_training.world_size | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `num_workers` | global_training.num_workers | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `backend` | global_training.backend | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `layer_type_dtypes` | global_training.layer_type_dtypes | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `layer_autocast` | global_training.layer_autocast | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `data_parallelism` | global_training.data_parallelism | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `fsdp_cpu_offload` | global_training.fsdp_cpu_offload | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `torch_compile` | global_training.torch_compile | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `float32_matmul_precision` | global_training.float32_matmul_precision | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `train_config.ModelInterfaceSpecModel`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `input_columns` | model.interfaces.<interface>.input_columns | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `policy` | policy_bindings.<interface> | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `prediction_distributions` | model.interfaces.<interface>.prediction_distributions | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `static_conditioning` | model.interfaces.<interface>.static_conditioning | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `preconditioning_image_encoder` | model.interfaces.<interface>.preconditioning_image_encoder | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `target_columns` | model.interfaces.<interface>.target_columns | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `categorical_hashing` | model.interfaces.<interface>.categorical_hashing | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `categorical_decoder_special_tokens` | model.interfaces.<interface>.categorical_decoder_special_tokens | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `feature_layout` | model.interfaces.<interface>.feature_layout | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `ingestion` | model.interfaces.<interface>.ingestion | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `decoder` | model.interfaces.<interface>.decoder | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `train_config.ModelSpecModel`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `backbone` | backbone | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `interfaces` | interfaces | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## Policy run fields (authored through `SequifierConfig`)

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `project_root` | project_root | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `model_name` | model_name | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `device` | device | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `seed` | seed | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `enforce_determinism` | global_training.enforce_determinism | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `model` | model | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `policy_binding` | policy_bindings.<interface> | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `execution_model` | resolved model execution | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `policy_interface` | dataset/phase.model_interface | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `context_length` | global_training.context_length | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `units` | policy_bindings.<interface>.units | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `initialize_from` | initialize_from | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `normalization` | global_training.normalization | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `sources` | dataset_training.<dataset>.parts.<part> (sequence) | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `preferences` | dataset_training.<dataset>.parts.<part> (preferences) | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `phases` | training_plan.phases | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `validation_fraction` | global_training.validation_fraction | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `episode_cache_size` | global_training.episode_cache_size | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `resume` | global_training.resume | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `model_path` | model_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `report_path` | report_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `checkpoint_path` | global_training.resume.checkpoint_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `phase_directory` | phase_directory | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `train_config.DatasetPartSpecModel`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `metadata_config_path` | metadata_config_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `image_path` | image_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `preconditioning_image_path` | preconditioning_image_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `file_order` | file_order | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `train_config.DatasetFreezingSpecModel`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `backbone` | backbone | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `ingestion` | ingestion | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `ingestion_adapter` | ingestion_adapter | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `decoder` | decoder | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `train_config.DatasetTrainingSpecModel`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `model_interface` | model_interface | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `parts` | parts | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `criterion` | criterion | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `class_weights` | class_weights | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `loss_weights` | loss_weights | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `class_share_log_columns` | class_share_log_columns | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `freeze` | freeze | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `train_config.TrainingSourceSpecModel`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `source` | source | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `weight` | weight | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `batches_per_selection` | batches_per_selection | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `train_config.TrainingPhaseSpecModel`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `algorithm` | algorithm | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `name` | name | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `epochs` | epochs | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `mode` | mode | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `selection` | selection | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `sources` | sources | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `train_config.TrainingPlanModel`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `curriculum_training` | curriculum_training | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `curriculum_column` | curriculum_column | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `phases` | phases | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `train_config.EvaluationMonitorSpecModel`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `source` | source | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `metric` | metric | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `mode` | mode | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `train_config.EvaluationSpecModel`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `sources` | sources | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `monitor` | monitor | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `train_config.SequifierConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `project_root` | project_root | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `model_name` | model_name | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `device` | device | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `seed` | seed | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `global_training` | global_training | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `model` | model | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `dataset_training` | dataset_training | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `training_plan` | training_plan | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `evaluation` | evaluation | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `export_generative_model` | export_generative_model | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `export_embedding_model` | export_embedding_model | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `embedding_layer_names` | embedding_layer_names | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `export_onnx` | export_onnx | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `export_pt` | export_pt | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `export_with_dropout` | export_with_dropout | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## Mixed run fields (authored through `SequifierConfig`)

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `run_type` | removed dispatch hint; phase algorithms determine execution | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `project_root` | project_root | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `model_name` | model_name | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `device` | device | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `seed` | seed | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `model` | model | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `initialize_from` | initialize_from | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `global_training` | global_training | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `dataset_training` | dataset_training | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `evaluation` | evaluation | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `policy_interface` | dataset/phase.model_interface | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `policy_bindings` | policy_bindings | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `sources` | dataset_training.<dataset>.parts.<part> (sequence) | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `preferences` | dataset_training.<dataset>.parts.<part> (preferences) | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `units` | policy_bindings.<interface>.units | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `phases` | training_plan.phases | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `resume` | global_training.resume | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `validation_fraction` | global_training.validation_fraction | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `episode_cache_size` | global_training.episode_cache_size | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `checkpoint_path` | global_training.resume.checkpoint_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `model_path` | model_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `report_path` | report_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `phase_directory` | phase_directory | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `export_onnx` | export_onnx | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_config.ReturnDefinition`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `gamma` | gamma | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `scale` | scale | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `finite_horizon` | finite_horizon | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_config.ActionFieldConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `type` | type | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `values` | values | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `scale` | scale | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `learn_scale` | learn_scale | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `min_scale` | min_scale | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `max_scale` | max_scale | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `lower` | lower | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `upper` | upper | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `active_when` | active_when | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `inactive_value` | inactive_value | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `atoms` | atoms | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_config.PolicySchemaConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `alignment` | resolved from explicit model inputs | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `observation_columns` | model input columns / resolved binding | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `static_columns` | model static conditioning / resolved binding | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `categorical_columns` | metadata types / resolved binding | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `observation_n_classes` | metadata.n_classes | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `static_n_classes` | metadata.n_classes | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `id_maps` | metadata.id_maps | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `categorical_encoding` | resolved fixed codec convention | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `action_column` | named actions key | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `action_values` | actions.<name>.values | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `actions` | policy_bindings.<interface>.actions | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `action_order` | resolved decoder/activity dependency order | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `decision_position` | policy_bindings.<interface>.decision_position | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `previous_action_inputs` | policy_bindings.<interface>.previous_action_inputs | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `initial_inputs` | policy_bindings.<interface>.initial_inputs | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `input_schema` | resolved ExecutionSchema | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `input_statistics` | metadata / resolved input statistics | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `input_encoding` | dataset part encoding | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `return_conditioned` | policy_bindings.<interface>.return_conditioned | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `initial_desired_return` | policy_bindings.<interface>.initial_desired_return | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `returns` | policy_bindings.<interface>.returns | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_config.CategoricalPolicyConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `context_length` | resolved model context | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `interface_name` | resolved interface name | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_plan.PhaseConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `name` | training_plan.phases[].name | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `handoff` | training_plan.phases[].handoff | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `model_interface` | training_plan.phases[].model_interface | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `freeze` | training_plan.phases[].freeze | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `representation` | resolved binding identity | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `optimization` | training_plan.phases[].optimization | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `evaluation_interval` | training_plan.phases[].evaluation_interval | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `checkpoint_interval` | training_plan.phases[].checkpoint_interval | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_plan.BatchedPhaseConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `batch_size` | training_plan.phases[].batch_size | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `accumulation_steps` | training_plan.phases[].accumulation_steps | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_plan.OfflinePhaseConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `sources` | training_plan.phases[].sources | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `validation_sources` | training_plan.phases[].evaluation.sources | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_plan.PolicyImitationPhaseConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `algorithm` | training_plan.phases[].algorithm | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `epochs` | training_plan.phases[].epochs | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `monitor` | training_plan.phases[].evaluation.monitor.metric | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `direction` | training_plan.phases[].evaluation.monitor.mode | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_plan.IQLPhaseConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `monitor` | training_plan.phases[].evaluation.monitor.metric | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `direction` | training_plan.phases[].evaluation.monitor.mode | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `algorithm` | training_plan.phases[].algorithm | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `name` | training_plan.phases[].name | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `updates` | training_plan.phases[].updates | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `batch_size` | training_plan.phases[].batch_size | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `gamma` | training_plan.phases[].gamma | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `expectile` | training_plan.phases[].expectile | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `inverse_temperature` | training_plan.phases[].inverse_temperature | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `max_weight` | training_plan.phases[].max_weight | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `target_update_rate` | training_plan.phases[].target_update_rate | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `evaluation_interval` | training_plan.phases[].evaluation_interval | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `checkpoint_interval` | training_plan.phases[].checkpoint_interval | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_plan.OnlineEnvironmentConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `factory` | factory | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `config` | config | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `protocol_version` | fixed by adapter type; checked against implementation | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `implementation_id` | implementation_id | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `units` | units | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_plan.FrozenNormalizationConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `source` | source | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `statistics` | statistics | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_plan.EvaluationEpisodesConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `environment` | environment | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `seeds` | seeds | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `max_episode_steps` | max_episode_steps | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `timeout_seconds` | timeout_seconds | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `selection` | selection | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_plan.OnlineEvaluationConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `metric` | evaluation.monitor.metric | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `direction` | evaluation.monitor.mode | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_plan.PolicyResumeConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `policy` | global_training.resume.policy | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `mode` | global_training.resume.mode | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_plan.PPOPhaseConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `algorithm` | training_plan.phases[].algorithm | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `name` | training_plan.phases[].name | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `environment` | training_plan.phases[].environment | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `evaluation` | training_plan.phases[].evaluation | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `normalization` | training_plan.phases[].normalization | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `seed_policy` | training_plan.phases[].seed_policy | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `total_env_steps` | training_plan.phases[].total_env_steps | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `rollout_length` | training_plan.phases[].rollout_length | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `allow_partial_rollout` | training_plan.phases[].allow_partial_rollout | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `optimization_epochs` | training_plan.phases[].optimization_epochs | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `minibatch_size` | training_plan.phases[].minibatch_size | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `accumulation_steps` | training_plan.phases[].accumulation_steps | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `gamma` | training_plan.phases[].gamma | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `gae_lambda` | training_plan.phases[].gae_lambda | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `clip_epsilon` | training_plan.phases[].clip_epsilon | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `value_clip` | training_plan.phases[].value_clip | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `value_coefficient` | training_plan.phases[].value_coefficient | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `entropy_coefficient` | training_plan.phases[].entropy_coefficient | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `target_kl` | training_plan.phases[].target_kl | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `normalize_advantages` | training_plan.phases[].normalize_advantages | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `reward_scale` | training_plan.phases[].reward_scale | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `archive_directory` | training_plan.phases[].archive_directory | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `archive_scores` | training_plan.phases[].archive_scores | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `max_buffer_bytes` | training_plan.phases[].max_buffer_bytes | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_plan.ReferenceConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `source` | source | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `path` | path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_plan.DPOPhaseConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `algorithm` | training_plan.phases[].algorithm | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `preferences` | training_plan.phases[].preferences | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `validation_preferences` | training_plan.phases[].evaluation.sources | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `updates` | training_plan.phases[].updates | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `beta` | training_plan.phases[].beta | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `scoring_chunk_size` | training_plan.phases[].scoring_chunk_size | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `reference` | training_plan.phases[].reference | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `monitor` | training_plan.phases[].evaluation.monitor.metric | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `direction` | training_plan.phases[].evaluation.monitor.mode | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_plan.GRPOEvaluationConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `metric` | evaluation.monitor.metric | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `direction` | evaluation.monitor.mode | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_plan.GRPOPhaseConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `scoring_chunk_size` | training_plan.phases[].scoring_chunk_size | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `algorithm` | training_plan.phases[].algorithm | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `environment` | training_plan.phases[].environment | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `evaluation` | training_plan.phases[].evaluation | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `normalization` | training_plan.phases[].normalization | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `reference` | training_plan.phases[].reference | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `total_groups` | training_plan.phases[].total_groups | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `groups_per_update` | training_plan.phases[].groups_per_update | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `group_size` | training_plan.phases[].group_size | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `max_steps` | training_plan.phases[].max_steps | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `max_env_steps` | training_plan.phases[].max_env_steps | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `max_buffer_bytes` | training_plan.phases[].max_buffer_bytes | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `optimization_epochs` | training_plan.phases[].optimization_epochs | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `clip_epsilon` | training_plan.phases[].clip_epsilon | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `advantage_epsilon` | training_plan.phases[].advantage_epsilon | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `kl_coefficient` | training_plan.phases[].kl_coefficient | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `old_policy_target_kl` | training_plan.phases[].old_policy_target_kl | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `reward_scale` | training_plan.phases[].reward_scale | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `archive_directory` | training_plan.phases[].archive_directory | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_train_config.PolicyTableColumns`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `sequence_id` | dataset part columns.sequence_id | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `item_position` | dataset part columns.item_position | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `observations` | dataset part columns.observations | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `static_inputs` | dataset part columns.static_inputs | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `action` | dataset part columns.action | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `actions` | dataset part columns.actions | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `previous_actions` | dataset part columns.previous_actions | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `reward` | dataset part columns.reward | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `terminated` | dataset part columns.terminated | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `truncated` | dataset part columns.truncated | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `previous_action` | dataset part columns.previous_action | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `successors` | dataset part columns.successors | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `bootstrap_allowed` | dataset part columns.bootstrap_allowed | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `group` | dataset part columns.group | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `episode_id` | dataset part columns.episode_id | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_train_config.SequenceSourceConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `project_root` | resolved root | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `units` | dataset_training.<dataset>.parts.<part>.units | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `paths` | dataset_training.<dataset>.parts.<part>.paths | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `manifest` | dataset_training.<dataset>.parts.<part>.manifest | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `format` | dataset_training.<dataset>.parts.<part>.format | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `columns` | dataset_training.<dataset>.parts.<part>.columns | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `episode_end` | dataset_training.<dataset>.parts.<part>.episode_end | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `episode_namespace` | dataset_training.<dataset>.parts.<part>.episode_namespace | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `metadata_config_path` | dataset_training.<dataset>.parts.<part>.metadata_config_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `encoding` | dataset_training.<dataset>.parts.<part>.encoding | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `static_data_path` | dataset_training.<dataset>.parts.<part>.static_data_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `split_index` | dataset_training.<dataset>.parts.<part>.split_index | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `action_semantics` | dataset_training.<dataset>.parts.<part>.action_semantics | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_infer_config.PolicyRolloutConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `environment` | rollout.environment | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `trajectory_path` | rollout.trajectory_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `seed_index` | rollout.seed_row_index | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `max_episode_steps` | rollout.max_episode_steps | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_infer_config.PolicyInferenceConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `project_root` | project_root | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `model_path` | model_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `execution_mode` | model_type | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `model_type` | model_type | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `model_interface` | model_interface | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `data_path` | data_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `read_format` | read_format | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `write_format` | write_format | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `output_path` | output_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `output_probabilities` | output_probabilities | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `inference_batch_size` | inference_batch_size | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `device` | device | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `seed` | seed | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `enforce_determinism` | runtime deterministic and policy selection | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `desired_return` | desired_return | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `desired_return_column` | desired_return_column | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `previous_action_column` | previous_action_columns.<single-action> | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `previous_action_columns` | previous_action_columns | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `sequence_id_column` | sequence_id_column | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `item_position_column` | item_position_column | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `rollout` | rollout | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `autoregressive` | explicit rollout for environment interaction | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `generation_steps` | rollout.max_steps | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `rollout_config.InputBinding`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `input` | input | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `source` | source | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `key` | key | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `rollout_config.AdapterConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `factory` | factory | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `config` | config | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `protocol_version` | fixed by adapter type; checked against implementation | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `rollout_config.RolloutConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `mode` | rollout.type | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `environment` | rollout.environment | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `bindings` | rollout.bindings | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `trajectory_path` | rollout.trajectory_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `seed_index` | rollout.seed_window_index | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `max_episode_steps` | rollout.max_episode_steps | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `sequence_import_config.SequenceImportConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `operation` | sequence import operation | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `project_root` | sequence import project_root | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `source` | sequence import source | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `policy` | policy_binding + source/model metadata | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `output_path` | sequence import output_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `episodes_per_shard` | sequence import episodes_per_shard | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `policy_optimization.OptimizationConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `optimizer` | optimizer | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `scheduler` | scheduler | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `learning_rate` | learning_rate | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `gradient_clip` | gradient_clip | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `infer_config._InferenceConfigBase`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `execution_mode` | execution_mode | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `project_root` | project_root | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `preprocessing_data_path` | preprocessing_data_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `metadata_config_path` | metadata_config_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `model_path` | model_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `model_type` | model_type | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `training_objective` | training_objective | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `data_path` | data_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `training_config_path` | training_config_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `dataset` | dataset | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `part` | part | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `model_interface` | model_interface | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `read_format` | read_format | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `write_format` | write_format | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `input_columns` | input_columns | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `static_columns` | static_columns | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `static_features_path` | static_features_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `image_path` | image_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `preconditioning_image_path` | preconditioning_image_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `target_columns` | target_columns | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `column_data_types` | column_data_types | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `target_column_types` | target_column_types | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `enforce_determinism` | deterministic | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `output_probabilities` | output_probabilities | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `decode_categories` | decode_categories | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `seed` | seed | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `device` | device | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `context_length` | context_length | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `target_offset` | target_offset | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `window_stride` | window_stride | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `prediction_length` | prediction_length | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `decoder_support` | decoder_support | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `prediction_alignment_steps` | prediction_alignment_steps | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `inference_batch_size` | inference_batch_size | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `sample_from_distribution_columns` | sample_from_distribution_columns | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `infer_with_dropout` | infer_with_dropout | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `rollout` | rollout | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `autoregressive` | autoregressive | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `generation_steps` | generation_steps | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `infer_config.InferenceConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `data_path` | data_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `column_data_types` | column_data_types | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `target_column_types` | target_column_types | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## `components.ResumeConfig`

| Baseline field | New owner / behavior | Acceptance coverage |
| --- | --- | --- |
| `policy` | global_training.resume.policy | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |
| `checkpoint_path` | global_training.resume.checkpoint_path | ../sequifier-test/local/unit/test_public_config.py; ../sequifier-test/local/integration/test_public_config_execution.py |

## Operation and combination coverage

In addition to field coverage, acceptance tests exercise same-design recovery, native supervised execution and initialization, mixed supervised/imitation/supervised handoff, single categorical and return-conditioned policies, IQL/PPO/DPO/GRPO updates, raw/manifest/prepared sources, explicit and automatic holdouts, custom inference outputs and probabilities, and both environment protocols. Model component, distribution, HPO and source-import cases are covered in their focused suites.

Completion requires all referenced tests to pass and every supported baseline combination to retain an execution path. Rejection of already supported combinations is a regression, even if individual fields still parse.
