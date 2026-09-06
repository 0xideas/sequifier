This page contains the auto-generated API reference documentation.

Stable integration API
----------------------

``sequifier.api`` is the stable model boundary for sibling packages. It exposes
the composable network, portable artifacts, explicit encode/decode and tracing
types, canonical parameter naming, and parameter catalogs without exposing the
training lifecycle.

.. automodule:: sequifier.api
   :members:
   :undoc-members:

Training integration API
------------------------

Update-aware integrations use ``sequifier.training_api`` for optimization,
step identity, directives, distributed strategies, and run state.

.. automodule:: sequifier.training_api
   :members:
   :undoc-members:


Preprocessing Config
-------------------------
.. automodule:: sequifier.config.preprocess_config
   :members: PreprocessorModel

Training Config
---------------------
.. automodule:: sequifier.config.train_config
   :members: SequifierConfig, ResolvedSequifierConfig, TrainModel, GlobalTrainingSpecModel, ModelSpecModel, ModelInterfaceSpecModel, DatasetTrainingSpecModel, TrainingPlanModel, LoadedTrainConfig, load_train_config, load_train_config_with_source, resolve_sequifier_config

Inference Config
---------------------
.. automodule:: sequifier.config.infer_config
   :members: InferenceConfig, ResolvedInferenceConfig, InfererModel, resolve_inference_config

Config Composition and Metadata
---------------------------------
.. automodule:: sequifier.config.composition
   :members:

.. automodule:: sequifier.config.metadata
   :members:

Hyperparameter Search Config
---------------------------------
.. automodule:: sequifier.config.hyperparameter_search_config
   :members: CanonicalHyperparameterSearchConfig, HyperparameterSearchConfig, load_hyperparameter_search_config, compile_canonical_hyperparameter_search_config

Non-standard Optimizers
--------------------------
.. automodule:: sequifier.optimizers.ademamix
   :members:


Internals
------------

.. automodule:: sequifier.sequifier
   :members:

.. automodule:: sequifier.preprocess
   :members:

.. automodule:: sequifier.train
   :members:

.. automodule:: sequifier.infer
   :members:

.. automodule:: sequifier.make
   :members:

.. automodule:: sequifier.hyperparameter_search
   :members:

.. automodule:: sequifier.helpers
   :members:


.. automodule:: sequifier.io.yaml
   :members:

.. automodule:: sequifier.io.sequifier_dataset_from_folder_pt
   :members:

.. automodule:: sequifier.io.sequifier_dataset_from_folder_pt_lazy
   :members:

.. automodule:: sequifier.io.sequifier_dataset_from_folder_parquet
   :members:

.. automodule:: sequifier.io.sequifier_dataset_from_folder_parquet_lazy
   :members:

.. automodule:: sequifier.io.sequifier_dataset_from_file
   :members:

.. automodule:: sequifier.io.window_sampling
   :members:

.. automodule:: sequifier.optimizers.optimizers
   :members:
