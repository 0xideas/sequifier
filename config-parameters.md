#### Preprocessing

- `project_path`: the path in which the sequifier project is configured and executed. To set up a sequifier project, run `sequifier make PROJECT_NAME`, and then PROJECT_NAME is the root directory for that sequifier project. Usually, you would execute the subsequent sequifier commands from that directory, and then the `project_path` is `.`
- `data_path`: the path to the file that contains the input data to the preprocessing step
- `read_format`: the file type of the input data, currently 'csv' and 'parquet' are accepted. Default is 'csv'
- `write_format`: the file type of the preprocessed data, i.e. the input data to the training step. Default is 'parquet'
- `selected_columns`: a list of input columns that should be preprocessd into the training data format. Especially when the input data has a lot of columns that will not be used in modelling, it is worth setting the columns to be included in the training data. If this value isn't set, all columns are included.
- `group_proportions`: in training and evaluating models, it is always necessary to split the data into training, evaluation and test data sets. With this parameter, the relative size of these groups is configured. The 'standard' approach is to use the group created from the first value as the training dataset, the second as the validation and the third as the test dataset. If fewer or more splits are desired, this is also possible by passing the corresponding number of 'proportion' values. The data is split per sequence, so that, for example, the first 80% of each sequence is used for training, the subsequent 10% for validation and the last 10% for testing. If training, validation and test sets should be each composed of different sequences, the value '1.0' has to be passed here, and the preprocessing step output has to be manually split into these datasets afterwards.
- `seq_length`: the sequence length of the preprocessed data, which is usually identical to the sequence length of the model input in the training step. If a range of model input sequence lengths need to be evaluated, the largest of these values should be passed here, so that preprocessing doesn't have to be repeated subsequently
- `seq_step_sizes`: gives the relative index of each subsequent subsequence, for each "group" in `group_proportions`. For example, if it is set to 1, each subsequence starts one row after the previous one. If set to `seq_length`, there is no or minimal overlap between subsequences. The overlap is non-zero when the sequence length is not divisible by `seq_length`, as the subsequences are then aranged so that no values of the sequence are ignored. The default value of `seq_step_size` is `seq_length` for all groups. If provided, it has to be a list of the same length as `group_proportions`
- `max_rows`: the maximum number of input rows to process, in case of very large data and when only a subset of the data is needed, for example to validate a preprocessing configuration. Default is 'None', in which case all rows are processed
- `seed`: random seed used with numpy
- `n_cores`: the number of cores used in parallel to preprocess the data. If left empty, it uses the number of available CPU cores. Occasionally, parallel processing leads to bugs, in which case it is worth setting `n_cores` to 1, even if it takes longer.

#### Training

##### General training config parameters

- `project_path`: same as in preprocessing, sets the path to the project directory relative to the execution location in the filesystem. Should usually be `.`
- `model_name`: name of the model being trained, used in various file paths (e.g. logging, model checkpoints, model weight outputs)
- `read_format`: file format of input data, 'csv' and 'parquet' are allowed, default is 'parquet'
- `ddconfig_path`: 'ddconfig' stands for 'data driven config', which is a json file created through the preprocessing step that contains various metadata. The path to the ddconfig is typically 'PROJECT_FOLDER/configs/ddconfig/DATASET_NAME.json'
- `selected_columns`: input columns to train the model on. Note that these are 'columns' in the input data to the preprocessing step, in the input to the training data these are values of the 'inputCol' column
- `target_columns`: target columns for model training. Note that these are 'columns' in the input data to the preprocessing step, in the input to the training data these are values of the 'inputCol' column
- `target_column_types`: the column types of the target columns. Each target column needs to be mapped to 'categorical' or 'real', as these are the two options
- `seq_length`: the sequence length of the input sequences to the model
- `inference_batch_size`: the batch size used in the model after export. This is mainly relevant for the onnx model format
- `seed`: seed set for numpy and pytorch
- `export_onnx`: export ONNX format model, defaults to True
- `export_pt`: export pytorch model with `torch.save`, defaults to False
- `export_with_dropout`: export model with dropout, applies only to torch.save export, defaults to False
- `model_spec`: the specification of the transformer model architecture, beyond the input and target columns
- `training_spec`: the specification of the training run configuration

##### Model Spec

- `d_model`: the number of expected features in the input (unless d_model is smaller than `nhead`, in which case `nhead` is used
- `d_model_by_column`: the embedding dimensions for each input column. They mus sum to `d_model`. Defaults to 'None', in which case embedding space is automatically equally distributed between input columns.
- `nhead`: the number of heads in the multiheadattention models
- `d_hid`: the dimension of the feedforward network model
- `nlayers`: the number of layers of the transformer model

##### Training Spec

- `device`: the torch.device to train the model on, 'cuda', 'cpu' or 'mps' for apple silicon devices
- `epochs`: number of epochs
- `iter_save`: the interval in epochs for checkpointing
- `batch_size`: training batch size
- `log_interval`: the interval in batches for logging
- `class_share_log_columns`: a list of column names for which the class share of the predictions on the validation set should be logged, defaults to all categorical columns
- `lr`: learning rate
- `accumulation_steps`: if gradient accumulation should be used, set to something higher than 1
- `early_stopping_epochs`: number of epochs to wait for a validation loss that improves on the existing minimum. If this happens, counting starts from 0, if it doesn't, training stops. Defaults to 'None', in which case no early stopping is used.
- `dropout`: the dropout value of the transformer model
- `criterion`: the map from each target column to the loss function to be applied. All pytorch loss functions can be used, but the loss function must work with the column type it is applied to (i.e. categorical or real)
- `optimizer`: the optimizer itself is specified with the 'name' attribute, other optimizer hyperparameters should be specified after
- `scheduler`: the learning rate scheduler. Like with the optimizer, the scheduler itself can be specified with the 'name' attribute, with other hyperparameters coming after
- `continue_training`: load the last checkpoint with the same model name and continue training from there
- `loss_weights`: map from columns to loss weights, to weight loss of each column separately. Can be necessary when using differently calibrated loss functions for different variables

##### Data driven config parameters

These parameters are typically read from the data driven config file created through the preprocessing step, with the path to that config file passed in the training config. For transparency and in case the preprocessing step isn't used, they are listed here.

- `training_data_path`: path to training data, optional if `ddconfig_path` is passed, in which case it will use the first of the preprocessing output data files if no value is specified in the config file
- `validation_data_path`: path to validation data, optional if `ddconfig_path` is passed, in which case it will use the second-to-last of the preprocessing output data files if no value is specified in the config file
- `column_types`: a map from each column to a numeric type, either 'int64' or 'float64'
- `n_classes`: the number of classes for each categorical column
- `id_maps`: for each categorical column, a map from the distinct values of that column to their index+1, for modelling purposes

#### Inference

- `project_path`: same as in preprocessing, sets the path to the project directory relative to the execution location in the filesystem. Should usually be `.`
- `ddconfig_path`: 'ddconfig' stands for 'data driven config', which is a json file created through the preprocessing step that contains various metadata. The path to the ddconfig is typically 'PROJECT_FOLDER/configs/ddconfig/DATASET_NAME.json'
- `model_path`: path to model output by training step
- `data_path`: path to data to infer on, typically the last data set from the preprocessing step output
- `read_format`: the file type of the input data, currently 'csv' and 'parquet' are accepted. Default is 'parquet'
- `write_format`: the file type of the inference output, defaults to 'csv'
- `selected_columns`: model input columns, must be identical to training config
- `target_columns`: model output columns, must be identical to training config
- `target_column_types`: the column types of the target columns. Each target column needs to be mapped to 'categorical' or 'real', as these are the two options. Must be identical to trainign config
- `output_probabilities`: output the probability distributions across categorical values for categorical target columns, default to False
- `map_to_id`: map categorical output values back into the space of input values to the preprocessing step, defaults to True
- `device`: the torch.device to infer the model with, 'cuda', 'cpu' or 'mps' for apple silicon devices. Values other than 'cpu' for onnx models depend on GPU tooling for onnx
- `seq_length`: the input sequence length to the model, must be the same as in the training config
- `inference_batch_size`: batch size for inference, must be the same as in the training config for onnx models, can be different for torch.save models
- `autoregression`: infer every step after the first step from predicted values, rather than ground truth values, defaults to True
- `autoregression_extra_steps`: infer not only for the steps in each sequence, but add N additional steps. Only works with autoregressive inference
- `training_config_path`: path to the training config, must be passed to infer torch.save models
- `infer_with_dropout`: apply dropout active during inference, works only with torch.save models, defaults to False

There are additional parameters to the inference step that are typically read from the data driven config, as listed above. If the data driven config path isn't passed, these need to be set manually for inference as well.
