<img src="./design/sequifier.png">


### one-to-one and many-to-one autoregression made easy

Sequifier enables sequence classification or regression for time based sequences using transformer models, via CLI.
The specific configuration of preprocessing, which takes a single or multi-variable columnar data file and creates
training, validation and test sequences, training, which trains a transformer model, and inference, which calculates
model outputs for data (usually the test data from preprocessing), is done via configuration yaml files.

\
\
\
## Overview
The sequifier package enables:
  - the extraction of sequences for training
  - the configuration and training of a transformer classification or regression model
  - using multiple input and output sequences
  - inference on data with a trained model


## Other materials
If you want to first get a more specific understanding of the transformer architecture, have a look at
the [Wikipedia article.](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model))

If you want to see a benchmark on a small synthetic dataset with 10k cases, agains a random forest,
an xgboost model and a logistic regression, check out [this notebook.](./documentation/demos/benchmark-small-data.ipynb)


## Complete example how to build and apply a transformer sequence classifier with sequifier

1. create a conda environment with python >=3.9 activate and run
```console
pip install sequifier
```
2. To create the project folder with the config templates in the configs subfolder, run
```console
sequifier make YOUR_PROJECT_NAME
```
3. cd into the `YOUR_PROJECT_NAME` folder, create a `data` folder and add your data and adapt the config file `preprocess.yaml` in the configs folder to take the path to the data
4. run
```console
sequifier preprocess
```
5. the preprocessing step outputs a "data driven config" at `configs/ddconfigs/[FILE NAME]`. It contains the number of classes found in the data, a map of classes to indices and the oaths to train, validation and test splits of data. Adapt the `dd_config` parameter in `train.yaml` and `infer.yaml` in to the path `configs/ddconfigs/[FILE NAME]`
6. Adapt the config file `train.yaml` to specify the transformer hyperparameters you want and run
```console
sequifier train
```
7. adapt `data_path` in `infer.yaml` to one of the files output in the preprocessing step
8. run
```console
sequifier infer
```
9. find your predictions at `[PROJECT PATH]/outputs/predictions/sequifier-default-best-predictions.csv`


## More detailed explanations of the three steps
#### Preprocessing of data into sequences for training

The preprocessing step is designed for scenarios where for timeseries or timeseries-like data,
the prediction of the next data point of one or more variables from prior values of these
variables and (optionally) other variables is of interest.

This step presupposes input data with three columns: "sequenceId" and "itemPosition", and a column
with the variable that is the prediction target.
"sequenceId" separates different sequences and the itemPosition column
provides values that enable sequential sorting. Often this will simply be a timestamp.
You can find an example of the preprocessing input data at [documentation/example_inputs/preprocessing_input.csv](./documentation/example_inputs/preprocessing_input.csv)

The data can then be processed and split into training, validation and testing datasets of all
valid subsequences in the original data with the command:

```console
sequifier preprocess --config_path=[CONFIG PATH]
```

The config path specifies the path to the preprocessing config and the project
path the path to the (preferably empty) folder the output files of the different
steps are written to.

The default config can be found on this path:

[configs/preprocess.yaml](./configs/preprocess.yaml)



#### Configuring and training the sequence classification model

The training step is executed with the command:

```console
sequifier train --config_path=[CONFIG PATH]
```

If the data on which the model is trained DOES NOT come from the preprocessing step, the flag

```console
--on-unprocessed
```

should be added.

If the training data does not come from the preprocessing step, both train and validation
data have to take the form of a csv file with the columns "sequenceId", "subsequenceId", "col_name", [SEQ LENGTH], [SEQ LENGTH - 1],...,"1", "0".
You can find an example of the preprocessing input data at [documentation/example_inputs/training_input.csv](./documentation/example_inputs/training_input.csv)

The training step is configured using the config. The two default configs can be found here:

[configs/train.yaml](./configs/train.yaml)

depending on whether the preprocessing step was executed.


#### Inferring on test data using the trained model

Inference is done using the command:

```console
sequifier infer --config_path=[CONFIG PATH]
```

and configured using a config file. The default version can be found here:

[configs/infer.yaml](./configs/infer.yaml)
