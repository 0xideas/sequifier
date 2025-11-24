<img src="./design/sequifier.png">


### What is sequifier?

Sequifier makes training and inference of powerful causal transformer models fast and trustworthy.

The process looks like this:

<img src="./design/sequifier-illustration.png">


### Value Proposition

Implementing a model from scratch takes time, and there are a surprising number of aspects to consider. The idea is: why not do it once, make it configurable, and then use the same implementation across domains and datasets.

This gives us a number of benefits:

- rapid prototyping
- configurable architecture
- trusted implementation (you can't create bugs inadvertedly)
- standardized logging
- native multi-gpu support
- native multi-core preprocessing
- scales to datasets larger than RAM
- hyperparameter search
- can be used for prediction, generation and embeddding on/of arbitrary sequences

The only requirement is having input data in the right format, and having sequifier installed.

### Data Formats

The basic data format that is used as input to the library takes the following form:

|sequenceId|itemPosition|column1|column2|...|
|----------|------------|-------|-------|---|
|0|0|"high"|12.3|...|
|0|1|"high"|10.2|...|
|...|...|...|...|...|
|1|0|"medium"|20.6|...|
|...|...|...|...|...|

The two columns "sequenceId" and "itemPosition" have to be present, and then there must be at least one feature column. There can also be many feature columns, and these can be categorical or real valued.

Data of this input format can be transformed into the format that is used for model training and inference, which takes this form:

|sequenceId|subsequenceId|startItemPosition|columnName|[Subsequence Length]|[Subsequence Length - 1]|...|0|
|----------|-------------|-----------------|----------|--------------------|------------------------| - |-|
|0|0|0|column1|"high"|"high"|...|"low"|
|0|0|0|column2|12.3|10.2|...|14.9|
|...|...|...|...|...|...|...|...|
|1|0|15|column1|"medium"|"high"|...|"medium"|
|1|0|15|column2|20.6|18.5|...|21.6|
|...|...|...|...|...|...|...|...|

On inference, the output is returned in the library input format, introduced first.

|sequenceId|itemPosition|column1|column2|...|
|----------|------------|-------|-------|---|
|0|963|"medium"|8.9|...|
|0|964|"low"|6.3|...|
|...|...|...|...|...|
|1|732|"medium"|14.4|...|
|...|...|...|...|...|

There are five standalone commands within sequifier: `make`, `preprocess`, `train`, `infer` and `hyperparameter-search`. `make` sets up a new sequifier project in a new folder, `preprocess` preprocesses the data from the input format into subsequences of a fixed length, `train` trains a model on the preprocessed data, `infer` generates outputs from data in the preprocessed format and outputs it in the initial input format, and `hyperparameter-search` executes multiple training runs to find optimal configurations.

The input data can be a single csv or parquet file, or a folder of csv or parquet files. The preprocessing output can be a csv or parquet file *per split*, or a folder of multiple torch tensor (pt) files *per split*. The training step does not output any data files (it outputs model files and logs). The inference output can be a single csv or parquet file, or a folder of csv and parquet files. In general, it is recommended to store every step as a single file if the initial input is a single file, and a folder of files if the initial data is a folder of files. For the folder "flow", the preprocessing step write format has to be "pt".

## Other materials

To get more details on the specific configuration options, go to these [docs page.](sequifier.com)

If you want to first get a more specific understanding of the transformer architecture, have a look at
the [Wikipedia article.](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model))

If you want to see a benchmark on a small synthetic dataset with 10k cases, agains a random forest,
an xgboost model and a logistic regression, check out [this notebook.](https://www.google.com/search?q=./documentation/demos/benchmark-small-data.ipynb)

## Complete example how to build and apply a transformer sequence classifier with sequifier

1.  create a conda environment with python \>=3.9 activate and run

<!-- end list -->

```console
pip install sequifier
```

2.  To create the project folder with the config templates in the configs subfolder, run

<!-- end list -->

```console
sequifier make YOUR_PROJECT_NAME
```

3.  cd into the `YOUR_PROJECT_NAME` folder, create a `data` folder and add your data and adapt the config file `preprocess.yaml` in the configs folder to take the path to the data
4.  run

<!-- end list -->

```console
sequifier preprocess
```

5.  the preprocessing step outputs a metadata config at `configs/metadata_configs/[FILE NAME]`. It contains the number of classes found in the data, a map of classes to indices and the paths to train, validation and test splits of data. Adapt the `metadata_config_path` parameter in `train.yaml` and `infer.yaml` to the path `configs/metadata_configs/[FILE NAME]`
6.  Adapt the config file `train.yaml` to specify the transformer hyperparameters you want and run

<!-- end list -->

```console
sequifier train
```

7.  adapt `data_path` in `infer.yaml` to one of the files output in the preprocessing step
8.  run

<!-- end list -->

```console
sequifier infer
```

9.  find your predictions at `[PROJECT ROOT]/outputs/predictions/sequifier-default-best-predictions.csv`


### Embedding Model

While Sequifier's primary use case is training predictive or generative causal transformer models, it also supports the export of embedding models.

Configuration:
- Training: Set export_embedding_model: true in the training config.
- Inference: Set model_type: embedding in the inference config.

Technical Details: The generated embedding has dimensionality `dim_model` and consists of the final hidden state (activations) of the transformer's last layer corresponding to the last token in the sequence. Because the model is trained on a causal objective, this is a "forward-looking" embedding: it is optimized to compress the sequence history into a representation that maximizes information about the future state of the data.


### Hyperparameter Search

Sequifier supports hyperparameter search (grid search or random sampling) via a dedicated command.

```console
sequifier hyperparameter-search --config-path=[CONFIG PATH]
```

You must provide a hyperparameter search config file.


### Distributed Training

Sequifier supports distributed training using torch `DistributedDataParallel`. To make use of multi gpu support, the write format of the preprocessing step must be set to 'pt' and `merge_output` must be set to `false` in the preprocessing config.

## Citation

Please cite with:

```bibtex
@software{sequifier_2025,
  author = {Luithlen, Leon},
  title = {sequifier - causal transformer models for multivariate sequence modelling},
  year = {2025},
  publisher = {GitHub},
  version = {v1.0.0.0},
  url = {https://github.com/0xideas/sequifier}
}
```
