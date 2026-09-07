<img src="./design/sequifier.png">


## What is sequifier?

Sequifier is the short, efficient, scalable path from tabular sequences to your own transformer model.

It offers three core commands `preprocess`, `train` and `infer`, each of them configurable, fully parallelised and tested exhaustively.

They enable you to go from multivariate sequence data to a model that ingests such data, and emits either a) this data, b) a subset c) other variables d) embeddings.

If input and target variables are the same, it supports full autoregressive inference.

Through configuration, most modern architectural variants such as RoPE, GQA, SwiGLU, and RMSNorm are supported. The ambition is to follow the frontier when industrial-grade implementations become available.

The process looks like this:

<img src="./design/sequifier-illustration.png">

### Applications

Multivariate tabular transformers have *many* applications. Here are a few:

Finance:
- Transaction Foundation Models
- Order Book Transformers [TransLOB](https://arxiv.org/abs/2003.00130)
- Volatility Forecasting ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0952197624003816))
- Macroeconomic Models ([BISTRO](https://www.bis.org/publications/bistro-general-purpose-oracle-macroeconomic-time-series))

Health/Bio:
- Health Records ([BEHRT](https://www.nature.com/articles/s41598-020-62922-y), [Med-BERT](https://www.nature.com/articles/s41746-021-00455-y))
- Treatment Outcome Prediction ([G-Transformer](https://arxiv.org/abs/2406.05504))
- Health Trajectory Prediction ([ETHOS](https://www.nature.com/articles/s41746-024-01235-0))
- Glucose Forecasting ([GluForecast](https://arxiv.org/html/2606.18640v1))
- Cardiovascular Monitoring ([Heart Language Model](https://www.nature.com/articles/s41598-024-84270-x))

Cybersecurity:
- Network Intrusion Detection ([paper](https://arxiv.org/pdf/2309.01070))
- Encrypted Traffic Analysis ([Criss-Cross Traffic Transformer](https://ieeexplore.ieee.org/document/11396346))
- Insider Threat Detection ([paper](https://www.nature.com/articles/s41598-025-12063-x))
- In-Vehicle Intrusion Detection ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705124007251))
- IoT Anomaly Detection ([paper](https://www.sciopen.com/article/10.32604/cmc.2024.053765))
- System Logs Anomaly Detection([DeepEAD](https://ieeexplore.ieee.org/document/10279563))

Industrial & IoT:
- Remaining Useful Life Estimation ([paper](https://www.sciencedirect.com/science/article/abs/pii/S095219762400633X))
- Fault Diagnosis ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0957582026001485))
- Anomaly Detection for Industrial Control Systems ([paper](https://onlinelibrary.wiley.com/doi/10.1155/2024/5459452))
- Soft Sensing ([Debutanizer](https://www.sciencedirect.com/science/article/abs/pii/S1876107024003249))
- Battery Management ([DS-transformer](https://www.nature.com/articles/s41598-026-52202-6))
- Production Line Modelling ([paper](https://www.sciencedirect.com/science/article/pii/S2212827126007791?))

Agriculture & Environment:
- Soil Moisture Forecasting ([paper](https://link.springer.com/chapter/10.1007/978-3-032-19763-4_18))
- Crop Water Demand ([paper](https://www.mdpi.com/2073-4395/12/3/656))
- Rainfall–runoff modelling ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0022169422003560))
- Drought Forecasting ([paper](https://www.sciencedirect.com/science/article/abs/pii/S1364815225000787))
- Land Surface Dynamics ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0022169425002446))

Neuroscience:
- Neural Population Dynamics ([Neural Data Transformer](https://pmc.ncbi.nlm.nih.gov/articles/PMC10541112/))
- Neural + Motor Modelling ([Intracortical Motor Decoder](https://www.biorxiv.org/content/10.1101/2025.02.02.634313v1))
- fMRI State Prediction ([paper](https://arxiv.org/abs/2412.19814))
- Seizure Detection ([BIOT](https://arxiv.org/abs/2305.10351))
- Magnetoencephalography Data ([MEG-GPT](https://arxiv.org/abs/2510.18080))

Animal Communication:
- Sperm Whales ([whale-gpt](https://github.com/0xideas/whale-gpt), [paper](https://github.com/0xideas/whale-gpt))
- Bengalese Finches ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8746767/))
- Zebra Finches ([ZF-Aim](https://www.biorxiv.org/content/10.64898/2026.02.12.705387v1))


Sequifier aims to standardise model implementation across these fields, to make them comparable, transfer learnings between domains, and converge on optimal solutions faster.


### Value Proposition

For the individual researcher, sequifier cuts the development time of a model significantly: typically, some preprocessing and the subsequent model evaluation are specific to the modelling problem, but all the steps in between are taken care of.

This enables:

- rapid prototyping on a configurable architecture
- trusted implementation (you can't create bugs inadvertedly)
- scaling preprocessing across cores and training across GPUs and nodes
- hyperparameter optimization using Optuna (Bayesian, Random, or Grid search)


### The Six Commands

There are six standalone commands within sequifier: `make`, `preprocess`, `train`, `infer`, `hyperparameter-search`, and `visualize-training`.

`make` sets up a new sequifier project in a new folder, `preprocess` preprocesses the data from the input format into subsequences of a fixed length, `train` trains a model on the preprocessed data, `infer` generates predictions, probabilities, or embeddings from data in the preprocessed format, `hyperparameter-search` executes multiple training runs using Optuna to find optimal configurations, and `visualize-training` reads structured training metrics to generate interactive HTML plots of your loss curves.

There are documentation pages for each command, except make:

 - [preprocess documentation](./documentation/configs/preprocess.md)
 - [train documentation](./documentation/configs/train.md)
 - [infer documentation](./documentation/configs/infer.md)
 - [hyperparameter-search documentation](./documentation/configs/hyperparameter-search.md)
 - [visualize-training documentation](./documentation/commands/visualize-training.md)


### Other Materials

To get the full documentation, visit [sequifier.com](https://sequifier.com)

## Structure of a Sequifier Project

Sequifier is designed with a specific folder structure in mind:

```text
YOUR_PROJECT_NAME/
├── configs/
│   ├── preprocess.yaml
│   ├── train.yaml
│   └── infer.yaml
├── data/
│   └── (Place your CSV/Parquet files here)
├── models/
├── checkpoints/
├── outputs/
│   ├── embeddings(?)
│   ├── predictions(?)
│   ├── probabilities(?)
│   └── visualization/
├── logs/
├── state/
└── scripts/

```

The `sequifier` commands should typically be run in the project root.

Within YOUR_PROJECT_NAME, you can also add other folders for additional steps, such as `notebooks` or `scripts` for pre- or postprocessing, and `analysis`, `visualizations` or `evals` for files you generate in other, manual steps.

### Data Transformations in Sequifier

The basic input data format is this:

|sequenceId|itemPosition|column1|column2|...|
|----------|------------|-------|-------|---|
|0|0|"high"|12.3|...|
|0|1|"high"|10.2|...|
|...|...|...|...|...|
|1|0|"medium"|20.6|...|
|...|...|...|...|...|

The two columns "sequenceId" and "itemPosition" have to be present, and there must be one or more feature columns.

`sequifier preprocess` splits sequences into subsequences, normalises real variables and maps categorical variables to integers/tokens. The subsequence length is the sum of `window_length` and `max_target_offset`.

|sequenceId|subsequenceId|startItemPosition|leftPadLength|inputCol|[Subsequence Length- 1]|[Subsequence Length - 2]|...|0|
|----------|-------------|-----------------|-------------|--------|-------------------|-------------------| - |-|
|0|0|0|0|column1|"high"|"high"|...|"low"|
|0|0|0|0|column2|12.3|10.2|...|14.9|
|...|...|...|...|...|...|...|...|...|
|1|0|15|0|column1|"medium"|"high"|...|"medium"|
|1|0|15|0|column2|20.6|18.5|...|21.6|
|...|...|...|...|...|...|...|...|...|

Generative inference returns a row-oriented table with the predicted target
columns plus identifiers for the source sequence and model window:

|sequenceId|subsequenceId|windowStartOffset|itemPosition|column1|column2|...|
|----------|-------------|-----------------|------------|-------|-------|---|
|0|0|0|963|"medium"|8.9|...|
|0|0|0|964|"low"|6.3|...|
|...|...|...|...|...|...|...|
|1|4|0|732|"medium"|14.4|...|
|...|...|...|...|...|...|...|



### Complete Example of Training and Inferring a Transformer Model

Once you have your data in the input format described above, you can train a transformer model in a couple of steps on them.

1.  Create and activate an environment with Python \>=3.10, then run

```console
pip install sequifier
```

2.  To create the project folder with the config templates in the configs subfolder, run

```console
sequifier make YOUR_PROJECT_NAME
```

3.  cd into the `YOUR_PROJECT_NAME` folder, create a `data` folder and add your data and adapt `preprocessing_data_path` in `preprocess.yaml` to point to the data
4.  run

```console
sequifier preprocess
```

5.  the preprocessing step outputs metadata at `configs/metadata_configs/[INPUT BASENAME].json`. For a single dataset and part, reference that file from `dataset.part.metadata_config_path` in `train.yaml`; named configurations use `dataset_training.<dataset>.parts.<part>.metadata_config_path`. Inference may still use `preprocessing_data_path` or `metadata_config_path`
6.  Adapt the config file `train.yaml` to specify the transformer hyperparameters you want and run


```console
sequifier train
```

7.  point `model_path` in `infer.yaml` at the default ONNX export. Keep the
    scaffold's explicit contract, or replace it with `training_config_path` and
    `dataset`; see the [ONNX/PT trade-offs](./documentation/configs/infer.md#onnx-or-pt)
8.  run


```console
sequifier infer
```

9.  find your predictions at `[PROJECT ROOT]/outputs/predictions/[EXPORTED_MODEL_BASENAME]/part-000.[FORMAT]`, for example `outputs/predictions/your-model-best-3/part-000.csv`


## Key Features

### Causal Modelling

### Autoregressive Inference


### Causal Embedding Model

While Sequifier's primary use case is training predictive or generative causal transformer models, it also supports the export of embedding models.

Configuration:

- Training: Set export_embedding_model: true in the training config.
- Inference: Set model_type: embedding in the inference config.

Technical Details: Selected activations are restricted to the configured final
`prediction_length` positions and concatenated in configuration order along the
feature dimension. Backbone selectors contribute `dim_model` values. Decoder MLP
hidden-block selectors contribute their configured hidden width and receive the
same flattened `decoding_support * dim_model` windows used during training. The
default, `embedding_layer_names: [backbone.final_norm]`, preserves the final
normalized backbone representation.

If you are interested in activations *other* than the last backbone layer, you can configure the exact layers you want to contribute to the export using `embedding_layer_names`. You can pass an ordered list, such as
- Activation sources: Set `embedding_layer_names` to an ordered list such as `[backbone.layers.1, decoder.branches.default.hidden_blocks.0]`, and the activations of these layers will be concatenated and output.

Layer names follow the network hierarchy using zero-based indices: `backbone.layers.<index>` selects a transformer block output, `backbone.final_norm` the normalized backbone output, and `decoder.branches.<branch>.hidden_blocks.<index>` an MLP decoder hidden-block output; the same scheme applies to BERT embedding models.

### BERT Model

Sequifier also supports training and inference of BERT-style masked reconstruction models.

Configuration:

- Preprocessing: Set `max_target_offset: 0` for equal-width input and target windows.
- Training: Set `training_objective: bert`, configure `bert_spec`, and set decoder `prediction_length` equal to `context_length`. Enable generative and/or embedding export according to the desired inference.
- Inference: Set `model_type: generative` to reconstruct explicitly masked input, or `model_type: embedding` to output contextual representations.

Technical Details: BERT-style models use bidirectional attention and learn by reconstructing positions sampled according to `bert_spec`. Inference does not apply random masking; inputs that should be reconstructed must be masked explicitly, for example using `mask_column` during preprocessing. Embedding inference returns one contextual representation for every valid position in the input window.


### Structured Ingestion

### Multi-Part Datasets

### Composable Configs


### Distributed Training

Sequifier supports distributed training using torch `DistributedDataParallel` and `FullyShardedDataParallel`. To make use of multi gpu support, the preprocessing step must write sharded output with `merge_output: false`. `write_format: pt` is the recommended file format; sharded `parquet` is also supported but currently considered beta for distributed training.

For the full guide on how to configure a distributed run, check the [multi-GPU training guide](./documentation/training/multi-gpu-training.md).

### System Requirements

Sequifier currently runs on MacOS and Ubuntu.

## Citation

Please cite with:

```bibtex
@software{sequifier_2025,
  author = {Luithlen, Leon},
  title = {sequifier - transformers for multivariate sequence generation and representation learning},
  year = {2025},
  publisher = {GitHub},
  version = {v2.0.0.0},
  url = {https://github.com/0xideas/sequifier}
}

```
