<p align="center">
  <img src="./design/sequifier.png" alt="Sequifier" width="720">
</p>

# What is sequifier?

Sequifier is the terraform for foundation modelling: It provides a declarative interface to data preparation, model architecture, training and optimization hyperparameters, and inference.

It provides shared infrastructure and guarantees to a wide range of foundation modelling applications that are individually too small to sustain their own ecosystem, unlike language modelling, audio and image or video generation.

Any implementation of a foundation model will operate under several constraints: computate constraints such as RAM, GPU memory, disk speed and device connectivity, scaling dimenstions such as process parallelisation and multi-gpu and multi-node training, and desirable properties such as efficiency, idempotency, early failure on misconfiguration, resumability and reproducibility. Sequifier aims to push out the pareto frontier on these dimensions and provide the best starting point for any de novo foundation modelling initiative.

# How does it work?

Sequifier has three core commands, `preprocess`, `train`, and `infer`. Each is configurable via a config file, that specifies the data layout and structure and the other attributes that you want from your model.

They enable you to go from multivariate sequence data, images and/or video to a model that ingests such data, and emits either multivariate sequence data (but not images/video) or embeddings.

If input and target variables are the same, or a missing input variables can be computed from target variables, it supports full autoregressive inference.

Through configuration, most modern architectural variants such as RoPE, GQA, SwiGLU, and RMSNorm are supported. The ambition is to follow the frontier when industrial-grade implementations become available.

You can train the transformer using supervised losses or reinforcement learning algorithms such as imitation learning, implicit Q learning (IQL), direct policy optimization (DPO), proximal policy optimization (PPO) or group relative policy optimization (GRPO).

## What can I use it for?

Multivariate transformers have *many* applications.

These independent references illustrate some of them; inclusion does not imply that the cited work used or is affiliated with sequifier.

**Finance**

- Transaction Foundation Models ([Overview](https://sistemalabs.com/blog/shaping-the-problem), [EWE-1](https://sistemalabs.com/blog/introducing-ewe-1))
- Order Book Transformers ([TransLOB](https://arxiv.org/abs/2003.00130))
- Volatility Forecasting ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0952197624003816))
- Macroeconomic Models ([BISTRO](https://www.bis.org/publications/bistro-general-purpose-oracle-macroeconomic-time-series))

**Health/Bio**

- Health Records ([BEHRT](https://www.nature.com/articles/s41598-020-62922-y), [Med-BERT](https://www.nature.com/articles/s41746-021-00455-y))
- Treatment Outcome Prediction ([G-Transformer](https://arxiv.org/abs/2406.05504))
- Health Trajectory Prediction ([ETHOS](https://www.nature.com/articles/s41746-024-01235-0))
- Glucose Forecasting ([GluForecast](https://arxiv.org/html/2606.18640v1))
- ECG rhythm/arrhythmia monitoring ([Heart Language Model](https://www.nature.com/articles/s41598-024-84270-x))

**Cybersecurity**

- Network Intrusion Detection ([paper](https://arxiv.org/pdf/2309.01070))
- Encrypted Traffic Analysis ([Criss-Cross Traffic Transformer](https://ieeexplore.ieee.org/document/11396346))
- Insider Threat Detection ([paper](https://www.nature.com/articles/s41598-025-12063-x))
- In-Vehicle Intrusion Detection ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705124007251))
- IoT Anomaly Detection ([paper](https://www.sciopen.com/article/10.32604/cmc.2024.053765))
- System Logs Anomaly Detection ([DeepEAD](https://ieeexplore.ieee.org/document/10279563))

**Industrial & IoT**

- Remaining Useful Life Estimation ([paper](https://www.sciencedirect.com/science/article/abs/pii/S095219762400633X))
- Fault Diagnosis ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0957582026001485))
- Anomaly Detection for Industrial Control Systems ([paper](https://onlinelibrary.wiley.com/doi/10.1155/2024/5459452))
- Soft Sensing ([Debutanizer](https://www.sciencedirect.com/science/article/abs/pii/S1876107024003249))
- Battery State-of-Health Estimation ([DS-transformer](https://www.nature.com/articles/s41598-026-52202-6))
- Production Line Modelling ([paper](https://www.sciencedirect.com/science/article/pii/S2212827126007791?))

**Agriculture & Environment**

- Soil Moisture Forecasting ([paper](https://link.springer.com/chapter/10.1007/978-3-032-19763-4_18))
- Crop Water Demand ([paper](https://www.mdpi.com/2073-4395/16/16/1600))
- Rainfall–runoff modelling ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0022169422003560))
- Drought Forecasting ([paper](https://www.sciencedirect.com/science/article/abs/pii/S1364815225000787))
- Land Surface Dynamics ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0022169425002446))

**Neuroscience**

- Neural Population Dynamics ([Neural Data Transformer](https://pmc.ncbi.nlm.nih.gov/articles/PMC10541112/))
- Neural + Motor Modelling ([Intracortical Motor Decoder](https://www.biorxiv.org/content/10.1101/2025.02.02.634313v1))
- fMRI State Prediction ([paper](https://arxiv.org/abs/2412.19814))
- Seizure Detection ([BIOT](https://arxiv.org/abs/2305.10351))
- Magnetoencephalography Data ([MEG-GPT](https://arxiv.org/abs/2510.18080))

**Animal Communication**

- Sperm Whales ([whale-gpt](https://github.com/0xideas/whale-gpt), [WhAM](https://github.com/Project-CETI/wham/))
- Bengalese Finches ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8746767/))
- Zebra Finches ([ZF-Aim](https://www.biorxiv.org/content/10.64898/2026.02.12.705387v1))

Sequifier aims to standardise model implementation across these fields, to make them comparable, transfer learnings between domains, and converge on optimal solutions faster.

## Empowering researchers

For the individual researcher, sequifier cuts the development time of a model significantly: typically, some preprocessing and the subsequent model evaluation are specific to the modelling problem, but all the steps in between are taken care of.

This enables:

- rapid prototyping on a configurable architecture
- trusted implementation (you can't create bugs inadvertently)
- scaling preprocessing across cores and training across GPUs and nodes
- hyperparameter optimization using Optuna

## Production-grade model lifecycles

Not every foundation model has a research team behind it: some will be trained and fine tuned automatically for individual users or user segments, or different deployments, machines, or domains of various kinds, with the specifics depending on the application. This creates considerable complexity in any real life deployment of foundation models.

Defining and training such models using sequifier reduces that complexity significantly: it makes the management of model versions and iterations much easier, facilitates provenance and change tracking, and makes reproduction of any trained checkpoint much simpler. Instead of several complex codebases glued together with hacky scripts, you get clean configs and standardized artefacts, that can be rerun in isolation.

## The Seven Commands

There are seven standalone commands within sequifier: `make`, `preprocess`, `train`, `infer`, `hyperparameter-search`, `visualize-training`, and `docs`.

| Command | Purpose |
| --- | --- |
| `make` | Create a new sequifier project with config templates. |
| `preprocess` | Convert input data into fixed-length subsequences. |
| `train` | Train supervised models or ordered imitation, IQL, PPO, DPO and GRPO policy phases. |
| `infer` | Generate predictions, probabilities, embeddings, or policy actions and rollouts. |
| `hyperparameter-search` | Use Optuna to find optimal configurations across multiple training runs. |
| `visualize-training` | Generate interactive HTML plots from structured training metrics. |
| `docs` | Print the consolidated documentation, or the guide for `train`, `preprocess`, or `infer`. |

Run `sequifier docs` for all documentation, or `sequifier docs train`, `sequifier docs preprocess`, or `sequifier docs infer` for one command guide.

The processing and visualization commands have dedicated documentation pages:

- [canonical configuration names](./documentation/configs/naming.md)
- [preprocess documentation](./documentation/configs/preprocess.md)
- [train documentation](./documentation/configs/train.md)
- [policy training](./documentation/policy-learning.md), including BC, IQL, DPO, GRPO and [online categorical PPO](./documentation/policy-learning.md#online-categorical-ppo)
- [infer documentation](./documentation/configs/infer.md)
- [hyperparameter-search documentation](./documentation/configs/hyperparameter-search.md)
- [visualize-training documentation](./documentation/commands/visualize-training.md)

## Other Materials

To get the full documentation, visit [sequifier.com](https://sequifier.com)

For changes in each published version, see the [release notes](https://github.com/0xideas/sequifier/releases).

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
├── state/ (created when hyperparameter search runs)
└── scripts/
```

The `sequifier` commands should typically be run in the project root.

Within YOUR_PROJECT_NAME, you can also add other folders for additional steps, such as `notebooks` or `scripts` for pre- or postprocessing, and `analysis`, `visualizations` or `evals` for files you generate in other, manual steps.

## Data Formats

The core modelling primitive is the `itemPosition`: it represents the unit of the sequence that the core model learns to represent. At each `itemPosition` value, each input variable is projected into a latent space, and these are concatenated and ingested into the core model. An input variable can be real-valued, discrete/categorical, a uni- or multivariate sequence itself (this type is ingested using `structured input`), or an image (or equivalently, a frame from a video). Real-valued variables are passed to a fixed position in the core model, or transformed through a single neural layer, discrete variables are projected to an embedding space, sequences are fed through a small transformer model, and images are ingested through a ResNet model, which has either frozen weights from a separate training run, or is newly initialized. This is a high-level overview, with many of the specifics configurable.

The second modelling primitive is the `static variable`. It is a variable which is either real-valued or categorical, and is embedded separately from the variables that vary with `itemPosition` - they stay constant across a whole individual sequence, and vary only between sequences. Each static variable is embedded separately and these embeddings are concatenated. This joint embedding for all static variables is prepended to the sequence of `itemPosition`-variant embeddings, and treated as the "first" token for any window based on that sequence.

## Target Variables

Sequifier currently supports categorical and real-valued target variables, but not images (sequences will be supported, but aren't yet). This is because training generative models for video outputs requires modelling approaches that are not supported by sequifier, and the recommendation is to use other frameworks that are specialised in the development of such models.

### Complete Example of Training and Inferring a Transformer Model

Once you have your data in the input format described above, you can train a transformer model in a couple of steps:

1. **Install sequifier.** Create and activate an environment with Python >=3.10, then run:

   ```console
   pip install sequifier
   ```

2. **Create a project.** Generate a project folder with config templates in its `configs` subfolder:

   ```console
   sequifier make YOUR_PROJECT_NAME
   ```

3. **Add your data.** Change into the `YOUR_PROJECT_NAME` folder and place a CSV with `sequenceId`, `itemPosition` and a numeric `value` column at `data/events.csv`. For another schema, adjust the preprocessing columns and the model interface.

4. **Preprocess the data.** Run:

   ```console
   sequifier preprocess
   ```

5. **Reference the generated metadata.** Preprocessing outputs metadata at `configs/metadata_configs/[INPUT BASENAME].json`. Reference that file from `datasets.<dataset>.sources.<source>.metadata_config_path` in `train.yaml`. Inference uses `input.metadata_config_path` or the selected artifact metadata.

6. **Configure and train the model.** Adapt `train.yaml` with your transformer hyperparameters, then run:

   ```console
   sequifier train
   ```

7. **Configure inference.** The scaffold selects the final model at `models/forecast.pt` and the prepared test data. To use different inputs or another export, select a different `input.split` or specify `input.path`, and set `artifact.paths` in `infer.yaml`. See [inference configuration](./documentation/configs/infer.md).

   **GPU ONNX inference:** `run.execution.device: cuda` requires `onnxruntime-gpu` built for
   your CUDA and cuDNN versions. The CPU-only `onnxruntime` package does not
   provide `CUDAExecutionProvider`.

8. **Run inference.** Run:

   ```console
   sequifier infer
   ```

9. **Find your predictions.** They are written to `[PROJECT ROOT]/outputs/predictions/[EXPORTED_MODEL_BASENAME]/part-000.[FORMAT]`, for example `outputs/predictions/your-model-best-3/part-000.csv`.

## Key Features

### Optimisation Objectives

#### Causal Modelling

Predict the values the target variables take at the next `itemPosition`. The generalisation of "next-token prediction".

#### Final-value Causal Modelling

In final-value causal models, the final value of each target variable within the subsequence is projected back in time as target. The idea is that the sequence of events leading up to the final value is a continuous accrual of evidence for an outcome, with the final value being the resolution. For example, the sequence of clicks through an online shop are in search of a product, and the product that is actually purchased at the end is the resolution.

#### Next-occurrence Causal Modelling

Next-occurrence causal modelling is a generalisation of final-value causal modelling: instead of taking the last value of each target variable as target, it takes the next value at a position where another categorical variable matches a criterion value as target. To illustrate this, final-value causal modelling is equivalent to next-occurrence causal modelling where the criterion variable is 'is_last', which is '0' up to the last position, where it is '1', and the criterion value is '1'. The values at the last position are projected 'back' across the subsequence, only now, we also have the option to use a different criterion variable, set it to '1' at multiple locations, and train the model to predict 'next relevant event', rather than just 'last event'.

#### BERT Model

Masks for BERT-style training are sampled from a distribution that is parametrized in the training config, and applies with sampled span lengths to several adjacent `itemPositions`, up to a configured "mask-budget". Masked values are replaced either with a mask token, a random sample, or left unchanged (in which case it operates as a mask on losses only). Per-target variable masking is currently not supported.

### Training Algorithms

Sequifier supports supervised learning, where conventional supervised losses are applied to individual real-valued and categorical target variables separately and then aggregated, and several reinforcement learning algoirthms. These are behaviour cloning, implicit Q learning, direct policy optimisation, proximal policy optimisation and group relative policy optimisation.

### Outputs

#### Predictions

For causal models, predictions consist in `prediction_length` output values per target variable and input window.

#### Autoregressive Generation

For autoregressive inference, either all the input variables must also be target variables, or those inputs that aren't target variables must be derivable from target variables, through a simulator or something like it. The derivation logic must be integrated into inference through the forecast rollout adapter. The `prediction_length` must be set to 1.

#### Probabilities

Probabilities can be output for categorical target variables and are returned as one row per target variable per `itemPosition`, with the categories to which each probability applies in the columns.

#### Embeddings

Embeddings can be exported per `itemPosition`, and are configurable: by default the activations of the last core model layer are exported, but other layers can be used when set in  `embedding_layer_names`. If multiple layers are specified, the activations from the named layers are concatenated and output as on large embedding. Each row of output embeddings is an `itemPosition` for a given sequence, and the columns are the positions in the embedding space.


### Model Architecture Elements

#### Causal Decoding of Target Variables

By default, sequifier decodes the global latent state into target variables in parallel, and independently from each other. The sampled value of one target variable has no influence on the probability distribution of any other target variable. A causal decoder head changes this: the target variables are predicted one by one, and the materialised value for a target variable conditions all subsequent target variable predictions for that `itemPosition`.

#### Temporal Convolution

Inspired by order book transformer (TransLOB), sequifier includes the ability to apply convolutions across variables and across `itemPositions` (but not both at the same time). This is the only mechanism which breaks the atomicity of the `itemPosition` primitive: values of a given input variable across `itemPositions` are passed through a small convolutional neural network, and thereby the earlier values are "mixed into" the latent representation of a given `itemPosition`. For early `itemPositions`, the sequence is padded with zeros.

#### Hashing

Inspired by work in transaction foundation models, sequifier provides two hashing pathways: single-hashing, which occurs in preprocessing and isn't reversible. This is unavoidable, for if the cardinality of the variable is to be reduced, collisions will happen by definition. The alternative is multi-hashing, where each value is hashed multiple times, and each distinct input value is mapped to a unique *combination* of hash values. This uniqueness is enforced by sequifier, and enables reconstruction of the original value during inference. During training, the configured losses apply to the hash values, rather than the original domain.


### Multi-Part Datasets

Multi-part datasets let a model train on separately preprocessed data with
distinct metadata configs. Parts must share the schema, categorical mappings,
normalization policy, `max_target_offset`, per-split window widths, alignment
settings, dtypes, and file/folder storage form. Set `metadata_config_path` to
the first part's metadata when preprocessing later parts to reuse its fitted
category and normalization meanings.

### Composable Configs

Between different training runs and hyperparameter searches, a lot of configuration can get duplicated, and it becomes hard to follow what differentiates them and where they overlap. One approach to address this is to create different config fragments, and compose them into full training and hyperparameter tuning configurations using separate 'top-level' configs, and assemble the fragments by listing them as value in `additional_config_paths`. Fragments can contribute disjoint nested fields, while duplicate fields, recursive fragment inclusion, and repeated files are rejected; command-line overrides are applied after composition.

### Distributed Training

Sequifier supports distributed training using torch `DistributedDataParallel` and `FullyShardedDataParallel`. To make use of multi gpu support, the preprocessing step must write sharded output with `merge_output: false`. `write_format: pt` is the recommended file format; sharded `parquet` is also supported but currently considered beta for distributed training.

For the full guide on how to configure a distributed run, check the [multi-GPU training guide](./documentation/training/multi-gpu-training.md).

### System Requirements

Sequifier currently runs on MacOS and Linux.

## Citation

Please cite with:

```bibtex
@software{sequifier_2025,
  author = {Luithlen, Leon},
  title = {sequifier - transformers for multivariate sequence generation and representation learning},
  year = {2025},
  publisher = {GitHub},
  version = {v3.1.2.0},
  url = {https://github.com/0xideas/sequifier}
}

```
