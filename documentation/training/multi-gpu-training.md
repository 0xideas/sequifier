# Distributed and Multi-Node Training in Sequifier

Sequifier natively supports multi-GPU and multi-node training using PyTorch's `DistributedDataParallel` (DDP) and `FullyShardedDataParallel` (FSDP).

## Policy and mixed plans

Set `run.execution.distribution` to configure the whole plan. The existing
`run.execution.supervised.distribution` setting also applies to every phase
of a mixed plan. When both settings are present they must agree.

```yaml
run:
  execution:
    device: cuda
    distribution:
      strategy: ddp
      world_size: 4
      backend: nccl
```

For FSDP2, set `strategy: fsdp` and supply `cpu_offload: true` or
`cpu_offload: false`. Supervised phases retain their existing compilation
settings. NCCL jobs use a Gloo group on the same ranks for CPU rollout data and
recovery metadata. All hosts need shared access to input and artifact paths.

Policy batch sizes, environment-step budgets and group counts apply to the **whole job**.
Each rank independently enforces `collection.max_buffer_bytes` on its local rollout.
Four active ranks can therefore retain up to four times that configured rollout estimate. Imitation
divides a seeded decision order across ranks. DPO assigns both alternatives of a
pair to one rank. IQL partitions each draw by position, preserving repeats.
PPO divides each horizon across persistent per-rank simulator streams and
computes GAE within each stream. GRPO assigns complete same-start groups,
including their simulator workers, to the owning rank. Rank zero evaluates the
prescribed online seeds. Sharded peers participate in the required model forwards.

Resume with the same world size. The native policy checkpoint includes shared
model and optimizer state and each rank's RNG and simulator state. Rank zero
publishes the selected actor, sequence output and reports. The phase report
includes per-rank timing for the current session, working-storage estimates and
the process peak CUDA allocation. Scoring time measures host time inside active
model forwards and overlaps collection and updates.

The ten policy algorithm and strategy combinations await update-parity, recovery,
and hardware qualification.

## 1. Preprocessing for Distributed Supervised Training

For distributed supervised training, your prepared data must be sharded into multiple files so that different GPUs can read different chunks simultaneously without memory bottlenecks.

In your `preprocess.yaml`, you **must** write sharded output:

```yaml
merge_output: false
```

For production multi-GPU training, use PyTorch tensor shards:

```yaml
write_format: pt
```

*Note: Distributed training is not supported if your data is kept as a single `csv` or `parquet` file. You must use `merge_output: false` to generate a folder of sharded files.*

> **Beta Notice for Parquet in Distributed Training:**
> While `write_format: parquet` is supported for distributed training, it is currently considered **Beta**. Because Parquet chunk reading relies on Polars' multi-threading, using it alongside PyTorch's multiprocess `DataLoader` in heavy multi-GPU environments can lead to CPU thread contention, high RAM usage, or NCCL timeouts.
> **Recommendation:** For production multi-GPU runs, use `write_format: pt`. It relies on native PyTorch serialization and is significantly more stable under heavy hardware loads.


## 2. Configuration: `train.yaml`

Once your data is preprocessed into `.pt` shards, or beta `.parquet` shards, you need to tell the Sequifier training engine to expect a distributed environment.

In your `train.yaml`, configure `run.execution.supervised.distribution` and each prepared part’s `read_format`:

```yaml
run:
  execution:
    device: cuda
    supervised:
      distribution:
        strategy: fsdp
        cpu_offload: false
        world_size: 32
        backend: nccl
      compilation:
        torch_compile: inner
datasets:
  events:
    sources:
      train:
        read_format: pt
        metadata_config_path: metadata_configs/events.json
        type: prepared
```

When shards do not divide evenly across ranks, Sequifier automatically pads shorter ranks with repeated samples for step alignment. Those repeats are masked out of loss calculation, so each real sample contributes once.

## 3. Launching the Training Job

How you launch the training depends on whether you are using a single machine with multiple GPUs, or multiple machines (nodes) connected over a network.

### Scenario A: Single-Node, Multi-GPU

If you are running on a single machine that has multiple GPUs (e.g., an AWS EC2 instance with 4x A100s), Sequifier can handle process generation internally using `torch.multiprocessing.spawn`.

You simply run the standard command:

```bash
sequifier train --config-path configs/train.yaml

```

Sequifier will read the `world_size` config parameter and automatically spawn that exact number of worker processes.

### Scenario B: Multi-Node, Multi-GPU (HPC / Slurm)

Sequifier cannot automatically spawn Python processes across physical network boundaries. For multi-node training, you must use an external cluster manager (like Slurm) combined with PyTorch's `torchrun` utility.

When `sequifier` detects `torchrun` environment variables (like `RANK` and `WORLD_SIZE`), it bypasses its internal spawner and attaches to the distributed network established by the cluster. In that mode, the environment `WORLD_SIZE` is used.

Here is a standard `sbatch` script template for launching Sequifier across multiple nodes:

```bash
#!/bin/bash
#[SBATCH COMMANDS]

MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

srun torchrun \
    [-- torchrun args]...
    $(which sequifier) train --config-path=configs/train.yaml
```

### Important Considerations for Multi-Node

* **Batch Size:** For supervised phases, `batch_size` is the **per-process** batch size. If `batch_size` is 100 and `world_size` is 32, each synchronized backward pass covers 3,200 samples. With gradient accumulation, the samples per optimizer update are `batch_size * world_size * accumulation_batches` (apart from a final partial accumulation window).
* **Learning Rate:** You may need to scale your `learning_rate` up if you drastically increase your global batch size via distributed training.
* **Data Access:** All nodes must have access to the same shared filesystem (e.g., NFS, GPFS) where the `project_root` and the sharded preprocessing output are stored.
