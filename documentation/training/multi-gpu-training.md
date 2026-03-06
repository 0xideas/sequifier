# Distributed and Multi-Node Training in Sequifier

Sequifier natively supports multi-GPU and multi-node training using PyTorch's `DistributedDataParallel` (DDP).

## 1. Prerequisites: Preprocessing for DDP

To use distributed training, your data must be sharded into multiple files so that different GPUs can read different chunks simultaneously without memory bottlenecks.

In your `preprocess.yaml`, you **must** set the following:

```yaml
merge_output: false
```

typically, you'd also want to set

```yaml
write_format: pt
```

*Note: Distributed training is not supported if your data is kept as a single `csv` or `parquet` file.*

## 2. Configuration: `train.yaml`

Once your data is preprocessed into `.pt` shards, you need to tell the Sequifier training engine to expect a distributed environment.

In your `train.yaml`, update the `training_spec` block:

```yaml
training_spec:
  distributed: true
  world_size: 32       # The TOTAL number of GPUs across all nodes (e.g., 8 nodes * 4 GPUs = 32)
  backend: nccl        # 'nccl' is the standard and most efficient backend for NVIDIA GPUs
  sampling_strategy: 'oversampling' # if the number of files isn't perfectly divisible by the number of GPUs, you need to choose either 'oversampling' or 'undersampling'. If it is perfectly divisible, you can set it to 'exact'

```

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

When `sequifier` detects `torchrun` environment variables (like `RANK` and `WORLD_SIZE`), it bypasses its internal spawner and attaches to the distributed network established by the cluster.

Here is a standard `sbatch` script template for launching Sequifier across multiple nodes:

```bash
#!/bin/bash
#SBATCH --job-name=sequifier_multinode
#SBATCH --nodes=8                  # Number of nodes
#SBATCH --gres=gpu:4               # GPUs per node
#SBATCH --ntasks-per-node=1        # One task per node (torchrun handles the rest)
#SBATCH --cpus-per-task=80         # CPU cores per node

# ... python env setup ...

MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

srun torchrun \
    --nnodes=8 \
    --nproc_per_node=4 \
    --rdzv_id=sequifier_job \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_NODE:29400 \
    $(which sequifier) train --config-path=configs/train.yaml
```

### Important Considerations for Multi-Node

* **Batch Size:** The `batch_size` in your `train.yaml` is the **per-GPU** batch size. If your `batch_size` is 100, and your `world_size` is 32, your effective global batch size is 3,200.
* **Learning Rate:** You may need to scale your `learning_rate` up if you drastically increase your global batch size via distributed training.
* **Data Access:** All nodes must have access to the same shared filesystem (e.g., NFS, GPFS) where the `project_root` and the `.pt` data shards are stored.
