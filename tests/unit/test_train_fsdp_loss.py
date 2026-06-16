import datetime
import importlib
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.utils.data import DataLoader

from sequifier.io.batch import SequifierBatch
from sequifier.train import TransformerModel

_train_module = importlib.import_module("sequifier.train")
MixedPrecisionPolicy = getattr(_train_module, "MixedPrecisionPolicy")
fully_shard = getattr(_train_module, "fully_shard")
init_device_mesh = getattr(_train_module, "init_device_mesh")

pytestmark = pytest.mark.skipif(
    not dist.is_available()
    or not dist.is_nccl_available()
    or not torch.cuda.is_available()
    or torch.cuda.device_count() < 2,
    reason="FSDP2 loss tests require two CUDA devices and NCCL",
)


class _TinyRealModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(2))

    def forward(self, seq_len):
        return {"real_col": self.param[0].expand(seq_len, 1, 1)}


class _TinyTrainEpochRealModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(2))

    def forward(self, data, metadata=None, return_logits=False):
        seq_len = data["real_col"].shape[1]
        return {"real_col": self.param[0].expand(seq_len, 1, 1)}


class _TinyMultiTargetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cat_logits = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.0]))
        self.real_param = torch.nn.Parameter(torch.tensor([0.5, 0.0]))

    def forward(self, seq_len):
        return {
            "cat_col": self.cat_logits.reshape(1, 1, 3).expand(seq_len, 1, 3),
            "real_col": self.real_param[0].expand(seq_len, 1, 1),
        }


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _loss_shell(target_column_types, *, device, loss_weights=None):
    model = TransformerModel.__new__(TransformerModel)
    model.target_columns = list(target_column_types)
    model.target_column_types = dict(target_column_types)
    model.loss_weights = loss_weights
    model.device = device
    model.hparams = SimpleNamespace(
        training_spec=SimpleNamespace(
            data_parallelism="FSDP",
            distributed=True,
            training_objective="causal",
            layer_autocast=False,
            layer_type_dtypes=None,
        )
    )
    model.criterion = {}
    model.n_classes = {}

    for col, col_type in target_column_types.items():
        if col_type == "real":
            model.criterion[col] = torch.nn.MSELoss(reduction="none")
        elif col_type == "categorical":
            model.criterion[col] = torch.nn.CrossEntropyLoss(reduction="none")
            model.n_classes[col] = 3
        else:
            raise ValueError(col_type)

    return model


class _IdentityScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None


class _NoopLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


def _real_batch(seq_len, selected_count, target_value, device):
    targets = {"real_col": torch.full((1, seq_len), float(target_value), device=device)}
    mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
    mask[:, :selected_count] = True
    metadata = {"target_valid_mask": mask}
    return targets, metadata


def _real_epoch_batch(seq_len, selected_count, target_value):
    targets = {"real_col": torch.full((1, seq_len), float(target_value))}
    mask = torch.zeros(1, seq_len, dtype=torch.bool)
    mask[:, :selected_count] = True
    return SequifierBatch(
        inputs={"real_col": torch.ones(1, seq_len, dtype=torch.float32)},
        targets=targets,
        metadata={
            "attention_valid_mask": torch.ones(1, seq_len, dtype=torch.bool),
            "target_valid_mask": mask,
        },
    )


def _concat_real_batch(seq_lens, selected_counts, target_values):
    target_parts = []
    mask_parts = []
    for seq_len, selected_count, target_value in zip(
        seq_lens, selected_counts, target_values
    ):
        target_parts.append(torch.full((1, seq_len), float(target_value)))
        mask = torch.zeros(1, seq_len, dtype=torch.bool)
        mask[:, :selected_count] = True
        mask_parts.append(mask)
    return (
        {"real_col": torch.cat(target_parts, dim=1)},
        {"target_valid_mask": torch.cat(mask_parts, dim=1)},
    )


def _single_process_real_update(seq_lens, selected_counts, target_values, lr):
    model = _TinyRealModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    shell = _loss_shell({"real_col": "real"}, device="cpu")
    targets, metadata = _concat_real_batch(seq_lens, selected_counts, target_values)
    loss, _ = TransformerModel._calculate_loss(
        shell,
        model(sum(seq_lens)),
        targets,
        metadata,
    )
    loss.backward()
    optimizer.step()
    return model.param.detach().clone()


def _single_process_accumulated_real_update(microbatches, lr, accumulation_steps):
    model = _TinyRealModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    shell = _loss_shell({"real_col": "real"}, device="cpu")
    for microbatch in microbatches:
        targets, metadata = _concat_real_batch(
            microbatch["seq_lens"],
            microbatch["selected_counts"],
            microbatch["target_values"],
        )
        loss, _ = TransformerModel._calculate_loss(
            shell,
            model(sum(microbatch["seq_lens"])),
            targets,
            metadata,
        )
        (loss / accumulation_steps).backward()
    optimizer.step()
    return model.param.detach().clone()


def _analytical_accumulated_real_update(microbatches, lr, accumulation_steps):
    accumulated_grad = 0.0
    for microbatch in microbatches:
        selected_counts = torch.tensor(microbatch["selected_counts"], dtype=torch.float)
        target_values = torch.tensor(microbatch["target_values"], dtype=torch.float)
        token_count = selected_counts.sum().item()
        if token_count == 0:
            continue

        weighted_target_mean = (
            selected_counts * target_values
        ).sum().item() / token_count
        accumulated_grad += -2.0 * weighted_target_mean / accumulation_steps

    return torch.tensor([-lr * accumulated_grad, 0.0])


def _analytical_unnormalized_accumulated_real_update(microbatches, lr):
    accumulated_grad = 0.0
    for microbatch in microbatches:
        selected_counts = torch.tensor(microbatch["selected_counts"], dtype=torch.float)
        target_values = torch.tensor(microbatch["target_values"], dtype=torch.float)
        token_count = selected_counts.sum().item()
        if token_count == 0:
            continue

        weighted_target_mean = (
            selected_counts * target_values
        ).sum().item() / token_count
        accumulated_grad += -2.0 * weighted_target_mean

    return torch.tensor([-lr * accumulated_grad, 0.0])


def _analytical_combined_real_update(microbatches, lr):
    weighted_target_sum = 0.0
    token_count = 0.0
    for microbatch in microbatches:
        selected_counts = torch.tensor(microbatch["selected_counts"], dtype=torch.float)
        target_values = torch.tensor(microbatch["target_values"], dtype=torch.float)
        weighted_target_sum += (selected_counts * target_values).sum().item()
        token_count += selected_counts.sum().item()

    if token_count == 0:
        return torch.tensor([0.0, 0.0])
    return torch.tensor([lr * 2.0 * weighted_target_sum / token_count, 0.0])


def _single_process_multi_target_update(
    seq_lens,
    cat_targets,
    real_targets,
    loss_weights,
    lr,
):
    model = _TinyMultiTargetModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    shell = _loss_shell(
        {"cat_col": "categorical", "real_col": "real"},
        device="cpu",
        loss_weights=loss_weights,
    )
    targets = {
        "cat_col": torch.tensor([sum(cat_targets, [])], dtype=torch.long),
        "real_col": torch.tensor([sum(real_targets, [])], dtype=torch.float32),
    }
    metadata = {"target_valid_mask": torch.ones(1, sum(seq_lens), dtype=torch.bool)}
    loss, _ = TransformerModel._calculate_loss(
        shell,
        model(sum(seq_lens)),
        targets,
        metadata,
    )
    loss.backward()
    optimizer.step()
    return torch.cat(
        [
            model.cat_logits.detach().reshape(-1),
            model.real_param.detach().reshape(-1),
        ]
    )


def _state_vector(model, keys, rank):
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    state = get_model_state_dict(model, options=options)
    if rank != 0:
        return None

    values = []
    for key in keys:
        values.append(state[key].reshape(-1).to(dtype=torch.float32))
    return torch.cat(values)


def _mixed_precision_policy(case):
    reduce_dtype = case.get("reduce_dtype")
    if reduce_dtype is None:
        return None

    return MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=reduce_dtype,
        output_dtype=torch.bfloat16,
    )


def _fsdp_case_worker(rank, world_size, init_method, case, queue):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        mesh = init_device_mesh("cuda", (world_size,))
        mp_policy = _mixed_precision_policy(case)

        if case["kind"] in {
            "real_update",
            "accumulation_update",
            "empty_window_state",
            "train_epoch_empty_window_state",
        }:
            if case["kind"] == "train_epoch_empty_window_state":
                model = _TinyTrainEpochRealModel().to(device)
                fully_shard(model, mesh=mesh, mp_policy=mp_policy)
                shell = _loss_shell({"real_col": "real"}, device=str(device))
                shell._data_parallel_group = mesh.get_group()
                shell.input_columns = ["real_col"]
                shell.rank = rank
                shell.accumulation_steps = case["accumulation_steps"]
                shell.scheduler_step_on = "batch"
                shell.start_batch = 0
                shell.log_interval = case["accumulation_steps"]
                shell.scaler = _IdentityScaler()
                shell.logger = _NoopLogger()
                shell.save_latest_interval_minutes = None
                shell.save_batch_interval_minutes = None
                shell.save_batch_interval_minutes_val_loss = False
                shell.last_latest_save_time = 0.0
                shell.last_batch_save_time = 0.0
                shell.parameters = model.parameters
                shell.optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=case["lr"],
                    weight_decay=case["weight_decay"],
                )
                shell.scheduler = torch.optim.lr_scheduler.StepLR(
                    shell.optimizer,
                    step_size=1,
                    gamma=0.1,
                )
                before = _state_vector(model, ["param"], rank)
                train_loader = [
                    _real_epoch_batch(
                        microbatch["seq_lens"][rank],
                        microbatch["selected_counts"][rank],
                        microbatch["target_values"][rank],
                    )
                    for microbatch in case["microbatches"]
                ]

                TransformerModel._train_epoch(
                    shell,
                    DataLoader(train_loader, batch_size=None),
                    DataLoader([], batch_size=None),
                    epoch=1,
                    ddp_model=model,
                )

                after = _state_vector(model, ["param"], rank)
                if rank == 0:
                    assert before is not None
                    assert after is not None
                    queue.put(
                        [
                            float((after - before).abs().max().item()),
                            float(shell.scheduler.get_last_lr()[0]),
                            float(shell.scheduler.last_epoch),
                            float(len(shell.optimizer.state)),
                        ]
                    )
                return

            model = _TinyRealModel().to(device)
            fully_shard(model, mesh=mesh, mp_policy=mp_policy)
            shell = _loss_shell({"real_col": "real"}, device=str(device))
            shell._data_parallel_group = mesh.get_group()

            if case["kind"] == "empty_window_state":
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=case["lr"],
                    weight_decay=case["weight_decay"],
                )
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=1,
                    gamma=0.1,
                )
                before = _state_vector(model, ["param"], rank)
                accumulated_global_token_count = torch.zeros(
                    (), dtype=torch.int64, device=device
                )

                for batch_idx, microbatch in enumerate(case["microbatches"]):
                    seq_len = microbatch["seq_lens"][rank]
                    targets, metadata = _real_batch(
                        seq_len,
                        microbatch["selected_counts"][rank],
                        microbatch["target_values"][rank],
                        device,
                    )
                    loss, _, _, _, global_count = (
                        TransformerModel._calculate_training_loss(
                            shell,
                            model(seq_len),
                            targets,
                            metadata,
                        )
                    )
                    (loss / case["accumulation_steps"]).backward()
                    accumulated_global_token_count += global_count.detach()
                    optimizer_step_due = (batch_idx + 1) % case[
                        "accumulation_steps"
                    ] == 0 or (batch_idx + 1) == len(case["microbatches"])
                    optimizer_step_performed = False
                    if (
                        optimizer_step_due
                        and accumulated_global_token_count.detach().cpu().item() > 0
                    ):
                        optimizer.step()
                        optimizer.zero_grad()
                        optimizer_step_performed = True

                    if optimizer_step_due:
                        if not optimizer_step_performed:
                            optimizer.zero_grad()
                        accumulated_global_token_count.zero_()

                    if optimizer_step_performed:
                        scheduler.step()

                after = _state_vector(model, ["param"], rank)
                if rank == 0:
                    assert before is not None
                    assert after is not None
                    queue.put(
                        [
                            float((after - before).abs().max().item()),
                            float(scheduler.get_last_lr()[0]),
                            float(scheduler.last_epoch),
                            float(len(optimizer.state)),
                        ]
                    )
                return

            optimizer = torch.optim.SGD(model.parameters(), lr=case["lr"])
            if case["kind"] == "accumulation_update":
                for microbatch in case["microbatches"]:
                    seq_len = microbatch["seq_lens"][rank]
                    targets, metadata = _real_batch(
                        seq_len,
                        microbatch["selected_counts"][rank],
                        microbatch["target_values"][rank],
                        device,
                    )
                    loss, _ = TransformerModel._calculate_loss(
                        shell,
                        model(seq_len),
                        targets,
                        metadata,
                    )
                    (loss / case["accumulation_steps"]).backward()
            else:
                seq_len = case["seq_lens"][rank]
                targets, metadata = _real_batch(
                    seq_len,
                    case["selected_counts"][rank],
                    case["target_values"][rank],
                    device,
                )
                loss, _ = TransformerModel._calculate_loss(
                    shell,
                    model(seq_len),
                    targets,
                    metadata,
                )
                loss.backward()

            optimizer.step()
            state = _state_vector(model, ["param"], rank)
            if rank == 0:
                assert state is not None
                queue.put(state.tolist())
            return

        if case["kind"] == "multi_target_update":
            model = _TinyMultiTargetModel().to(device)
            fully_shard(model, mesh=mesh, mp_policy=mp_policy)
            shell = _loss_shell(
                {"cat_col": "categorical", "real_col": "real"},
                device=str(device),
                loss_weights=case["loss_weights"],
            )
            shell._data_parallel_group = mesh.get_group()
            optimizer = torch.optim.SGD(model.parameters(), lr=case["lr"])
            seq_len = case["seq_lens"][rank]
            targets = {
                "cat_col": torch.tensor(
                    [case["cat_targets"][rank]], dtype=torch.long, device=device
                ),
                "real_col": torch.tensor(
                    [case["real_targets"][rank]], dtype=torch.float32, device=device
                ),
            }
            metadata = {
                "target_valid_mask": torch.ones(
                    1, seq_len, dtype=torch.bool, device=device
                )
            }
            loss, _ = TransformerModel._calculate_loss(
                shell,
                model(seq_len),
                targets,
                metadata,
            )
            loss.backward()
            optimizer.step()
            state = _state_vector(model, ["cat_logits", "real_param"], rank)
            if rank == 0:
                assert state is not None
                queue.put(state.tolist())
            return

        raise ValueError(case["kind"])
    finally:
        dist.destroy_process_group()
        torch.cuda.empty_cache()


def _run_fsdp_case(case):
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    init_method = f"tcp://127.0.0.1:{_free_port()}"
    mp.spawn(
        _fsdp_case_worker,
        args=(2, init_method, case, queue),
        nprocs=2,
        join=True,
    )
    return torch.tensor(queue.get(), dtype=torch.float32)


def test_fsdp_unequal_token_counts_match_single_process_reference():
    case = {
        "kind": "real_update",
        "seq_lens": [100, 10],
        "selected_counts": [100, 10],
        "target_values": [2.0, 8.0],
        "lr": 0.1,
    }

    fsdp_param = _run_fsdp_case(case)
    reference_param = _single_process_real_update(
        case["seq_lens"], case["selected_counts"], case["target_values"], case["lr"]
    )

    assert torch.allclose(fsdp_param, reference_param, atol=1e-5)


@pytest.mark.parametrize("reduce_dtype", [torch.bfloat16, torch.float32])
def test_fsdp_mixed_precision_unequal_token_counts_match_reference(reduce_dtype):
    case = {
        "kind": "real_update",
        "seq_lens": [100, 10],
        "selected_counts": [100, 10],
        "target_values": [2.0, 8.0],
        "lr": 0.1,
        "reduce_dtype": reduce_dtype,
    }

    fsdp_param = _run_fsdp_case(case)
    reference_param = _single_process_real_update(
        case["seq_lens"], case["selected_counts"], case["target_values"], case["lr"]
    )

    assert torch.allclose(fsdp_param, reference_param, atol=1e-2, rtol=1e-2)


def test_fsdp_zero_token_rank_does_not_attenuate_populated_rank():
    case = {
        "kind": "real_update",
        "seq_lens": [5, 5],
        "selected_counts": [5, 0],
        "target_values": [3.0, 100.0],
        "lr": 0.1,
    }

    fsdp_param = _run_fsdp_case(case)
    reference_param = _single_process_real_update(
        case["seq_lens"], case["selected_counts"], case["target_values"], case["lr"]
    )

    assert torch.allclose(fsdp_param, reference_param, atol=1e-5)


@pytest.mark.parametrize("reduce_dtype", [torch.bfloat16, torch.float32])
def test_fsdp_mixed_precision_zero_token_rank_is_not_attenuated(reduce_dtype):
    case = {
        "kind": "real_update",
        "seq_lens": [5, 5],
        "selected_counts": [5, 0],
        "target_values": [3.0, 100.0],
        "lr": 0.1,
        "reduce_dtype": reduce_dtype,
    }

    fsdp_param = _run_fsdp_case(case)
    reference_param = _single_process_real_update(
        case["seq_lens"], case["selected_counts"], case["target_values"], case["lr"]
    )

    assert torch.allclose(fsdp_param, reference_param, atol=1e-2, rtol=1e-2)


def test_fsdp_multi_target_weighting_matches_single_process_reference():
    case = {
        "kind": "multi_target_update",
        "seq_lens": [3, 1],
        "cat_targets": [[0, 1, 2], [1]],
        "real_targets": [[1.0, 2.0, 3.0], [4.0]],
        "loss_weights": {"cat_col": 0.25, "real_col": 2.0},
        "lr": 0.1,
    }

    fsdp_param = _run_fsdp_case(case)
    reference_param = _single_process_multi_target_update(
        case["seq_lens"],
        case["cat_targets"],
        case["real_targets"],
        case["loss_weights"],
        case["lr"],
    )

    assert torch.allclose(fsdp_param, reference_param, atol=1e-5)


def test_fsdp_world_size_two_optimizer_update_matches_single_process_reference():
    case = {
        "kind": "real_update",
        "seq_lens": [6, 4],
        "selected_counts": [6, 4],
        "target_values": [1.0, 3.0],
        "lr": 0.1,
    }

    fsdp_param = _run_fsdp_case(case)
    reference_param = _single_process_real_update(
        case["seq_lens"], case["selected_counts"], case["target_values"], case["lr"]
    )

    assert torch.allclose(fsdp_param, reference_param, atol=1e-5)


def test_fsdp_gradient_accumulation_averages_per_microbatch_weighting():
    microbatches = [
        {
            "seq_lens": [1, 1],
            "selected_counts": [1, 1],
            "target_values": [1.0, 3.0],
        },
        {
            "seq_lens": [5, 5],
            "selected_counts": [5, 5],
            "target_values": [10.0, 2.0],
        },
    ]
    case = {
        "kind": "accumulation_update",
        "microbatches": microbatches,
        "accumulation_steps": 2,
        "lr": 0.05,
    }

    fsdp_param = _run_fsdp_case(case)
    reference_param = _analytical_accumulated_real_update(
        microbatches, case["lr"], case["accumulation_steps"]
    )
    unnormalized_param = _analytical_unnormalized_accumulated_real_update(
        microbatches, case["lr"]
    )
    combined_batch_param = _analytical_combined_real_update(microbatches, case["lr"])

    assert torch.allclose(fsdp_param, reference_param, atol=1e-5)
    assert not torch.allclose(fsdp_param, unnormalized_param, atol=1e-3)
    assert not torch.allclose(fsdp_param, combined_batch_param, atol=1e-3)


def test_fsdp_globally_empty_accumulation_window_leaves_state_unchanged():
    case = {
        "kind": "empty_window_state",
        "microbatches": [
            {
                "seq_lens": [3, 4],
                "selected_counts": [0, 0],
                "target_values": [1.0, 3.0],
            },
            {
                "seq_lens": [2, 5],
                "selected_counts": [0, 0],
                "target_values": [2.0, 4.0],
            },
        ],
        "accumulation_steps": 2,
        "lr": 0.01,
        "weight_decay": 0.1,
    }

    param_delta, lr, scheduler_epoch, optimizer_state_len = _run_fsdp_case(case)

    assert param_delta == 0
    assert lr == pytest.approx(case["lr"])
    assert scheduler_epoch == 0
    assert optimizer_state_len == 0


def test_fsdp_train_epoch_globally_empty_accumulation_window_leaves_state_unchanged():
    case = {
        "kind": "train_epoch_empty_window_state",
        "microbatches": [
            {
                "seq_lens": [3, 4],
                "selected_counts": [0, 0],
                "target_values": [1.0, 3.0],
            },
            {
                "seq_lens": [2, 5],
                "selected_counts": [0, 0],
                "target_values": [2.0, 4.0],
            },
        ],
        "accumulation_steps": 2,
        "lr": 0.01,
        "weight_decay": 0.1,
    }

    param_delta, lr, scheduler_epoch, optimizer_state_len = _run_fsdp_case(case)

    assert param_delta == 0
    assert lr == pytest.approx(case["lr"])
    assert scheduler_epoch == 0
    assert optimizer_state_len == 0
