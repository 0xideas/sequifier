import datetime
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from sequifier.io.batch import SequifierBatch
from sequifier.train import TransformerModel


class _TinyRealModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, seq_len):
        return {"real_col": self.param.expand(seq_len, 1, 1)}


class _TinyTrainEpochRealModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, data, metadata=None, return_logits=False):
        seq_len = data["real_col"].shape[1]
        return {"real_col": self.param.expand(seq_len, 1, 1)}


class _TinyMultiTargetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cat_logits = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.0]))
        self.real_param = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, seq_len):
        return {
            "cat_col": self.cat_logits.reshape(1, 1, 3).expand(seq_len, 1, 3),
            "real_col": self.real_param.expand(seq_len, 1, 1),
        }


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _loss_shell(target_column_types, *, data_parallelism=None, loss_weights=None):
    model = TransformerModel.__new__(TransformerModel)
    model.target_columns = list(target_column_types)
    model.target_column_types = dict(target_column_types)
    model.loss_weights = loss_weights
    model.device = "cpu"
    model.hparams = SimpleNamespace(
        training_spec=SimpleNamespace(
            data_parallelism=data_parallelism,
            distributed=data_parallelism is not None,
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


def _real_batch(seq_len, selected_count, target_value):
    targets = {"real_col": torch.full((1, seq_len), float(target_value))}
    mask = torch.zeros(1, seq_len, dtype=torch.bool)
    mask[:, :selected_count] = True
    metadata = {"target_valid_mask": mask}
    return targets, metadata


def _real_epoch_batch(seq_len, selected_count, target_value):
    targets, metadata = _real_batch(seq_len, selected_count, target_value)
    return SequifierBatch(
        inputs={"real_col": torch.ones(1, seq_len, dtype=torch.float32)},
        targets=targets,
        metadata={
            "attention_valid_mask": torch.ones(1, seq_len, dtype=torch.bool),
            **metadata,
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


def _single_process_real_grad(seq_lens, selected_counts, target_values):
    model = _TinyRealModel()
    shell = _loss_shell({"real_col": "real"})
    targets, metadata = _concat_real_batch(seq_lens, selected_counts, target_values)
    loss, _ = TransformerModel._calculate_loss(
        shell,
        model(sum(seq_lens)),
        targets,
        metadata,
    )
    loss.backward()
    return model.param.grad.detach().clone()


def _single_process_real_update(seq_lens, selected_counts, target_values, lr):
    model = _TinyRealModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    shell = _loss_shell({"real_col": "real"})
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


def _single_process_accumulated_real_update(microbatches, lr):
    model = _TinyRealModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    shell = _loss_shell({"real_col": "real"})
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
        loss.backward()
    optimizer.step()
    return model.param.detach().clone()


def _analytical_accumulated_real_update(microbatches, lr):
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

    return torch.tensor(-lr * accumulated_grad)


def _analytical_combined_real_update(microbatches, lr):
    weighted_target_sum = 0.0
    token_count = 0.0
    for microbatch in microbatches:
        selected_counts = torch.tensor(microbatch["selected_counts"], dtype=torch.float)
        target_values = torch.tensor(microbatch["target_values"], dtype=torch.float)
        weighted_target_sum += (selected_counts * target_values).sum().item()
        token_count += selected_counts.sum().item()

    if token_count == 0:
        return torch.tensor(0.0)
    return torch.tensor(lr * 2.0 * weighted_target_sum / token_count)


def _single_process_multi_target_grad(
    seq_lens, cat_targets, real_targets, loss_weights
):
    model = _TinyMultiTargetModel()
    shell = _loss_shell(
        {"cat_col": "categorical", "real_col": "real"},
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
    return torch.cat(
        [
            model.cat_logits.grad.detach(),
            model.real_param.grad.detach().reshape(1),
        ]
    )


def _ddp_case_worker(rank, world_size, init_method, case, queue):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        timeout=datetime.timedelta(seconds=30),
    )
    try:

        def put_result(tensor):
            detached = tensor.detach().cpu()
            if detached.numel() == 1:
                queue.put(float(detached.item()))
            else:
                queue.put(detached.tolist())

        if case["kind"] in {
            "real_grad",
            "real_update",
            "empty_grad",
            "accumulation_update",
            "empty_window_state",
            "train_epoch_empty_window_state",
        }:
            if case["kind"] == "train_epoch_empty_window_state":
                model = _TinyTrainEpochRealModel()
                ddp_model = DDP(model)
                shell = _loss_shell({"real_col": "real"}, data_parallelism="DDP")
                shell.input_columns = ["real_col"]
                shell.rank = rank
                shell.device = "cpu"
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

                train_loader = [
                    _real_epoch_batch(
                        microbatch["seq_lens"][rank],
                        microbatch["selected_counts"][rank],
                        microbatch["target_values"][rank],
                    )
                    for microbatch in case["microbatches"]
                ]
                before = model.param.detach().clone()

                TransformerModel._train_epoch(
                    shell,
                    DataLoader(train_loader, batch_size=None),
                    DataLoader([], batch_size=None),
                    epoch=1,
                    ddp_model=ddp_model,
                )

                if rank == 0:
                    queue.put(
                        [
                            float((model.param.detach() - before).abs().item()),
                            float(shell.scheduler.get_last_lr()[0]),
                            float(shell.scheduler.last_epoch),
                            float(len(shell.optimizer.state)),
                        ]
                    )
                return

            model = _TinyRealModel()
            ddp_model = DDP(model)
            shell = _loss_shell({"real_col": "real"}, data_parallelism="DDP")

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
                before = model.param.detach().clone()
                accumulated_global_token_count = torch.zeros((), dtype=torch.int64)

                for batch_idx, microbatch in enumerate(case["microbatches"]):
                    seq_len = microbatch["seq_lens"][rank]
                    targets, metadata = _real_batch(
                        seq_len,
                        microbatch["selected_counts"][rank],
                        microbatch["target_values"][rank],
                    )
                    loss, _, _, _, global_count = (
                        TransformerModel._calculate_training_loss(
                            shell,
                            ddp_model(seq_len),
                            targets,
                            metadata,
                        )
                    )
                    loss.backward()
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

                if rank == 0:
                    queue.put(
                        [
                            float((model.param.detach() - before).abs().item()),
                            float(scheduler.get_last_lr()[0]),
                            float(scheduler.last_epoch),
                            float(len(optimizer.state)),
                        ]
                    )
                return

            if case["kind"] == "accumulation_update":
                optimizer = torch.optim.SGD(model.parameters(), lr=case["lr"])
                for microbatch in case["microbatches"]:
                    seq_len = microbatch["seq_lens"][rank]
                    targets, metadata = _real_batch(
                        seq_len,
                        microbatch["selected_counts"][rank],
                        microbatch["target_values"][rank],
                    )
                    loss, _, _, _, _ = TransformerModel._calculate_training_loss(
                        shell,
                        ddp_model(seq_len),
                        targets,
                        metadata,
                    )
                    loss.backward()
                optimizer.step()
                if rank == 0:
                    put_result(model.param)
                return

            seq_len = case["seq_lens"][rank]
            targets, metadata = _real_batch(
                seq_len,
                case["selected_counts"][rank],
                case["target_values"][rank],
            )
            loss, _, _, _, _ = TransformerModel._calculate_training_loss(
                shell,
                ddp_model(seq_len),
                targets,
                metadata,
            )
            loss.backward()

            if case["kind"] == "real_update":
                optimizer = torch.optim.SGD(model.parameters(), lr=case["lr"])
                optimizer.step()
                result = model.param.detach().clone()
            else:
                result = model.param.grad.detach().clone()

            if rank == 0:
                put_result(result)
            return

        if case["kind"] == "metrics":
            shell = _loss_shell(
                {"cat_col": "categorical", "real_col": "real"},
                data_parallelism="DDP",
                loss_weights=case["loss_weights"],
            )
            sums = {
                "cat_col": torch.tensor(case["cat_sums"][rank], dtype=torch.float64),
                "real_col": torch.tensor(case["real_sums"][rank], dtype=torch.float64),
            }
            count = torch.tensor(case["counts"][rank], dtype=torch.float64)
            total, losses = TransformerModel._finalize_loss_components(
                shell,
                sums,
                count,
                ["cat_col", "real_col"],
                "training",
            )
            if rank == 0:
                queue.put(
                    [
                        float(total.detach().cpu().item()),
                        float(losses["cat_col"].detach().cpu().item()),
                        float(losses["real_col"].detach().cpu().item()),
                    ]
                )
            return

        if case["kind"] == "multi_target_grad":
            model = _TinyMultiTargetModel()
            ddp_model = DDP(model)
            shell = _loss_shell(
                {"cat_col": "categorical", "real_col": "real"},
                data_parallelism="DDP",
                loss_weights=case["loss_weights"],
            )
            seq_len = case["seq_lens"][rank]
            targets = {
                "cat_col": torch.tensor([case["cat_targets"][rank]], dtype=torch.long),
                "real_col": torch.tensor(
                    [case["real_targets"][rank]], dtype=torch.float32
                ),
            }
            metadata = {"target_valid_mask": torch.ones(1, seq_len, dtype=torch.bool)}
            loss, _, _, _, _ = TransformerModel._calculate_training_loss(
                shell,
                ddp_model(seq_len),
                targets,
                metadata,
            )
            loss.backward()
            if rank == 0:
                put_result(
                    torch.cat(
                        [
                            model.cat_logits.grad.detach(),
                            model.real_param.grad.detach().reshape(1),
                        ]
                    )
                )
            return

        raise ValueError(case["kind"])
    finally:
        dist.destroy_process_group()


def _run_ddp_case(case):
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    init_method = f"tcp://127.0.0.1:{_free_port()}"
    mp.spawn(
        _ddp_case_worker,
        args=(2, init_method, case, queue),
        nprocs=2,
        join=True,
    )
    return torch.tensor(queue.get())


pytestmark = pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed is not available",
)


def test_ddp_unequal_token_counts_match_single_process_reference():
    case = {
        "kind": "real_grad",
        "seq_lens": [100, 10],
        "selected_counts": [100, 10],
        "target_values": [2.0, 8.0],
    }

    ddp_grad = _run_ddp_case(case)
    reference_grad = _single_process_real_grad(
        case["seq_lens"], case["selected_counts"], case["target_values"]
    )
    old_equal_rank_mean = torch.tensor((-2.0 * 2.0 + -2.0 * 8.0) / 2.0)

    assert torch.allclose(ddp_grad, reference_grad, atol=1e-6)
    assert not torch.allclose(ddp_grad, old_equal_rank_mean, atol=1e-3)


def test_ddp_zero_token_rank_does_not_attenuate_populated_rank():
    case = {
        "kind": "real_grad",
        "seq_lens": [5, 5],
        "selected_counts": [5, 0],
        "target_values": [3.0, 100.0],
    }

    ddp_grad = _run_ddp_case(case)
    reference_grad = _single_process_real_grad(
        case["seq_lens"], case["selected_counts"], case["target_values"]
    )

    assert torch.allclose(ddp_grad, reference_grad, atol=1e-6)
    assert torch.allclose(ddp_grad, torch.tensor(-6.0), atol=1e-6)


def test_ddp_equal_token_counts_retain_equal_rank_mean_behavior():
    case = {
        "kind": "real_grad",
        "seq_lens": [4, 4],
        "selected_counts": [4, 4],
        "target_values": [1.0, 5.0],
    }

    ddp_grad = _run_ddp_case(case)
    equal_rank_mean = torch.tensor((-2.0 * 1.0 + -2.0 * 5.0) / 2.0)

    assert torch.allclose(ddp_grad, equal_rank_mean, atol=1e-6)


def test_ddp_multi_target_weighting_matches_single_process_reference():
    case = {
        "kind": "multi_target_grad",
        "seq_lens": [3, 1],
        "cat_targets": [[0, 1, 2], [1]],
        "real_targets": [[1.0, 2.0, 3.0], [4.0]],
        "loss_weights": {"cat_col": 0.25, "real_col": 2.0},
    }

    ddp_grad = _run_ddp_case(case)
    reference_grad = _single_process_multi_target_grad(
        case["seq_lens"],
        case["cat_targets"],
        case["real_targets"],
        case["loss_weights"],
    )

    assert torch.allclose(ddp_grad, reference_grad, atol=1e-6)


def test_ddp_world_size_two_optimizer_update_matches_single_process_reference():
    case = {
        "kind": "real_update",
        "seq_lens": [6, 4],
        "selected_counts": [6, 4],
        "target_values": [1.0, 3.0],
        "lr": 0.1,
    }

    ddp_param = _run_ddp_case(case)
    reference_param = _single_process_real_update(
        case["seq_lens"], case["selected_counts"], case["target_values"], case["lr"]
    )

    assert torch.allclose(ddp_param, reference_param, atol=1e-6)


def test_ddp_globally_empty_batch_gradients_are_finite_and_zero():
    case = {
        "kind": "empty_grad",
        "seq_lens": [3, 4],
        "selected_counts": [0, 0],
        "target_values": [1.0, 3.0],
    }

    ddp_grad = _run_ddp_case(case)

    assert torch.isfinite(ddp_grad)
    assert torch.equal(ddp_grad, torch.tensor(0.0))


def test_ddp_globally_empty_accumulation_window_leaves_state_unchanged():
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

    param_delta, lr, scheduler_epoch, optimizer_state_len = _run_ddp_case(case)

    assert param_delta == 0
    assert lr == pytest.approx(case["lr"])
    assert scheduler_epoch == 0
    assert optimizer_state_len == 0


def test_ddp_train_epoch_globally_empty_accumulation_window_leaves_state_unchanged():
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

    param_delta, lr, scheduler_epoch, optimizer_state_len = _run_ddp_case(case)

    assert param_delta == 0
    assert lr == pytest.approx(case["lr"])
    assert scheduler_epoch == 0
    assert optimizer_state_len == 0


def test_ddp_gradient_accumulation_preserves_per_microbatch_weighting():
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
        "lr": 0.05,
    }

    ddp_param = _run_ddp_case(case)
    reference_param = _analytical_accumulated_real_update(microbatches, case["lr"])
    combined_batch_param = _analytical_combined_real_update(microbatches, case["lr"])

    assert torch.allclose(ddp_param, reference_param, atol=1e-6)
    assert not torch.allclose(ddp_param, combined_batch_param, atol=1e-3)


def test_ddp_global_training_metric_finalization_matches_manual_sum_count():
    case = {
        "kind": "metrics",
        "cat_sums": [10.0, 5.0],
        "real_sums": [2.0, 8.0],
        "counts": [4.0, 6.0],
        "loss_weights": {"cat_col": 0.5, "real_col": 2.0},
    }

    total, cat_loss, real_loss = _run_ddp_case(case)

    global_count = sum(case["counts"])
    expected_cat = (
        sum(case["cat_sums"]) / global_count * case["loss_weights"]["cat_col"]
    )
    expected_real = (
        sum(case["real_sums"]) / global_count * case["loss_weights"]["real_col"]
    )

    assert cat_loss == pytest.approx(expected_cat)
    assert real_loss == pytest.approx(expected_real)
    assert total == pytest.approx(expected_cat + expected_real)
