# type: ignore

# from https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch. Thanks a lot!

import math

import torch
from torch.optim import Optimizer


class AdEMAMix(Optimizer):
    """Implements the AdEMAMix optimizer.

    This optimizer is based on the paper "AdEMAMix: A Novel Adaptive Optimizer for
    Deep Learning". It combines the advantages of Adam and EMA, and introduces a
    mixing term to further improve performance.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        learning_rate (float, optional): Learning rate (default: 1e-3).
        betas (Tuple[float, float, float], optional): Coefficients used for
            computing running averages of gradient and its square
            (default: (0.9, 0.999, 0.9999)).
        eps (float, optional): Term added to the denominator to improve
            numerical stability (default: 1e-8).
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0).
        alpha (float, optional): Mixing coefficient (default: 5.0).
        T_alpha_beta3 (int, optional): Time period for alpha and beta3 scheduling
            (default: None).
    """

    def __init__(
        self,
        params={},
        lr=1e-3,
        betas=(0.9, 0.999, 0.9999),
        eps=1e-8,
        weight_decay=0,
        alpha=5.0,
        T_alpha_beta3=None,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        assert len(betas) == 3, f"Invalid beta parameters: {betas}, expected 3"
        assert all(
            0.0 <= beta < 1.0 for beta in betas
        ), f"Invalid beta parameters: {betas}"
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            alpha=alpha,
            T_alpha_beta3=T_alpha_beta3,
        )
        super(AdEMAMix, self).__init__(params, defaults)

    def __setstate__(self, state):
        """Set the state of the optimizer.

        Args:
            state (dict): The state of the optimizer.
        """
        super(AdEMAMix, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss. (default: None)

        Returns:
            The loss, if the closure is provided. Otherwise, returns None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avg_slow = []
            state_steps = []

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError("AdEMAMix does not support sparse gradients")
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        # Slow exponential moving average
                        state["exp_avg_slow"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    exp_avg_slow.append(state["exp_avg_slow"])
                    state["step"] += 1
                    state_steps.append(state["step"])

            beta1, beta2, beta3 = group["betas"]
            alpha = group["alpha"]
            T_alpha_beta3 = group["T_alpha_beta3"]

            self._update_adamemix(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                exp_avg_slow,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                alpha=alpha,
                T_alpha_beta3=T_alpha_beta3,
                learning_rate=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
            )

        return loss

    def _update_adamemix(
        self,
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        exp_avg_slow,
        state_steps,
        beta1,
        beta2,
        beta3,
        alpha,
        T_alpha_beta3,
        learning_rate,
        weight_decay,
        eps,
    ):
        """Perform the AdEMAMix update for a single parameter group.

        Args:
            params (list[torch.Tensor]): List of parameters to update.
            grads (list[torch.Tensor]): List of gradients for each parameter.
            exp_avgs (list[torch.Tensor]): List of exponential moving averages of
                gradients.
            exp_avg_sqs (list[torch.Tensor]): List of exponential moving averages
                of squared gradients.
            exp_avg_slow (list[torch.Tensor]): List of slow exponential moving
                averages of gradients.
            state_steps (list[int]): List of steps for each parameter.
            beta1 (float): Coefficient for the first moment estimate.
            beta2 (float): Coefficient for the second moment estimate.
            beta3 (float): Coefficient for the slow moment estimate.
            alpha (float): Mixing coefficient.
            T_alpha_beta3 (int): Time period for alpha and beta3 scheduling.
            learning_rate (float): Learning rate.
            weight_decay (float): Weight decay.
            eps (float): Epsilon term for numerical stability.
        """
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            exp_avg_slow_i = exp_avg_slow[i]
            step = state_steps[i]

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            if T_alpha_beta3 is not None:
                alpha_t = min(step * alpha / T_alpha_beta3, alpha)
                beta3_t = min(
                    math.exp(
                        math.log(beta1)
                        * math.log(beta3)
                        / (
                            (1 - step / T_alpha_beta3) * math.log(beta3)
                            + (step / T_alpha_beta3) * math.log(beta1)
                        )
                    ),
                    beta3,
                )
            else:
                alpha_t = alpha
                beta3_t = beta3

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            exp_avg_slow_i.mul_(beta3_t).add_(grad, alpha=1 - beta3_t)

            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = learning_rate / bias_correction1

            if weight_decay != 0:
                param.add_(param, alpha=-weight_decay * learning_rate)

            param.addcdiv_(exp_avg + alpha_t * exp_avg_slow_i, denom, value=-step_size)
