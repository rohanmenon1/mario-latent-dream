from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.optim import Optimizer


class DreamerOptimizer(Optimizer):
    """PyTorch port of DreamerV3's AGC -> RMS scaling -> momentum optimizer."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float = 4e-5,
        agc: float = 0.3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-20,
        warmup_steps: int = 1000,
    ) -> None:
        defaults = dict(
            lr=lr,
            agc=agc,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            warmup_steps=warmup_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("DreamerOptimizer does not support sparse gradients")
                state = self.state[parameter]
                if "rms" not in state or "momentum" not in state:
                    state.clear()
                    state["step"] = 0
                    state["rms"] = torch.zeros_like(parameter)
                    state["momentum"] = torch.zeros_like(parameter)

                state["step"] += 1
                step = state["step"]
                clipped = adaptive_gradient_clip(parameter, grad, group["agc"])
                rms = state["rms"]
                rms.mul_(group["beta2"]).addcmul_(
                    clipped,
                    clipped,
                    value=1.0 - group["beta2"],
                )
                rms_hat = rms / (1.0 - group["beta2"] ** step)
                scaled = clipped / (torch.sqrt(rms_hat) + group["eps"])
                momentum = state["momentum"]
                momentum.mul_(group["beta1"]).add_(scaled, alpha=1.0 - group["beta1"])
                momentum_hat = momentum / (1.0 - group["beta1"] ** step)
                warmup = group["warmup_steps"]
                rate = group["lr"] * (min(step / warmup, 1.0) if warmup else 1.0)
                parameter.add_(momentum_hat, alpha=-rate)
        return loss


def adaptive_gradient_clip(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    clipping: float,
    eps: float = 1e-3,
) -> torch.Tensor:
    if clipping <= 0.0:
        return gradient
    parameter_norm = torch.linalg.vector_norm(parameter.flatten()).clamp_min(eps)
    gradient_norm = torch.linalg.vector_norm(gradient.flatten()).clamp_min(1e-6)
    maximum = clipping * parameter_norm
    scale = torch.clamp(maximum / gradient_norm, max=1.0)
    return gradient * scale
