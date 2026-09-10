from __future__ import annotations

import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    """DreamerV3 RMS normalization with a learned scale and no mean subtraction."""

    def __init__(self, dim: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (value.float() * rms * self.scale.float()).to(value.dtype)


class BlockLinear(nn.Module):
    """Independent linear projections over equally sized feature blocks."""

    def __init__(self, input_dim: int, output_dim: int, blocks: int, bias: bool = True) -> None:
        super().__init__()
        if input_dim % blocks or output_dim % blocks:
            raise ValueError("input_dim and output_dim must be divisible by blocks")
        self.blocks = blocks
        self.input_per_block = input_dim // blocks
        self.output_per_block = output_dim // blocks
        self.weight = nn.Parameter(
            torch.empty(blocks, self.input_per_block, self.output_per_block)
        )
        self.bias = nn.Parameter(torch.zeros(output_dim)) if bias else None
        trunc_normal_fan_in_(self.weight, fan_in=self.input_per_block)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape[:-1]
        grouped = value.reshape(*shape, self.blocks, self.input_per_block)
        output = torch.einsum("...bi,bio->...bo", grouped, self.weight)
        output = output.reshape(*shape, self.blocks * self.output_per_block)
        return output if self.bias is None else output + self.bias


def trunc_normal_fan_in_(tensor: torch.Tensor, *, fan_in: int | None = None) -> None:
    """DreamerV3's variance-corrected truncated-normal fan-in initialization."""

    if fan_in is None:
        if tensor.ndim == 2:
            fan_in = tensor.shape[1]
        elif tensor.ndim >= 3:
            fan_in = tensor.shape[1] * math.prod(tensor.shape[2:])
        else:
            fan_in = 1
    std = 1.1368 / math.sqrt(max(fan_in, 1))
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


def init_dreamer_module(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        trunc_normal_fan_in_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def scale_output_(module: nn.Linear | nn.Conv2d, scale: float) -> None:
    with torch.no_grad():
        module.weight.mul_(scale)

