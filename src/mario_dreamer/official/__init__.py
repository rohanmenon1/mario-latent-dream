"""Adapters for running the pinned upstream DreamerV3 implementation."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mario_dreamer.official.mario_env import MarioEmbodiedEnv

__all__ = ["MarioEmbodiedEnv"]


def __getattr__(name: str):
    if name == "MarioEmbodiedEnv":
        from mario_dreamer.official.mario_env import MarioEmbodiedEnv

        return MarioEmbodiedEnv
    raise AttributeError(name)
