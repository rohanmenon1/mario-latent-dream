from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    obs_size: int = 64
    grayscale: bool = False
    channels_first: bool = True
    crop_top: int = 32
    crop_bottom: int = 0
    crop_left: int = 0
    crop_right: int = 0


class FramePreprocessor:
    """Resize and normalize Mario frames for model input."""

    def __init__(self, config: PreprocessConfig) -> None:
        self.config = config

    @property
    def output_shape(self) -> tuple[int, ...]:
        channels = 1 if self.config.grayscale else 3
        if self.config.channels_first:
            return (channels, self.config.obs_size, self.config.obs_size)
        return (self.config.obs_size, self.config.obs_size, channels)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"expected RGB frame with shape HxWx3, got {frame.shape}")

        height, width = frame.shape[:2]
        y0 = self.config.crop_top
        y1 = height - self.config.crop_bottom
        x0 = self.config.crop_left
        x1 = width - self.config.crop_right
        if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
            raise ValueError(
                "invalid crop bounds for frame "
                f"{frame.shape}: top={self.config.crop_top}, bottom={self.config.crop_bottom}, "
                f"left={self.config.crop_left}, right={self.config.crop_right}"
            )
        frame = frame[y0:y1, x0:x1]

        resized = cv2.resize(
            frame,
            (self.config.obs_size, self.config.obs_size),
            interpolation=cv2.INTER_AREA,
        )

        if self.config.grayscale:
            resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)[..., None]

        normalized = resized.astype(np.float32) / 255.0
        if self.config.channels_first:
            normalized = np.transpose(normalized, (2, 0, 1))
        return normalized
