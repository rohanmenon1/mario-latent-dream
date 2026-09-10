import numpy as np

from mario_dreamer.envs import FramePreprocessor, PreprocessConfig


def test_rgb_preprocessing_shape_and_range():
    frame = np.full((240, 256, 3), 128, dtype=np.uint8)
    preprocess = FramePreprocessor(PreprocessConfig(obs_size=64, grayscale=False))

    output = preprocess(frame)

    assert output.shape == (3, 64, 64)
    assert output.dtype == np.float32
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_grayscale_preprocessing_shape():
    frame = np.zeros((240, 256, 3), dtype=np.uint8)
    preprocess = FramePreprocessor(PreprocessConfig(obs_size=32, grayscale=True))

    output = preprocess(frame)

    assert output.shape == (1, 32, 32)


def test_preprocessing_crops_before_resize():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:4] = 255
    preprocess = FramePreprocessor(
        PreprocessConfig(obs_size=4, grayscale=False, channels_first=False, crop_top=4)
    )

    output = preprocess(frame)

    assert output.shape == (4, 4, 3)
    assert output.max() == 0.0


def test_rejects_non_rgb_frames():
    frame = np.zeros((64, 64), dtype=np.uint8)
    preprocess = FramePreprocessor(PreprocessConfig())

    try:
        preprocess(frame)
    except ValueError as exc:
        assert "expected RGB frame" in str(exc)
    else:
        raise AssertionError("expected ValueError")
