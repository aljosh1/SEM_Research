"""Preprocessing behaviour and parameter validation."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import PreprocessingConfig
from src.preprocessing import preprocess


def test_disabled_preprocessing_is_identity(synthetic_field):
    res = preprocess(synthetic_field, PreprocessingConfig(enabled=False))
    assert not res.applied
    assert res.operations == []
    np.testing.assert_array_equal(res.image, synthetic_field)


def test_disabled_preprocessing_still_returns_a_copy(synthetic_field):
    res = preprocess(synthetic_field, PreprocessingConfig(enabled=False))
    res.image[0, 0] = 255
    assert synthetic_field[0, 0] != 255


def test_enabled_without_operations_applies_nothing(synthetic_field):
    res = preprocess(synthetic_field, PreprocessingConfig(enabled=True))
    assert res.operations == []
    np.testing.assert_array_equal(res.image, synthetic_field)


def test_gaussian_smoothing_reduces_local_variance(synthetic_field):
    cfg = PreprocessingConfig(enabled=True, gaussian_sigma=2.0)
    res = preprocess(synthetic_field, cfg)
    assert res.applied
    assert res.operations[0]["operation"] == "gaussian_blur"
    assert res.image.dtype == np.uint8
    # Laplacian energy is a proxy for high-frequency content
    import cv2
    before = cv2.Laplacian(synthetic_field, cv2.CV_64F).var()
    after = cv2.Laplacian(res.image, cv2.CV_64F).var()
    assert after < before


def test_median_filter_removes_isolated_pixels(uniform_image):
    img = uniform_image.copy()
    img[10, 10] = 255
    img[30, 40] = 0
    cfg = PreprocessingConfig(enabled=True, median_kernel=3)
    res = preprocess(img, cfg)
    assert res.image[10, 10] == 128
    assert res.image[30, 40] == 128


def test_percentile_stretch_spans_full_range(synthetic_field):
    cfg = PreprocessingConfig(enabled=True, percentile_clip=(1.0, 99.0))
    res = preprocess(synthetic_field, cfg)
    assert res.image.min() <= 5
    assert res.image.max() >= 250
    assert res.operations[0]["operation"] == "percentile_stretch"


def test_preserves_shape_and_dtype(synthetic_field):
    cfg = PreprocessingConfig(enabled=True, gaussian_sigma=1.0, median_kernel=3,
                              percentile_clip=(2.0, 98.0))
    res = preprocess(synthetic_field, cfg)
    assert res.image.shape == synthetic_field.shape
    assert res.image.dtype == np.uint8
    assert len(res.operations) == 3


def test_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2-D"):
        preprocess(np.zeros((4, 4, 3), np.uint8), PreprocessingConfig())


def test_rejects_wrong_dtype():
    with pytest.raises(ValueError, match="uint8"):
        preprocess(np.zeros((4, 4), np.float64), PreprocessingConfig())


def test_rejects_empty_image():
    with pytest.raises(ValueError, match="empty"):
        preprocess(np.zeros((0, 0), np.uint8), PreprocessingConfig())


def test_rejects_even_median_kernel(synthetic_field):
    with pytest.raises(ValueError, match="odd"):
        preprocess(synthetic_field, PreprocessingConfig(enabled=True, median_kernel=4))


def test_rejects_invalid_percentile_clip(synthetic_field):
    with pytest.raises(ValueError, match="percentile_clip"):
        preprocess(synthetic_field, PreprocessingConfig(enabled=True, percentile_clip=(90.0, 10.0)))


def test_config_validation_rejects_bad_values():
    from src.config import SegmentationConfig

    cfg = SegmentationConfig()
    cfg.binarization.sigma_multiplier = 0
    with pytest.raises(ValueError, match="sigma_multiplier"):
        cfg.validate()

    cfg = SegmentationConfig()
    cfg.preprocessing.median_kernel = 4
    with pytest.raises(ValueError, match="median_kernel"):
        cfg.validate()

    cfg = SegmentationConfig()
    cfg.calibration.pixel_size_um = -1
    with pytest.raises(ValueError, match="pixel_size_um"):
        cfg.validate()

    cfg = SegmentationConfig()
    cfg.classification.cutoff_percentile = 0
    with pytest.raises(ValueError, match="cutoff_percentile"):
        cfg.validate()
