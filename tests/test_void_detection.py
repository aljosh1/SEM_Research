"""Section 2.3: void-tip detection and void extension."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import VoidConfig
from src.void_detection import (
    detect_void_tips,
    detect_voids,
    extend_voids,
    extension_threshold,
    find_first_local_maximum,
)


# --------------------------------------------------------------------------- #
# Peak selection: "the local peak in the low range"
# --------------------------------------------------------------------------- #
def test_first_local_maximum_returns_lowest_index():
    counts = np.array([0, 5, 10, 4, 20, 9, 30, 1])
    assert find_first_local_maximum(counts) == 2


def test_first_local_maximum_includes_bin_zero():
    """A saturated dark-void spike in bin 0 is a legitimate local maximum.

    Skipping bin 0 was an implementation defect found on real SEM data: it
    returned a spurious peak in the body of the histogram.
    """
    counts = np.array([500, 20, 60, 10, 30, 5])
    assert find_first_local_maximum(counts) == 0


def test_first_local_maximum_none_for_rising_histogram():
    # strictly rising: no bin exceeds its next neighbour
    assert find_first_local_maximum(np.array([1, 2, 3, 4, 5])) is None


def test_first_local_maximum_falls_back_to_bin_zero_when_descending():
    # a monotonically falling histogram has its only peak at the lowest level
    assert find_first_local_maximum(np.array([5, 4, 3, 2, 1])) == 0


def test_first_local_maximum_ignores_flat_plateaus():
    # equal neighbours do not qualify as a strict maximum
    assert find_first_local_maximum(np.array([7, 7, 7, 7])) is None


def test_first_local_maximum_needs_three_bins():
    assert find_first_local_maximum(np.array([1, 2])) is None
    assert find_first_local_maximum(np.array([0])) is None


# --------------------------------------------------------------------------- #
# I_t and the tip mask
# --------------------------------------------------------------------------- #
def _bimodal_image() -> np.ndarray:
    """Dark tip population near 5, surface body near 150, bright rim near 240."""
    img = np.full((120, 120), 150, dtype=np.uint8)
    img[20:60, 20:60] = 4            # void tips
    img[0:4, :] = 240                # bright rim
    return img


def test_tip_threshold_comes_from_low_range_peak():
    img = _bimodal_image()
    res = detect_void_tips(img, VoidConfig())
    assert res.status == "ok"
    assert res.threshold is not None
    # The tip population occupies intensities 0..7, i.e. the lowest bins.
    assert res.peak_bin_index <= 3
    assert res.threshold <= 10
    assert res.tip_mask.dtype == np.bool_
    assert res.tip_mask[30, 30]
    assert not res.tip_mask[100, 100]


def test_tip_mask_matches_intensity_rule():
    img = _bimodal_image()
    res = detect_void_tips(img, VoidConfig())
    np.testing.assert_array_equal(res.tip_mask, img < res.threshold)


def test_uniform_image_reports_failure_not_a_threshold():
    res = detect_void_tips(np.full((50, 50), 100, np.uint8), VoidConfig())
    assert res.threshold is None
    assert "uniform" in res.status


def test_histogram_without_peak_reports_failure():
    ramp = np.tile(np.arange(256, dtype=np.uint8), (20, 1))
    res = detect_void_tips(ramp, VoidConfig())
    assert res.threshold is None
    assert "local maximum" in res.status or "uniform" in res.status


def test_empty_image_raises():
    with pytest.raises(ValueError, match="empty"):
        detect_void_tips(np.zeros((0, 0), np.uint8), VoidConfig())


def test_non_2d_image_raises():
    with pytest.raises(ValueError, match="2-D"):
        detect_void_tips(np.zeros((4, 4, 3), np.uint8), VoidConfig())


# --------------------------------------------------------------------------- #
# Extension: I_v = max(mu_l - 0.5 sigma_l, I_t)
# --------------------------------------------------------------------------- #
def test_extension_threshold_formula():
    values = np.array([0, 100, 0, 100], dtype=np.uint8)  # mu 50, sigma 50
    assert extension_threshold(values, 10.0, VoidConfig()) == pytest.approx(25.0)


def test_extension_threshold_never_below_i_t():
    values = np.full(10, 200, dtype=np.uint8)  # mu - 0.5*0 = 200
    assert extension_threshold(values, 50.0, VoidConfig()) == pytest.approx(200.0)
    flat_dark = np.full(10, 5, dtype=np.uint8)
    assert extension_threshold(flat_dark, 40.0, VoidConfig()) == pytest.approx(40.0)


def test_extension_only_processes_regions_with_tips():
    img = _bimodal_image()
    labels = np.zeros(img.shape, np.int32)
    labels[:, :] = 1
    labels[80:, :] = 2     # region without tips
    tip = detect_void_tips(img, VoidConfig())
    ext = extend_voids(img, labels, tip, VoidConfig())
    assert ext.per_region[1]["has_tip"] is True
    assert ext.per_region[2]["has_tip"] is False
    assert ext.per_region[2]["void_area"] == 0


def test_extension_stays_inside_its_region():
    img = _bimodal_image()
    labels = np.ones(img.shape, np.int32)
    tip = detect_void_tips(img, VoidConfig())
    ext = extend_voids(img, labels, tip, VoidConfig(restrict_to_region=True))
    assert np.all(labels[ext.void_mask] == 1)


def test_extension_skipped_when_tip_detection_failed():
    img = np.full((40, 40), 100, np.uint8)
    tip = detect_void_tips(img, VoidConfig())
    ext = extend_voids(img, np.ones(img.shape, np.int32), tip, VoidConfig())
    assert ext.diagnostics["status"] == "skipped"
    assert not ext.void_mask.any()


def test_shape_mismatch_raises():
    img = np.zeros((10, 10), np.uint8)
    tip = detect_void_tips(np.full((10, 10), 50, np.uint8), VoidConfig())
    with pytest.raises(ValueError, match="same shape"):
        extend_voids(img, np.zeros((8, 8), np.int32), tip, VoidConfig())


def test_detect_voids_wrapper_returns_both_products():
    img = _bimodal_image()
    labels = np.ones(img.shape, np.int32)
    tip, ext = detect_voids(img, labels, VoidConfig())
    assert tip.threshold is not None
    assert ext.void_mask.shape == img.shape
    assert ext.void_mask.sum() >= tip.tip_mask.sum()


def test_extension_is_deterministic():
    img = _bimodal_image()
    labels = np.ones(img.shape, np.int32)
    a = detect_voids(img, labels, VoidConfig())[1].void_mask
    b = detect_voids(img, labels, VoidConfig())[1].void_mask
    np.testing.assert_array_equal(a, b)
