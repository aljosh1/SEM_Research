"""Section 2.3 region characterization: the three-class decision rules."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import ClassificationConfig, VoidConfig, default_config
from src.region_classification import (
    CLASS_MIXED,
    CLASS_SURFACE,
    CLASS_VOID,
    classify_region,
    classify_regions,
    cutoff_intensity,
    histogram_peak_intensity,
)


CLS = ClassificationConfig()
VOID = VoidConfig()


# --------------------------------------------------------------------------- #
# Individual rules
# --------------------------------------------------------------------------- #
def test_region_without_void_is_surface():
    cls, why = classify_region(np.array([100, 120], np.uint8), 0, False, 10,
                               100.0, 30.0, CLS, VOID)
    assert cls == CLASS_SURFACE
    assert "no void" in why


def test_region_fully_covered_is_void():
    values = np.array([5, 6, 7, 8], np.uint8)
    cls, why = classify_region(values, area_void := 4, True, 10, 100.0, 30.0, CLS, VOID)
    assert area_void == len(values)
    assert cls == CLASS_VOID
    assert "fully covered" in why


def test_void_requires_both_ip_and_ic_conditions():
    """I_p <= I_t AND I_c < mu_g + j*sigma_g."""
    # peak at 0 (<= I_t=10); I_c(80) ~ 8 < limit 130 -> VOID
    dark = np.concatenate([np.zeros(80, np.uint8), np.full(20, 15, np.uint8)])
    cls, _ = classify_region(dark, 60, True, 10, 100.0, 30.0, CLS, VOID)
    assert cls == CLASS_VOID

    # peak at 15 (> I_t) even though I_c is small -> MIXED
    brighter = np.concatenate([np.zeros(20, np.uint8), np.full(80, 15, np.uint8)])
    cls2, why2 = classify_region(brighter, 20, True, 10, 100.0, 30.0, CLS, VOID)
    assert cls2 == CLASS_MIXED
    assert "I_p" in why2


def test_ic_failure_alone_forces_mixed():
    # Uniformly bright region with a small tip: peak 200 > I_t, I_c 200 >= limit
    values = np.concatenate([np.full(10, 5, np.uint8), np.full(90, 200, np.uint8)])
    cls, why = classify_region(values, 10, True, 10, 100.0, 30.0, CLS, VOID)
    assert cls == CLASS_MIXED


def test_missing_i_t_cannot_be_void():
    values = np.full(50, 10, np.uint8)
    cls, why = classify_region(values, 10, True, None, 100.0, 30.0, CLS, VOID)
    assert cls == CLASS_MIXED
    assert "unavailable" in why


def test_histogram_peak_intensity_uses_most_populated_bin():
    values = np.concatenate([np.full(10, 30, np.uint8), np.full(50, 60, np.uint8)])
    peak, counts, centres = histogram_peak_intensity(values, bin_width=2)
    assert 55 <= peak <= 65
    assert counts.sum() == len(values)


def test_cutoff_intensity_inverted_cdf_definition():
    """I_c is the smallest intensity with >= k% of pixels below it."""
    values = np.arange(100, dtype=np.uint8)  # 0..99
    ic = cutoff_intensity(values, 80, "inverted_cdf")
    assert ic == pytest.approx(79.0)     # 80 values (0..79) are <= 79
    assert (values < ic).sum() / values.size == pytest.approx(0.79)


def test_cutoff_intensity_rejects_bad_input():
    with pytest.raises(ValueError):
        cutoff_intensity(np.array([], np.uint8), 80, "inverted_cdf")
    with pytest.raises(ValueError):
        cutoff_intensity(np.array([1, 2], np.uint8), 0, "inverted_cdf")


def test_classify_region_rejects_empty():
    with pytest.raises(ValueError, match="no pixels"):
        classify_region(np.array([], np.uint8), 0, False, None, 1.0, 1.0, CLS, VOID)


# --------------------------------------------------------------------------- #
# Whole-image classification
# --------------------------------------------------------------------------- #
def _labels_and_void(img, region_specs):
    labels = np.zeros(img.shape, np.int32)
    void_stats = {}
    for rid, (mask, has_tip, void_area) in enumerate(region_specs, start=1):
        labels[mask] = rid
        void_stats[rid] = {
            "has_tip": has_tip,
            "void_area": void_area,
            "tip_pixels": 5 if has_tip else 0,
            "extension_threshold": 20.0 if has_tip else None,
        }
    return labels, void_stats


def test_three_classes_are_assigned_and_consistent():
    h, w = 60, 90
    img = np.full((h, w), 120, dtype=np.uint8)
    a = np.zeros((h, w), bool); a[5:25, 5:30] = True
    b = np.zeros((h, w), bool); b[5:25, 35:60] = True
    c = np.zeros((h, w), bool); c[30:50, 5:30] = True
    d = np.zeros((h, w), bool); d[30:50, 35:85] = True

    img[a] = 3                                   # deep void: fully covered
    img.flat[np.flatnonzero(b.ravel())[:100]] = 3  # partially dark: 100 of 500 px
    labels, stats = _labels_and_void(
        img,
        [(a, True, int(a.sum())),           # fully covered -> VOID
         (b, True, 100),                    # partially -> MIXED
         (c, False, 0),                     # no void -> SURFACE
         (d, False, 0)],
    )
    res = classify_regions(img, labels, stats, 10.0, CLS, VOID, n_regions=4)
    classes = [res.region_class[i] for i in range(1, 5)]
    assert classes == [CLASS_VOID, CLASS_MIXED, CLASS_SURFACE, CLASS_SURFACE]
    assert res.diagnostics["counts"] == {"VOID": 1, "SURFACE": 2, "MIXED": 1}
    assert res.class_image.shape == img.shape
    assert res.class_image.dtype == np.uint8
    # class_image agrees with class_map on every labelled pixel
    for rid in range(1, 5):
        m = labels == rid
        assert np.all(res.class_image[m] == res.region_class[rid])


def test_classification_statistics_are_recorded():
    img = np.full((30, 30), 100, np.uint8)
    img[5:15, 5:15] = 2
    mask = np.ones((30, 30), bool)
    labels, stats = _labels_and_void(img, [(mask, True, 100)])
    res = classify_regions(img, labels, stats, 10.0, CLS, VOID, n_regions=1)
    s = res.stats[0]
    assert s.area == 900
    assert s.void_area == 100
    assert s.void_fraction == pytest.approx(100 / 900)
    assert s.peak_intensity == 100    # majority of the region is the brighter surface
    assert res.diagnostics["k_percent"] == 80.0
    assert res.diagnostics["j"] == 1.0


def test_no_tip_detected_means_no_void_regions():
    img = np.full((20, 20), 150, np.uint8)
    labels = np.ones((20, 20), np.int32)
    res = classify_regions(img, labels, {}, None, CLS, VOID, n_regions=1)
    assert res.region_class[1] == CLASS_SURFACE
    assert res.diagnostics["counts"]["VOID"] == 0


def test_empty_region_ids_are_skipped():
    img = np.full((10, 10), 100, np.uint8)
    labels = np.zeros((10, 10), np.int32)
    labels[0:5, :] = 1
    labels[7:10, :] = 3          # gap: label 2 never appears
    areas = np.array([0, 50, 0, 30])
    stats = {1: {"has_tip": False, "void_area": 0}}
    res = classify_regions(img, labels, stats, None, CLS, VOID, n_regions=3)
    assert 1 in res.region_class
    assert 3 in res.region_class
    assert 2 not in res.region_class


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        classify_regions(np.zeros((5, 5), np.uint8), np.zeros((4, 4), np.int32),
                         {}, None, CLS, VOID)


def test_classification_is_deterministic(synthetic_field):
    cfg = default_config()
    from src.contour_detection import detect_initial_contours
    from src.contour_refinement import refine_contours
    from src.void_detection import detect_voids

    init = detect_initial_contours(synthetic_field, cfg.binarization, cfg.contour)
    ref = refine_contours(synthetic_field, init.labels, init.areas, init.statuses, cfg.contour)
    tip, ext = detect_voids(synthetic_field, ref.labels, cfg.void)
    a = classify_regions(synthetic_field, ref.labels, ext.per_region, tip.threshold,
                         cfg.classification, cfg.void, ref.n_regions)
    b = classify_regions(synthetic_field, ref.labels, ext.per_region, tip.threshold,
                         cfg.classification, cfg.void, ref.n_regions)
    np.testing.assert_array_equal(a.class_image, b.class_image)
