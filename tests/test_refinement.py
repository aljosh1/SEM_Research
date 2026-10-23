"""Section 2.2 refinement: candidate rule and local threshold."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import ContourConfig, default_config
from src.contour_detection import detect_initial_contours
from src.contour_refinement import (
    local_threshold,
    refine_contours,
    region_area_statistics,
    select_candidates,
)


def test_area_statistics_mean_and_std():
    areas = np.array([0, 100, 200, 300], dtype=np.int64)
    statuses = ["background", "closed", "closed", "closed"]
    mu, sd = region_area_statistics(areas, statuses, exclude_border=False)
    assert mu == pytest.approx(200.0)
    assert sd == pytest.approx(np.std([100, 200, 300]))


def test_border_region_is_excluded_when_requested():
    areas = np.array([0, 100, 200, 100_000], dtype=np.int64)
    statuses = ["background", "closed", "closed", "border"]
    mu_all, _ = region_area_statistics(areas, statuses, exclude_border=False)
    mu_ex, _ = region_area_statistics(areas, statuses, exclude_border=True)
    assert mu_all > mu_ex
    assert mu_ex == pytest.approx(150.0)


def test_area_statistics_of_empty_and_single_region():
    assert region_area_statistics(np.zeros(1, np.int64), ["background"]) == (0.0, 0.0)
    mu, sd = region_area_statistics(np.array([0, 50], np.int64),
                                    ["background", "closed"], exclude_border=False)
    assert mu == 50.0 and sd == 0.0


def test_candidate_rule_is_strictly_greater_than_mu_plus_sigma():
    # mean 200, std 81.65 -> cut-off 281.65 ; 300 qualifies, 200 does not
    areas = np.array([0, 100, 200, 300], dtype=np.int64)
    statuses = ["background"] + ["closed"] * 3
    ids, mu, sd, cut = select_candidates(areas, statuses, ContourConfig(), exclude_border=False)
    assert ids == [3]
    assert cut == pytest.approx(mu + sd, rel=1e-12)


def test_candidate_rule_with_zero_std_returns_nothing_above_cut():
    areas = np.array([0, 100, 100], dtype=np.int64)
    statuses = ["background", "closed", "closed"]
    ids, _, _, cut = select_candidates(areas, statuses, ContourConfig(), exclude_border=False)
    assert ids == []  # no region is strictly greater than mu + 0


def test_candidate_rule_respects_coefficient():
    # mean = 200, std = 81.65.  With c=0 the cut-off is 200 and the rule is
    # strict (S_i > 200), so only 300 qualifies.  With c=2 the cut-off is 363
    # and nothing qualifies.
    areas = np.array([0, 100, 200, 300], dtype=np.int64)
    statuses = ["background"] + ["closed"] * 3
    ids, _, _, cut = select_candidates(areas, statuses,
                                       ContourConfig(area_sigma_coeff=0.0),
                                       exclude_border=False)
    assert cut == pytest.approx(200.0)
    assert ids == [3]
    ids, _, _, _ = select_candidates(areas, statuses,
                                     ContourConfig(area_sigma_coeff=2.0),
                                     exclude_border=False)
    assert ids == []


def test_local_threshold_formula():
    values = np.full(100, 50, dtype=np.uint8)
    assert local_threshold(values, ContourConfig(local_sigma_coeff=0.5)) == pytest.approx(50.0)
    values2 = np.array([0, 100, 0, 100], dtype=np.uint8)  # mu 50, sigma 50
    assert local_threshold(values2, ContourConfig(local_sigma_coeff=0.5)) == pytest.approx(75.0)


def test_local_threshold_rejects_empty_region():
    with pytest.raises(ValueError, match="no pixels"):
        local_threshold(np.array([], dtype=np.uint8), ContourConfig())


def test_refinement_splits_a_large_bimodal_region():
    """A large region containing a dark core must split at its local threshold.

    Six small regions give a realistic mu_S / sigma_S, so the large region is a
    genuine candidate under the paper's rule.
    """
    gray = np.full((120, 200), 150, dtype=np.uint8)
    labels = np.ones(gray.shape, dtype=np.int32)

    # One large region (rows 20..110, cols 20..130); the dark core covers about
    # half of it, so mu_l - 0.5 sigma_l falls below the surface intensity.
    big = np.zeros(gray.shape, dtype=bool)
    big[20:110, 20:130] = True
    gray[30:100, 30:120] = 20
    labels[big] = 1

    # Five small uniform regions
    for i, c0 in enumerate(range(140, 190, 10), start=2):
        small = np.zeros(gray.shape, dtype=bool)
        small[20:35, c0:c0 + 7] = True
        labels[small] = i


    areas = np.bincount(labels.ravel()).astype(np.int64)
    areas[0] = 0
    statuses = ["background"] + ["closed"] * (len(areas) - 1)

    res = refine_contours(gray, labels, areas, statuses, ContourConfig(),
                          exclude_border_from_statistics=True)
    assert res.diagnostics["n_candidates"] == 1
    assert len(res.refined_regions) == 1
    assert res.refined_regions[0]["region_id"] == 1
    assert 20 < res.refined_regions[0]["local_threshold"] < 150
    assert res.refined_regions[0]["pieces"] >= 2   # the large region did split
    assert res.n_regions > 6
    assert np.all(res.labels > 0)  # labels still tile the image


def test_refinement_is_deterministic(synthetic_field):
    cfg = default_config()
    init = detect_initial_contours(synthetic_field, cfg.binarization, cfg.contour)
    a = refine_contours(synthetic_field, init.labels, init.areas, init.statuses, cfg.contour)
    b = refine_contours(synthetic_field, init.labels, init.areas, init.statuses, cfg.contour)
    np.testing.assert_array_equal(a.labels, b.labels)
    assert a.n_regions == b.n_regions


def test_refined_labels_tile_the_image(synthetic_field):
    cfg = default_config()
    init = detect_initial_contours(synthetic_field, cfg.binarization, cfg.contour)
    res = refine_contours(synthetic_field, init.labels, init.areas, init.statuses,
                          cfg.contour)
    assert np.all(res.labels > 0)
    assert res.areas[1:].sum() == synthetic_field.size


def test_refinement_rejects_mismatched_arrays():
    gray = np.zeros((10, 10), np.uint8)
    with pytest.raises(ValueError, match="equal shape"):
        refine_contours(gray, np.zeros((8, 8), np.int32), np.array([0, 5]),
                        ["background", "closed"], ContourConfig())
    with pytest.raises(ValueError, match="one entry per label"):
        refine_contours(gray, np.ones((10, 10), np.int32), np.array([0, 5]),
                        ["background"], ContourConfig())
