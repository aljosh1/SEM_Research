"""Threshold, thinning, edge linking and region labelling (paper section 2.1)."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import BinarizationConfig, ContourConfig, default_config
from src.contour_detection import (
    binned_histogram,
    count_neighbours,
    detect_initial_contours,
    endpoint_mask,
    global_threshold,
    gradient_direction,
    label_regions,
    link_edges,
    neighbourhood_offsets,
    thin,
)


# --------------------------------------------------------------------------- #
# Threshold: th = mu + 2*sigma over the intermediate range
# --------------------------------------------------------------------------- #
def test_threshold_formula_matches_mu_plus_two_sigma(synthetic_field):
    cfg = BinarizationConfig()
    res = global_threshold(synthetic_field, cfg)
    assert res.threshold == pytest.approx(res.mean + 2.0 * res.std)
    assert res.range_low < res.range_high


def test_extreme_bands_are_excluded_from_statistics(synthetic_field):
    # Two configurations that exclude different tails must give different mu.
    a = global_threshold(synthetic_field, BinarizationConfig(extreme_low_fraction=0.0,
                                                             extreme_high_fraction=0.0))
    b = global_threshold(synthetic_field, BinarizationConfig(extreme_low_fraction=0.25,
                                                             extreme_high_fraction=0.25))
    assert a.mean != pytest.approx(b.mean)
    assert b.range_low > a.range_low
    assert b.range_high < a.range_high


def test_binary_is_boolean_and_matches_rule(synthetic_field):
    res = global_threshold(synthetic_field, BinarizationConfig())
    assert res.binary.dtype == np.bool_
    np.testing.assert_array_equal(res.binary, synthetic_field > res.threshold)


def test_higher_sigma_multiplier_keeps_fewer_boundary_pixels(synthetic_field):
    low = global_threshold(synthetic_field, BinarizationConfig(sigma_multiplier=1.0))
    high = global_threshold(synthetic_field, BinarizationConfig(sigma_multiplier=3.0))
    assert high.binary.sum() < low.binary.sum()


def test_uniform_image_threshold_does_not_crash(uniform_image):
    # sigma == 0, so th == mu; the strict '>' rule then keeps every pixel.
    res = global_threshold(uniform_image, BinarizationConfig())
    assert res.threshold == pytest.approx(float(uniform_image.mean()), abs=1.0)
    assert res.std == 0.0
    assert res.binary.all()


def test_threshold_rejects_bad_input():
    with pytest.raises(ValueError, match="2-D"):
        global_threshold(np.zeros((2, 2, 2), np.uint8), BinarizationConfig())
    with pytest.raises(ValueError, match="empty"):
        global_threshold(np.zeros((0, 0), np.uint8), BinarizationConfig())


def test_binned_histogram_sums_to_total(synthetic_field):
    counts, centres = binned_histogram(synthetic_field, 5)
    assert counts.sum() == synthetic_field.size
    assert len(counts) == len(centres)


# --------------------------------------------------------------------------- #
# Thinning
# --------------------------------------------------------------------------- #
def test_thinning_produces_single_pixel_width():
    thick = np.zeros((40, 40), dtype=bool)
    thick[10:30, 10:25] = True  # 15-px-wide bar
    sk = thin(thick)
    assert sk.sum() < thick.sum()
    # each skeleton pixel has at most 2 neighbours (no thick clusters)
    assert count_neighbours(sk)[sk].max() <= 2


def test_thinning_of_empty_image_is_empty():
    sk = thin(np.zeros((10, 10), dtype=bool))
    assert sk.sum() == 0


def test_thinning_preserves_connectivity_of_a_line():
    line = np.zeros((20, 20), dtype=bool)
    line[10, 2:18] = True
    sk = thin(line)
    assert sk.sum() == 16
    assert sk[10].sum() == 16


# --------------------------------------------------------------------------- #
# Neighbour counting / endpoints
# --------------------------------------------------------------------------- #
def test_count_neighbours_does_not_wrap_the_frame():
    img = np.zeros((5, 5), dtype=bool)
    img[0, 0] = True
    img[4, 0] = True
    n = count_neighbours(img)
    assert n[0, 0] == 0  # no wrap from the opposite edge


def test_endpoint_mask_finds_ends_of_a_line():
    line = np.zeros((20, 20), dtype=bool)
    line[10, 5:15] = True
    ep = endpoint_mask(line)
    assert ep.sum() == 2
    assert ep[10, 5] and ep[10, 14]


# --------------------------------------------------------------------------- #
# Edge linking
# --------------------------------------------------------------------------- #
def test_neighbourhood_offsets_grow_with_k():
    offs = neighbourhood_offsets(5, 7)
    assert any(abs(dy) == 2 for dy, dx in offs)
    assert any(abs(dy) == 3 for dy, dx in offs)


def test_neighbourhood_offsets_reject_invalid_k():
    with pytest.raises(ValueError):
        link_edges(np.zeros((5, 5), bool), gradient_direction(np.zeros((5, 5), np.uint8)),
                   ContourConfig(link_k_start=4))
    with pytest.raises(ValueError):
        link_edges(np.zeros((5, 5), bool), gradient_direction(np.zeros((5, 5), np.uint8)),
                   ContourConfig(link_k_start=7, link_k_max=5))


def test_linking_joins_two_collinear_segments():
    gray = np.zeros((40, 40), dtype=np.uint8)
    gray[15:25, :] = 200  # a horizontal bright band -> horizontal edges
    # A 2-px gap is inside the k=5 neighbourhood, so the linker must close it.
    seg = np.zeros((40, 40), dtype=bool)
    seg[12, 5:14] = True
    seg[12, 16:32] = True
    direction = gradient_direction(gray)
    linked, diag = link_edges(seg, direction, ContourConfig())
    assert diag["links"] >= 1
    assert linked.sum() > seg.sum()
    assert linked[12, 14] and linked[12, 15]  # the gap is bridged


def test_linking_cannot_bridge_gaps_wider_than_k_max():
    """The paper caps the search at k = 7, i.e. a 3-px reach; wider gaps stay open."""
    seg = np.zeros((40, 40), dtype=bool)
    seg[12, 5:14] = True
    seg[12, 24:32] = True  # 10-px gap
    linked, diag = link_edges(seg, np.zeros((40, 40)), ContourConfig())
    assert diag["links"] == 0
    assert not linked[12, 15:24].any()


def test_linking_leaves_a_single_segment_unchanged():
    seg = np.zeros((30, 30), dtype=bool)
    seg[15, 5:20] = True
    linked, diag = link_edges(seg, np.zeros((30, 30)), ContourConfig())
    assert diag["links"] == 0


def test_linking_empty_image_returns_empty():
    linked, diag = link_edges(np.zeros((10, 10), bool), np.zeros((10, 10)), ContourConfig())
    assert linked.sum() == 0
    assert diag["links"] == 0


def test_linking_is_deterministic():
    rng = np.random.default_rng(3)
    gray = rng.integers(0, 255, (60, 60), dtype=np.uint8)
    seg = thin(gray > 200)
    a, _ = link_edges(seg, gradient_direction(gray), ContourConfig())
    b, _ = link_edges(seg, gradient_direction(gray), ContourConfig())
    np.testing.assert_array_equal(a, b)


def test_direction_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        link_edges(np.zeros((5, 5), bool), np.zeros((5, 6)), ContourConfig())


# --------------------------------------------------------------------------- #
# Region labelling / pruning
# --------------------------------------------------------------------------- #
def test_closed_shape_yields_one_enclosed_region():
    """A closed box yields the interior region plus the outside-the-box region.

    The frame-touching region is a legitimate region (``border``); the interior
    is ``closed``.  Both are reported, and each boundary pixel joins the region
    whose seed is nearest.
    """
    wall = np.zeros((60, 60), dtype=bool)
    wall[10, 10:50] = True
    wall[49, 10:50] = True
    wall[10:50, 10] = True
    wall[10:50, 49] = True
    res = label_regions(wall, min_area=1)
    assert res.n_regions == 2
    interior = int(res.labels[30, 30])
    exterior = int(res.labels[0, 0])
    assert interior != exterior
    assert res.statuses[interior] == "closed"
    assert res.statuses[exterior] == "border"
    # interior = 38x38 seed pixels plus its share of the 4-px boundary shell
    assert 38 * 38 <= res.areas[interior] <= 40 * 40


def test_frame_touching_region_is_retained_and_flagged():
    wall = np.zeros((40, 40), dtype=bool)
    wall[10, :] = True
    wall[20, 5:35] = True
    res = label_regions(wall, min_area=1)
    statuses = set(res.statuses[1:])
    assert "border" in statuses


def test_open_curve_does_not_create_a_region_by_itself():
    # An open stroke does not enclose anything, so exactly one region exists
    # (the whole frame-touching surface); no spurious interior region appears.
    wall = np.zeros((30, 30), dtype=bool)
    wall[15, 5:25] = True
    res = label_regions(wall, min_area=1)
    assert res.n_regions == 1
    assert res.statuses[1] == "border"


def test_regions_tile_the_image_without_gaps():
    wall = np.zeros((50, 50), dtype=bool)
    wall[25, :] = True
    wall[:, 30] = True
    res = label_regions(wall, min_area=1)
    assert np.all(res.labels > 0)  # every pixel belongs to exactly one region
    assert res.areas[1:].sum() == wall.size


def test_small_regions_are_discarded():
    # A tiny enclosed pocket between walls is too small to keep.
    wall = np.zeros((60, 60), dtype=bool)
    wall[10:20, 10:20] = True
    wall[12:18, 12:18] = False   # 6x6 = 36 px pocket
    res = label_regions(wall, min_area=500)
    assert res.n_regions == 1                 # only the frame-touching region
    assert res.statuses[1] == "border"
    # With a low threshold the pocket survives.
    assert label_regions(wall, min_area=10).n_regions == 2


def test_labelling_always_tiles_the_image():
    wall = np.zeros((40, 40), dtype=bool)
    wall[5, 5] = True           # isolated speck
    res = label_regions(wall, min_area=500)
    assert np.all(res.labels > 0)
    assert res.areas[1:].sum() == wall.size


def test_labelling_empty_boundary_gives_one_region():
    res = label_regions(np.zeros((20, 20), dtype=bool), min_area=1)
    assert res.n_regions == 1
    assert res.areas[1] == 400


def test_labelling_rejects_bad_input():
    with pytest.raises(ValueError, match="2-D"):
        label_regions(np.zeros((2, 2, 2), bool))
    with pytest.raises(ValueError, match="min_area"):
        label_regions(np.zeros((5, 5), bool), min_area=0)


# --------------------------------------------------------------------------- #
# End-to-end stage
# --------------------------------------------------------------------------- #
def test_initial_contour_stage_returns_consistent_products(synthetic_field):
    cfg = default_config()
    res = detect_initial_contours(synthetic_field, cfg.binarization, cfg.contour)
    assert res.labels.shape == synthetic_field.shape
    assert res.labels.max() == res.n_regions
    assert res.binary.dtype == np.bool_
    assert res.thinned.sum() <= res.binary.sum()
    assert res.diagnostics["threshold"] > 0
    assert res.n_regions >= 1
