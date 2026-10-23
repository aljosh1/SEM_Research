"""Measurement definitions, void counting, calibration and CSV export."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.quantification import (
    DEFAULT_MIN_VOID_AREA,
    area_fractions,
    circularity,
    equivalent_diameter,
    measure_void_components,
    size_distribution,
    summary_statistics,
    void_measurements_dataframe,
)
from src.region_classification import CLASS_MIXED, CLASS_SURFACE, CLASS_VOID


# --------------------------------------------------------------------------- #
# Formulas
# --------------------------------------------------------------------------- #
def test_equivalent_circular_diameter_formula():
    assert equivalent_diameter(0) == 0.0
    # a disc of area pi*r^2 has D_eq = 2r
    for r in (1.0, 5.0, 12.5):
        assert equivalent_diameter(math.pi * r * r) == pytest.approx(2 * r)


def test_equivalent_diameter_rejects_negative():
    with pytest.raises(ValueError):
        equivalent_diameter(-1)


def test_circularity_of_a_disc_is_one():
    # discrete disc: perimeter and area measured on a raster
    n = 200
    yy, xx = np.mgrid[0:n, 0:n]
    r = 40
    disc = ((yy - n / 2) ** 2 + (xx - n / 2) ** 2) <= r * r
    import cv2
    contours, _ = cv2.findContours(disc.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    area = float(disc.sum())
    per = float(sum(cv2.arcLength(c, True) for c in contours))
    assert circularity(area, per) == pytest.approx(1.0, abs=0.12)


def test_circularity_zero_perimeter_returns_zero():
    assert circularity(10.0, 0.0) == 0.0


def test_circularity_rejects_negative():
    with pytest.raises(ValueError):
        circularity(-1, 10)


# --------------------------------------------------------------------------- #
# Void component measurement
# --------------------------------------------------------------------------- #
def test_counts_individual_voids_not_regions():
    """Three separate blobs inside one region are three voids in one region."""
    void_mask = np.zeros((100, 100), dtype=bool)
    void_mask[10:20, 10:20] = True
    void_mask[10:20, 40:50] = True
    void_mask[60:75, 60:80] = True
    labels = np.ones((100, 100), dtype=np.int32)
    voids = measure_void_components(void_mask, labels, "img")
    assert len(voids) == 3
    assert {v.parent_region_id for v in voids} == {1}
    assert [v.void_id for v in voids] == [1, 2, 3]


def test_voids_are_ordered_by_descending_area():
    void_mask = np.zeros((60, 60), dtype=bool)
    void_mask[5:10, 5:10] = True       # 25
    void_mask[20:35, 20:45] = True     # 375
    void_mask[45:50, 45:55] = True     # 50
    labels = np.ones((60, 60), dtype=np.int32)
    voids = measure_void_components(void_mask, labels, "img")
    areas = [v.pixel_area for v in voids]
    assert areas == sorted(areas, reverse=True)
    assert voids[0].void_id == 1


def test_small_speckle_is_excluded():
    void_mask = np.zeros((40, 40), dtype=bool)
    void_mask[5, 5] = True             # 1 px
    void_mask[20:25, 20:25] = True     # 25 px
    labels = np.ones((40, 40), dtype=np.int32)
    voids = measure_void_components(void_mask, labels, "img", min_area=10)
    assert len(voids) == 1
    assert voids[0].pixel_area == 25


def test_void_is_attributed_to_its_parent_region():
    void_mask = np.zeros((40, 80), dtype=bool)
    void_mask[10:20, 50:60] = True
    labels = np.ones((40, 80), dtype=np.int32)
    labels[:, 40:] = 2
    voids = measure_void_components(void_mask, labels, "img")
    assert len(voids) == 1
    assert voids[0].parent_region_id == 2


def test_void_touching_border_is_flagged():
    void_mask = np.zeros((40, 40), dtype=bool)
    void_mask[0:10, 0:10] = True
    labels = np.ones((40, 40), dtype=np.int32)
    voids = measure_void_components(void_mask, labels, "img")
    assert voids[0].touches_border


def test_empty_void_mask_gives_no_voids():
    labels = np.ones((20, 20), dtype=np.int32)
    assert measure_void_components(np.zeros((20, 20), bool), labels, "img") == []


def test_aspect_ratio_is_major_over_minor():
    void_mask = np.zeros((60, 60), dtype=bool)
    void_mask[20:25, 10:50] = True     # 5 x 40 bar
    labels = np.ones((60, 60), dtype=np.int32)
    v = measure_void_components(void_mask, labels, "img")[0]
    assert v.major_axis_px >= v.minor_axis_px
    assert v.aspect_ratio == pytest.approx(v.major_axis_px / v.minor_axis_px)
    assert v.aspect_ratio > 5


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        measure_void_components(np.zeros((5, 5), bool), np.zeros((4, 4), np.int32), "x")


def test_min_area_validation():
    with pytest.raises(ValueError):
        measure_void_components(np.zeros((5, 5), bool), np.ones((5, 5), np.int32), "x",
                                min_area=0)


# --------------------------------------------------------------------------- #
# Area fractions
# --------------------------------------------------------------------------- #
def test_area_fractions_sum_to_one():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[:5, :] = CLASS_VOID
    img[5:8, :] = CLASS_MIXED
    img[8:, :] = CLASS_SURFACE
    f = area_fractions(img)
    assert f["VOID"] == pytest.approx(0.5)
    assert f["MIXED"] == pytest.approx(0.3)
    assert f["SURFACE"] == pytest.approx(0.2)
    assert sum(f.values()) == pytest.approx(1.0)


def test_area_fractions_reject_empty():
    with pytest.raises(ValueError):
        area_fractions(np.zeros((0, 0), np.uint8))


# --------------------------------------------------------------------------- #
# Summary + calibration
# --------------------------------------------------------------------------- #
def _voids():
    void_mask = np.zeros((50, 50), dtype=bool)
    void_mask[5:15, 5:15] = True
    void_mask[25:30, 25:35] = True
    labels = np.ones((50, 50), dtype=np.int32)
    return measure_void_components(void_mask, labels, "img")


def test_summary_reports_pixel_units_and_marks_calibration_absent():
    cls_img = np.zeros((50, 50), np.uint8)
    cls_img[:20] = CLASS_VOID
    cls_img[20:30] = CLASS_MIXED
    cls_img[30:] = CLASS_SURFACE
    s = summary_statistics(_voids(), cls_img, None)
    assert s["calibration_available"] is False
    assert s["void_number_density_per_mm2"] is None
    assert s["total_void_area_px"] == 100 + 50
    assert s["void_area_fraction"] == pytest.approx(20 / 50)
    assert "equivalent_diameter_mean_px" in s


def test_calibration_converts_to_physical_units():
    cls_img = np.zeros((50, 50), np.uint8)
    cls_img[:20] = CLASS_VOID
    px = 0.5  # um per pixel
    s = summary_statistics(_voids(), cls_img, px)
    assert s["calibration_available"] is True
    assert s["total_void_area_um2"] == pytest.approx((100 + 50) * px * px)
    assert s["void_number_density_per_mm2"] == pytest.approx(2 / (2500 * px * px / 1e6))
    assert s["equivalent_diameter_um_mean_um"] > 0


def test_summary_handles_no_voids():
    cls_img = np.full((20, 20), CLASS_SURFACE, np.uint8)
    s = summary_statistics([], cls_img, None)
    assert s["n_individual_voids"] == 0
    assert s["total_void_area_px"] == 0
    assert s["equivalent_diameter_mean_px"] is None


# --------------------------------------------------------------------------- #
# CSV export
# --------------------------------------------------------------------------- #
def test_void_dataframe_has_required_columns():
    df = void_measurements_dataframe(_voids(), None)
    for col in ("image_id", "void_id", "parent_region_id", "pixel_area",
                "equivalent_diameter_px", "major_axis_px", "minor_axis_px",
                "aspect_ratio", "perimeter_px", "circularity",
                "physical_area_um2", "equivalent_diameter_um"):
        assert col in df.columns
    assert len(df) == 2
    assert df["physical_area_um2"].isna().all()


def test_void_dataframe_fills_physical_columns_when_calibrated():
    df = void_measurements_dataframe(_voids(), 0.25)
    assert df["physical_area_um2"].notna().all()
    assert df["calibration_available"].all()
    # 0.25 um/px -> 100 px void = 100 * 0.0625 = 6.25 um^2
    assert df["physical_area_um2"].iloc[0] == pytest.approx(100 * 0.25 ** 2)


def test_empty_void_dataframe_keeps_schema():
    df = void_measurements_dataframe([], None)
    assert len(df) == 0
    assert "void_id" in df.columns


def test_dataframe_round_trips_through_csv(tmp_path):
    df = void_measurements_dataframe(_voids(), None)
    p = tmp_path / "v.csv"
    df.to_csv(p, index=False)
    back = pd.read_csv(p)
    assert list(back.columns) == list(df.columns)
    assert len(back) == len(df)


def test_size_distribution_returns_histogram():
    counts, edges, centres = size_distribution(_voids())
    assert counts.sum() == 2
    assert len(centres) == len(counts)


def test_size_distribution_with_few_voids_is_empty():
    counts, edges, centres = size_distribution(_voids()[:1])
    assert counts.size == 0
