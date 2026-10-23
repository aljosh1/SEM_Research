"""Integration tests: the complete pipeline, its outputs and reproducibility."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import default_config
from src.pipeline import OUTPUT_DIRS, run_pipeline
from src.region_classification import CLASS_MIXED, CLASS_SURFACE, CLASS_VOID


@pytest.fixture(scope="module")
def run_outputs(tmp_path_factory):
    """Run the pipeline once on a synthetic field; reuse across tests."""
    import cv2

    h, w = 240, 320
    img = np.full((h, w), 40, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (140, 110), 80, -1)
    cv2.rectangle(img, (180, 130), (300, 220), 70, -1)
    cv2.rectangle(img, (20, 20), (140, 110), 200, 2)
    cv2.rectangle(img, (180, 130), (300, 220), 200, 2)
    cv2.circle(img, (80, 170), 30, 10, -1)
    cv2.circle(img, (80, 170), 30, 210, 2)
    rng = np.random.default_rng(0)
    img = np.clip(img.astype(int) + rng.integers(-4, 5, size=img.shape), 0, 255).astype(np.uint8)

    out = tmp_path_factory.mktemp("run")
    src_img = out / "input.png"
    cv2.imwrite(str(src_img), img)
    cfg = default_config()
    result = run_pipeline(src_img, out / "out", cfg, image_id="synthetic")
    return result, out / "out", src_img, cfg


def test_all_output_directories_exist(run_outputs):
    _, out, _, _ = run_outputs
    for sub in OUTPUT_DIRS:
        assert (out / sub).is_dir(), sub


def test_required_files_are_written(run_outputs):
    _, out, _, _ = run_outputs
    expected = [
        "contours/01_binary.png", "contours/02_thinned.png", "contours/03_linked.png",
        "contours/04_region_labels.png", "contours/05_refined_region_labels.png",
        "masks/01_void_tip_mask.png", "masks/02_extended_void_mask.png",
        "masks/03_class_image.png", "masks/04_class_overlay.png",
        "measurements/void_measurements.csv",
        "measurements/region_measurements.csv",
        "measurements/summary_statistics.csv",
        "figures/01_original.png", "figures/07_final_segmentation.png",
        "figures/08_side_by_side.png", "figures/00_composite_overview.png",
        "run_metadata.json",
    ]
    for rel in expected:
        p = out / rel
        assert p.exists(), f"missing {rel}"
        assert p.stat().st_size > 0, f"empty {rel}"


def test_class_image_has_valid_ids_only(run_outputs):
    result = run_outputs[0]
    ids = set(np.unique(result.class_image).tolist())
    assert ids.issubset({0, CLASS_SURFACE, CLASS_MIXED, CLASS_VOID})
    assert CLASS_SURFACE in ids or CLASS_MIXED in ids or CLASS_VOID in ids


def test_masks_are_binary_after_round_trip(run_outputs):
    import cv2
    _, out, _, _ = run_outputs
    for name in ("01_void_tip_mask.png", "02_extended_void_mask.png", "01_binary.png"):
        m = cv2.imread(str(out / "masks" / name) if "void" in name or "extended" in name
                       else str(out / "contours" / name), cv2.IMREAD_GRAYSCALE)
        assert m is not None
        assert set(np.unique(m).tolist()).issubset({0, 255}), name


def test_region_csv_matches_image(run_outputs):
    result, out, _, _ = run_outputs
    df = pd.read_csv(out / "measurements" / "region_measurements.csv")
    assert len(df) == int(result.region_labels.max())
    assert df["region_area_px"].sum() == result.class_image.size
    assert set(df["region_class"]).issubset({"VOID", "SURFACE", "MIXED"})


def test_void_csv_consistent_with_mask(run_outputs):
    result, out, _, _ = run_outputs
    df = pd.read_csv(out / "measurements" / "void_measurements.csv")
    if len(df):
        # individual voids may be smaller than the mask (10-px floor)
        assert df["pixel_area"].sum() <= int(result.void_mask.sum())
        assert df["void_id"].tolist() == list(range(1, len(df) + 1))


def test_area_fractions_are_consistent(run_outputs):
    result, out, _, _ = run_outputs
    summary = pd.read_csv(out / "measurements" / "summary_statistics.csv")
    vals = dict(zip(summary["metric"], summary["value"]))
    frac = (vals["void_area_fraction"] + vals["surface_area_fraction"]
            + vals["mixed_area_fraction"])
    assert frac == pytest.approx(1.0, abs=1e-6)


def test_summary_csv_has_explicit_units(run_outputs):
    _, out, _, _ = run_outputs
    s = pd.read_csv(out / "measurements" / "summary_statistics.csv")
    assert set(s.columns) == {"metric", "value", "unit"}
    assert s["unit"].notna().all()


def test_metadata_is_complete(run_outputs):
    _, out, _, _ = run_outputs
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    for key in ("image", "parameters", "thresholds", "stages", "calibration",
                "warnings", "output_files", "processing"):
        assert key in meta
    assert meta["processing"]["outputs_complete"] is True
    assert meta["calibration"]["physical_measurements_available"] is False
    assert meta["parameters"]["binarization"]["sigma_multiplier"] == 2.0
    # every listed output must exist
    for f in meta["output_files"]:
        assert Path(f).exists(), f


def test_run_is_reproducible(tmp_path, run_outputs):
    """Same input + same parameters -> identical masks and CSV values."""
    result, out, src_img, cfg = run_outputs
    a = run_pipeline(src_img, tmp_path / "a", cfg, image_id="synthetic", make_figures=False)
    b = run_pipeline(src_img, tmp_path / "b", cfg, image_id="synthetic", make_figures=False)
    np.testing.assert_array_equal(a.class_image, b.class_image)
    np.testing.assert_array_equal(a.void_mask, b.void_mask)
    np.testing.assert_array_equal(a.region_labels, b.region_labels)
    for name in ("summary_statistics.csv", "void_measurements.csv",
                 "region_measurements.csv"):
        da = pd.read_csv(a.out_dir / "measurements" / name)
        db = pd.read_csv(b.out_dir / "measurements" / name)
        pd.testing.assert_frame_equal(da, db)


def test_calibration_flows_through_to_csv(tmp_path, run_outputs):
    result, out, src_img, cfg = run_outputs
    cfg2 = default_config()
    cfg2.calibration.pixel_size_um = 0.4
    cfg2.calibration.source = "unit test"
    res = run_pipeline(src_img, tmp_path / "cal", cfg2, image_id="cal", make_figures=False)
    df = pd.read_csv(res.out_dir / "measurements" / "void_measurements.csv")
    if len(df):
        assert df["calibration_available"].all()
        assert df["physical_area_um2"].notna().all()
    meta = json.loads((res.out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["calibration"]["pixel_size_um"] == 0.4
    assert meta["calibration"]["source"] == "unit test"


def test_no_figures_option_skips_figure_files(tmp_path, run_outputs):
    result, out, src_img, cfg = run_outputs
    res = run_pipeline(src_img, tmp_path / "nofig", cfg, make_figures=False)
    figs = list((res.out_dir / "figures").glob("*.png"))
    assert len(figs) <= 3          # only 01_original and 02_grayscale


# --------------------------------------------------------------------------- #
# Real SEM data
# --------------------------------------------------------------------------- #
def test_real_sem_image_runs_end_to_end(tmp_path, real_image_path):
    cfg = default_config()
    res = run_pipeline(real_image_path, tmp_path / "real", cfg,
                       image_id="real", make_figures=False)
    assert res.class_image.shape == res.void_mask.shape
    assert res.class_image.size > 0
    meta = json.loads((res.out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["image"]["dimensions"]["height"] > 100
    assert meta["thresholds"]["global_th"] > 0
    # measurement files must be readable and non-empty
    for name in ("void_measurements.csv", "region_measurements.csv",
                 "summary_statistics.csv"):
        df = pd.read_csv(res.out_dir / "measurements" / name)
        assert len(df) > 0, name
