"""Evaluation metrics: formulas, degenerate cases and annotation handling."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from src.evaluation import (
    AnnotationError,
    aggregate,
    binary_metrics,
    evaluate_run,
    find_annotation,
    per_class_metrics,
    results_to_frame,
)


# --------------------------------------------------------------------------- #
# Formula correctness on hand-computable masks
# --------------------------------------------------------------------------- #
def test_perfect_prediction_scores_one():
    gt = np.zeros((10, 10), bool)
    gt[2:5, 2:5] = True
    m = binary_metrics(gt.copy(), gt.copy())
    assert (m.tp, m.fp, m.fn, m.tn) == (9, 0, 0, 91)
    assert m.iou == 1.0
    assert m.dice == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0


def test_counts_and_derived_scores():
    # gt = 2x5 block (10 px); pred covers 6 of them plus 2 elsewhere
    gt = np.zeros((10, 10), bool)
    gt[0:2, 0:5] = True
    pred = np.zeros((10, 10), bool)
    pred[0:2, 0:3] = True      # TP = 6
    pred[8, 8:10] = True       # FP = 2
    m = binary_metrics(pred, gt)
    assert (m.tp, m.fp, m.fn, m.tn) == (6, 2, 4, 88)
    assert m.iou == pytest.approx(6 / 12)
    assert m.dice == pytest.approx(12 / 18)
    assert m.precision == pytest.approx(6 / 8)
    assert m.recall == pytest.approx(6 / 10)


def test_no_overlap_gives_zero_scores():
    gt = np.zeros((8, 8), bool)
    gt[0, 0] = True
    pred = np.zeros((8, 8), bool)
    pred[7, 7] = True
    m = binary_metrics(pred, gt)
    assert m.iou == 0.0
    assert m.dice == 0.0
    assert m.precision == 0.0
    assert m.recall == 0.0


def test_empty_prediction_and_empty_annotation_yields_undefined_not_one():
    """Both masks empty: the scores cannot be computed, so they must be None."""
    empty = np.zeros((6, 6), bool)
    m = binary_metrics(empty, empty)
    assert (m.tp, m.fp, m.fn, m.tn) == (0, 0, 0, 36)
    assert m.iou is None
    assert m.dice is None
    assert m.precision is None
    assert m.recall is None


def test_prediction_empty_but_annotation_not_gives_zero_recall_not_crash():
    gt = np.zeros((6, 6), bool)
    gt[1:3, 1:3] = True
    m = binary_metrics(np.zeros((6, 6), bool), gt)
    assert m.tp == 0 and m.fp == 0 and m.fn == 4
    assert m.precision is None      # no predicted positives: undefined
    assert m.recall == 0.0
    assert m.iou == 0.0


def test_shape_mismatch_raises():
    with pytest.raises(AnnotationError, match="Shape mismatch"):
        binary_metrics(np.zeros((5, 5), bool), np.zeros((5, 6), bool))


def test_non_boolean_input_is_treated_as_nonzero():
    pred = np.array([[0, 255], [0, 0]], np.uint8)
    gt = np.array([[0, 3], [0, 0]], np.uint8)
    m = binary_metrics(pred, gt)
    assert m.tp == 1


# --------------------------------------------------------------------------- #
# Per-class (one-vs-rest) scoring
# --------------------------------------------------------------------------- #
def test_per_class_metrics_excludes_ignore_label():
    gt = np.array([[1, 1, 2], [3, 3, 0]], np.uint8)   # one pixel ignored
    pred = np.array([[1, 2, 2], [3, 1, 3]], np.uint8)
    m = per_class_metrics(pred, gt)
    assert set(m) == {"SURFACE", "MIXED", "VOID"}
    # SURFACE: gt has 2 px (row0 c0,c1); pred has row0c0 and row1c1 -> tp=1
    assert m["SURFACE"].tp == 1
    assert m["SURFACE"].fp == 1
    assert m["SURFACE"].fn == 1
    # VOID: gt has 2 px (row1 c3,c4); pred has row1 c3 and row2 c5 -> tp=1, fn=1
    assert m["VOID"].tp == 1
    assert m["VOID"].fn == 1
    # the ignored pixel (gt 0, pred 3) is excluded from every comparison
    assert m["VOID"].fp == 0


def test_per_class_shape_mismatch_raises():
    with pytest.raises(AnnotationError):
        per_class_metrics(np.zeros((4, 4), np.uint8), np.zeros((3, 3), np.uint8))


# --------------------------------------------------------------------------- #
# Annotation discovery and end-to-end scoring
# --------------------------------------------------------------------------- #
def _write_grey(path, arr):
    cv2.imwrite(str(path), arr.astype(np.uint8))


def test_find_annotation_returns_none_when_absent(tmp_path):
    assert find_annotation("foo.tif", tmp_path) is None


def test_find_annotation_prefers_void_mask(tmp_path):
    _write_grey(tmp_path / "foo_void_mask.png", np.zeros((4, 4)))
    _write_grey(tmp_path / "foo_classes.png", np.zeros((4, 4)))
    path, kind = find_annotation("foo.tif", tmp_path)
    assert path.name == "foo_void_mask.png"
    assert kind == "void_mask"


def _make_run(tmp_path, void_mask, class_img, image_name="foo.tif"):
    run = tmp_path / "run"
    (run / "masks").mkdir(parents=True)
    cv2.imwrite(str(run / "masks" / "02_extended_void_mask.png"), void_mask.astype(np.uint8))
    cv2.imwrite(str(run / "masks" / "03_class_image.png"), class_img.astype(np.uint8))
    (run / "run_metadata.json").write_text(
        json.dumps({"image": {"path": str(tmp_path / image_name)}}), encoding="utf-8"
    )
    return run


def test_evaluate_run_without_annotation_reports_status(tmp_path):
    void = np.zeros((10, 10), np.uint8)
    run = _make_run(tmp_path, void * 255, void)
    res = evaluate_run(run, tmp_path / "foo.tif", tmp_path / "annot")
    assert res["status"] == "no_annotation"
    assert "message" in res


def test_evaluate_run_with_void_mask_computes_scores(tmp_path):
    gt = np.zeros((10, 10), np.uint8)
    gt[2:5, 2:5] = 255
    pred = gt.copy()
    class_img = np.where(gt > 0, 3, 1).astype(np.uint8)
    run = _make_run(tmp_path, pred, class_img)
    ann = tmp_path / "annot"
    ann.mkdir()
    _write_grey(ann / "foo_void_mask.png", gt)
    res = evaluate_run(run, tmp_path / "foo.tif", ann)
    assert res["status"] == "ok"
    assert res["void_mask"]["iou"] == 1.0
    assert res["counts_check"]["annotated_void_px"] == 9


def test_evaluate_run_with_class_map_computes_per_class(tmp_path):
    gt = np.array([[1, 1], [3, 3]], np.uint8)
    class_img = gt.copy()
    run = _make_run(tmp_path, (gt == 3).astype(np.uint8) * 255, class_img)
    ann = tmp_path / "annot"
    ann.mkdir()
    _write_grey(ann / "foo_classes.png", gt)
    res = evaluate_run(run, tmp_path / "foo.tif", ann)
    assert res["status"] == "ok"
    assert res["annotation_kind"] == "class_map"
    assert res["per_class"]["VOID"]["iou"] == 1.0
    assert res["void_mask"]["iou"] == 1.0


def test_class_annotation_with_bad_labels_raises(tmp_path):
    gt = np.array([[1, 9], [3, 3]], np.uint8)
    run = _make_run(tmp_path, np.zeros((2, 2), np.uint8), gt)
    ann = tmp_path / "annot"
    ann.mkdir()
    _write_grey(ann / "foo_classes.png", gt)
    with pytest.raises(AnnotationError, match="unexpected labels"):
        evaluate_run(run, tmp_path / "foo.tif", ann)


def test_missing_run_mask_raises(tmp_path):
    ann = tmp_path / "annot"
    ann.mkdir()
    _write_grey(ann / "foo_void_mask.png", np.zeros((4, 4)))
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(AnnotationError, match="Could not read"):
        evaluate_run(run, tmp_path / "foo.tif", ann)


# --------------------------------------------------------------------------- #
# Aggregation / reporting
# --------------------------------------------------------------------------- #
def test_results_to_frame_marks_unscored_images():
    df = results_to_frame([
        {"image": "a", "status": "ok", "annotation_kind": "void_mask",
         "void_mask": {"tp": 1, "fp": 0, "fn": 0, "tn": 9, "iou": 1.0,
                       "dice": 1.0, "precision": 1.0, "recall": 1.0}},
        {"image": "b", "status": "no_annotation"},
    ])
    assert len(df) == 2
    assert df.loc[df.image == "b", "status"].iloc[0] == "no_annotation"


def test_aggregate_counts_only_scored_images():
    results = [
        {"image": "a", "status": "ok", "void_mask": {"iou": 0.5, "dice": 0.6,
                                                     "precision": 0.7, "recall": 0.8}},
        {"image": "b", "status": "ok", "void_mask": {"iou": 0.7, "dice": 0.8,
                                                     "precision": 0.9, "recall": 0.6}},
        {"image": "c", "status": "no_annotation"},
    ]
    agg = aggregate(results)
    assert agg["n_images"] == 2
    assert agg["n_images_without_annotation"] == 1
    assert agg["void_iou_mean"] == pytest.approx(0.6)
    assert agg["void_iou_n"] == 2


def test_aggregate_with_no_scored_images_reports_none():
    agg = aggregate([{"image": "a", "status": "no_annotation"}])
    assert agg["n_images"] == 0
    assert agg["void_iou_mean"] is None
