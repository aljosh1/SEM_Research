"""Evaluation against manual (ground-truth) annotations.

The reference material contains no annotated masks, so nothing is scored by
default.  This module exists so that validation becomes a single command once
annotations are supplied, and so the definitions are fixed in code rather than
worked out ad hoc afterwards.

Accepted annotation formats
---------------------------
1. **Binary void mask** - a 2-D image where non-zero (or white) means "void".
2. **Class index map** - a 2-D image whose pixel values are the class ids used by
   the pipeline: 0 = unlabelled/boundary, 1 = SURFACE, 2 = MIXED, 3 = VOID.
   Stored as ``uint8`` PNG/TIFF.  A legend file is optional.

File lookup
-----------
Given an input image ``data/sample/foo.tif`` and ``--annotations-dir
data/annotations``, the evaluator looks for, in order::

    data/annotations/foo.tif_void_mask.png
    data/annotations/foo_void_mask.png
    data/annotations/foo_void_mask.tif
    data/annotations/foo_classes.png

Definitions (from the project brief)::

    TP = |pred AND gt|            FP = |pred AND NOT gt|
    FN = |NOT pred AND gt|        TN = |NOT pred AND NOT gt|

    IoU       = TP / (TP + FP + FN)
    Dice      = 2TP / (2TP + FP + FN)
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)

Degenerate cases return ``None`` rather than 0 or 1, because a metric that cannot
be computed must not be silently reported as a perfect or zero score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .region_classification import CLASS_MIXED, CLASS_NAMES, CLASS_SURFACE, CLASS_VOID

LOGGER = logging.getLogger(__name__)

#: Suffixes searched for an annotation, in priority order.
VOID_MASK_SUFFIXES = ("_void_mask.png", "_void_mask.tif", "_void_mask.tiff")
CLASS_MAP_SUFFIXES = ("_classes.png", "_classes.tif", "_classes.tiff")


class AnnotationError(ValueError):
    """Raised when an annotation file is missing, malformed or mismatched."""


@dataclass
class BinaryMetrics:
    """Confusion counts and derived scores for a binary mask comparison."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def iou(self) -> float | None:
        denom = self.tp + self.fp + self.fn
        return None if denom == 0 else self.tp / denom

    @property
    def dice(self) -> float | None:
        denom = 2 * self.tp + self.fp + self.fn
        return None if denom == 0 else 2 * self.tp / denom

    @property
    def precision(self) -> float | None:
        denom = self.tp + self.fp
        return None if denom == 0 else self.tp / denom

    @property
    def recall(self) -> float | None:
        denom = self.tp + self.fn
        return None if denom == 0 else self.tp / denom

    def as_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "iou": self.iou, "dice": self.dice,
            "precision": self.precision, "recall": self.recall,
        }


def _as_binary(arr: np.ndarray) -> np.ndarray:
    """Treat any non-zero value as foreground."""
    return arr.astype(bool)


def _load_annotation(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise AnnotationError(f"Could not read annotation file: {path}")
    if img.ndim == 3:
        # colour annotations: treat any non-black pixel as foreground
        img = img.max(axis=2)
    return img


def find_annotation(
    image_path: str | Path, annotations_dir: str | Path
) -> tuple[Path, str] | None:
    """Locate the annotation for an image.

    Returns
    -------
    (path, kind) or None
        ``kind`` is ``"void_mask"`` or ``"class_map"``.
    """
    stem = Path(image_path).stem
    d = Path(annotations_dir)
    for suffix in VOID_MASK_SUFFIXES:
        p = d / f"{stem}{suffix}"
        if p.exists():
            return p, "void_mask"
    for suffix in CLASS_MAP_SUFFIXES:
        p = d / f"{stem}{suffix}"
        if p.exists():
            return p, "class_map"
    return None


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> BinaryMetrics:
    """Confusion counts for two boolean masks of identical shape."""
    if pred.shape != gt.shape:
        raise AnnotationError(
            f"Shape mismatch: prediction {pred.shape} vs annotation {gt.shape}. "
            "Predicted and annotated masks must cover the same pixels."
        )
    p = pred.astype(bool)
    g = gt.astype(bool)
    return BinaryMetrics(
        tp=int(np.count_nonzero(p & g)),
        fp=int(np.count_nonzero(p & ~g)),
        fn=int(np.count_nonzero(~p & g)),
        tn=int(np.count_nonzero(~p & ~g)),
    )


def per_class_metrics(
    pred_classes: np.ndarray, gt_classes: np.ndarray, ignore_label: int = 0
) -> dict[str, BinaryMetrics]:
    """One-vs-rest confusion counts for each of the three classes.

    Pixels whose annotation equals ``ignore_label`` (unlabelled/boundary) are
    excluded from every class comparison, since they carry no ground truth.
    """
    if pred_classes.shape != gt_classes.shape:
        raise AnnotationError(
            f"Shape mismatch: prediction {pred_classes.shape} vs annotation "
            f"{gt_classes.shape}"
        )
    valid = gt_classes != ignore_label
    out: dict[str, BinaryMetrics] = {}
    for cls_id, name in ((CLASS_SURFACE, "SURFACE"), (CLASS_MIXED, "MIXED"),
                         (CLASS_VOID, "VOID")):
        gt_c = (gt_classes == cls_id) & valid
        pr_c = (pred_classes == cls_id) & valid
        out[name] = binary_metrics(pr_c, gt_c)
    return out


def evaluate_run(
    out_dir: str | Path,
    image_path: str | Path,
    annotations_dir: str | Path,
) -> dict[str, Any]:
    """Score one pipeline run against its annotation.

    The prediction is read from the run directory written by
    :func:`src.pipeline.run_pipeline`:

    * VOID comparison uses ``masks/02_extended_void_mask.png``.
    * Per-class comparison uses ``masks/03_class_image.png``.

    Parameters
    ----------
    out_dir:
        Directory produced by a pipeline run.
    image_path:
        The original image, used to find the matching annotation.
    annotations_dir:
        Folder containing annotation files.

    Returns
    -------
    dict
        With ``status`` (``"ok"`` / ``"no_annotation"``), the void-mask scores,
        per-class scores and metadata describing what was compared.

    Raises
    ------
    AnnotationError
        If an annotation exists but cannot be interpreted.
    """
    out = Path(out_dir)
    found = find_annotation(image_path, annotations_dir)
    if found is None:
        return {
            "status": "no_annotation",
            "image": Path(image_path).name,
            "message": (
                "No annotation found. Expected one of "
                f"{[Path(image_path).stem + s for s in VOID_MASK_SUFFIXES[:-1]]} "
                f"in {annotations_dir}"
            ),
        }

    ann_path, kind = found
    ann = _load_annotation(ann_path)
    result: dict[str, Any] = {
        "status": "ok",
        "image": Path(image_path).name,
        "annotation": str(ann_path),
        "annotation_kind": kind,
    }

    if kind == "void_mask":
        pred_path = out / "masks" / "02_extended_void_mask.png"
        pred = _load_annotation(pred_path)
        if pred.shape != ann.shape:
            h, w = ann.shape[:2]
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
            result["resized_prediction"] = True
        m = binary_metrics(_as_binary(pred), _as_binary(ann))
        result["void_mask"] = m.as_dict()
        result["counts_check"] = {
            "predicted_void_px": int(_as_binary(pred).sum()),
            "annotated_void_px": int(_as_binary(ann).sum()),
        }
    else:
        pred_path = out / "masks" / "03_class_image.png"
        pred = _load_annotation(pred_path).astype(np.uint8)
        gt = ann.astype(np.uint8)
        if pred.shape != gt.shape:
            h, w = gt.shape[:2]
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
            result["resized_prediction"] = True
        bad = set(np.unique(gt).tolist()) - {0, 1, 2, 3}
        if bad:
            raise AnnotationError(
                f"Class annotation contains unexpected labels {sorted(bad)}; "
                "expected 0 (ignore), 1 SURFACE, 2 MIXED, 3 VOID."
            )
        pc = per_class_metrics(pred, gt)
        result["per_class"] = {k: v.as_dict() for k, v in pc.items()}
        # VOID is available from the class map too, using the same pixels.
        result["void_mask"] = pc["VOID"].as_dict()

    return result


def results_to_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten evaluation results into one CSV row per image."""
    rows = []
    for r in results:
        if r.get("status") != "ok":
            rows.append({"image": r["image"], "status": "no_annotation"})
            continue
        void = r.get("void_mask", {})
        row = {
            "image": r["image"],
            "status": "ok",
            "annotation_kind": r["annotation_kind"],
            "void_tp": void.get("tp"),
            "void_fp": void.get("fp"),
            "void_fn": void.get("fn"),
            "void_tn": void.get("tn"),
            "void_iou": void.get("iou"),
            "void_dice": void.get("dice"),
            "void_precision": void.get("precision"),
            "void_recall": void.get("recall"),
        }
        for cls in ("SURFACE", "MIXED"):
            c = r.get("per_class", {}).get(cls)
            if c:
                row[f"{cls.lower()}_iou"] = c["iou"]
                row[f"{cls.lower()}_dice"] = c["dice"]
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and count of each metric over the scored images.

    Reports ``n_images`` so a mean over three images cannot be mistaken for one
    over thirty.  Metrics that were ``None`` (undefined) are excluded from the
    mean and counted separately.
    """
    scored = [r for r in results if r.get("status") == "ok"]
    out: dict[str, Any] = {"n_images": len(scored),
                           "n_images_without_annotation": len(results) - len(scored)}
    for key in ("iou", "dice", "precision", "recall"):
        vals = [r["void_mask"][key] for r in scored if r["void_mask"].get(key) is not None]
        out[f"void_{key}_mean"] = float(np.mean(vals)) if vals else None
        out[f"void_{key}_n"] = len(vals)
    return out
