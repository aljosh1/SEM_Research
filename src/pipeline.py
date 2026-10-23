"""End-to-end pipeline: image -> segmentation -> measurements -> outputs.

The orchestrator keeps the scientific stages independent of file export: each
stage module returns data structures, and this module decides what to write.
Every artefact written is recorded in ``run_metadata.json`` together with the
parameters that produced it, so a run can always be reproduced or audited.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from . import __version__
from .config import PARAMETER_DOCS, SegmentationConfig
from .contour_detection import detect_initial_contours, global_threshold, binned_histogram
from .contour_refinement import refine_contours
from .image_io import ImageRecord, load_image
from .preprocessing import preprocess
from .quantification import (
    area_fractions,
    measure_void_components,
    size_distribution,
    summary_statistics,
    void_measurements_dataframe,
)
from .region_classification import CLASS_NAMES, classify_regions
from .void_detection import detect_voids, detect_void_tips
from . import visualization as viz

LOGGER = logging.getLogger(__name__)

#: Output sub-directories created for every run.
OUTPUT_DIRS = ("contours", "masks", "figures", "measurements")


@dataclass
class PipelineOutputs:
    """Handles to every product of a run."""

    out_dir: Path
    image: ImageRecord
    class_image: np.ndarray
    void_mask: np.ndarray
    region_labels: np.ndarray
    summary: dict[str, Any]
    files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _write_mask(path: Path, mask: np.ndarray) -> None:
    """Save a binary mask as 0/255 uint8 PNG plus an optional TIFF copy."""
    cv2.imwrite(str(path), (mask.astype(bool).astype(np.uint8) * 255))
    cv2.imwrite(str(path.with_suffix(".tiff")), (mask.astype(bool).astype(np.uint8) * 255))


def run_pipeline(
    image_path: str | Path,
    out_dir: str | Path,
    cfg: SegmentationConfig,
    image_id: str | None = None,
    export_tiff: bool = True,
    make_figures: bool = True,
) -> PipelineOutputs:
    """Run all stages and write every artefact.

    Parameters
    ----------
    image_path:
        Input image (PNG / TIFF / JPEG).
    out_dir:
        Root output directory for this run.
    cfg:
        Validated configuration.
    image_id:
        Identifier used in CSV rows; defaults to the file stem.
    export_tiff:
        Also write TIFF copies of masks and the segmentation.
    make_figures:
        Generate the figure set (disable for batch runs).

    Returns
    -------
    PipelineOutputs
        With the in-memory products and the list of written files.
    """
    cfg.validate()
    started = _now()
    t_start = time.perf_counter()
    out = Path(out_dir)
    for sub in OUTPUT_DIRS:
        (out / sub).mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    warnings: list[str] = []

    # ---- Stage 1: load -----------------------------------------------------
    rec = load_image(image_path)
    image_id = image_id or rec.path.stem
    cv2.imwrite(str(out / "figures" / "01_original.png"), rec.gray)
    files.append(str(out / "figures" / "01_original.png"))

    # ---- Stage 2: optional preprocessing -----------------------------------
    pre = preprocess(rec.gray, cfg.preprocessing)
    gray = pre.image
    if pre.applied:
        cv2.imwrite(str(out / "figures" / "02_preprocessed.png"), gray)
        files.append(str(out / "figures" / "02_preprocessed.png"))
    else:
        cv2.imwrite(str(out / "figures" / "02_grayscale.png"), gray)
        files.append(str(out / "figures" / "02_grayscale.png"))

    hist_counts, hist_centres = binned_histogram(gray, 1)
    thr = global_threshold(gray, cfg.binarization)

    # ---- Stage 3: initial contour ------------------------------------------
    initial = detect_initial_contours(gray, cfg.binarization, cfg.contour)
    _write_mask(out / "contours" / "01_binary.png", initial.binary)
    _write_mask(out / "contours" / "02_thinned.png", initial.thinned)
    _write_mask(out / "contours" / "03_linked.png", initial.linked)
    cv2.imwrite(str(out / "contours" / "04_region_labels.png"),
                (initial.labels % 256).astype(np.uint8))
    for f in ("01_binary.png", "02_thinned.png", "03_linked.png", "04_region_labels.png"):
        files.append(str(out / "contours" / f))

    # ---- Stage 4: refinement ------------------------------------------------
    refined = refine_contours(
        gray, initial.labels, initial.areas, initial.statuses, cfg.contour
    )
    cv2.imwrite(str(out / "contours" / "05_refined_region_labels.png"),
                (refined.labels % 256).astype(np.uint8))
    files.append(str(out / "contours" / "05_refined_region_labels.png"))

    # ---- Stage 5: void detection -------------------------------------------
    tip = detect_void_tips(gray, cfg.void)
    if tip.threshold is None:
        warnings.append(f"Void-tip detection failed: {tip.status}. "
                        "No VOID regions can be assigned for this image.")
        tip.tip_mask = np.zeros(gray.shape, dtype=bool)
    ext = detect_voids(gray, refined.labels, cfg.void)[1]
    _write_mask(out / "masks" / "01_void_tip_mask.png", tip.tip_mask)
    _write_mask(out / "masks" / "02_extended_void_mask.png", ext.void_mask)
    files += [str(out / "masks" / "01_void_tip_mask.png"),
              str(out / "masks" / "02_extended_void_mask.png")]

    # ---- Stage 6: classification -------------------------------------------
    cls = classify_regions(
        gray, refined.labels, ext.per_region, tip.threshold,
        cfg.classification, cfg.void, n_regions=refined.n_regions,
    )
    cv2.imwrite(str(out / "masks" / "03_class_image.png"), cls.class_image)
    files.append(str(out / "masks" / "03_class_image.png"))

    # ---- Stage 7: measurements ---------------------------------------------
    voids = measure_void_components(ext.void_mask, refined.labels, image_id)
    px = cfg.calibration.pixel_size_um
    void_df = void_measurements_dataframe(voids, px)
    void_df.to_csv(out / "measurements" / "void_measurements.csv", index=False)
    files.append(str(out / "measurements" / "void_measurements.csv"))

    region_rows = []
    for s in cls.stats:
        region_rows.append({
            "image_id": image_id,
            "region_id": s.region_id,
            "region_class": CLASS_NAMES[s.cls],
            "region_area_px": s.area,
            "region_area_fraction": s.area / (cls.class_image.size),
            "n_associated_voids": int((void_df["parent_region_id"] == s.region_id).sum())
            if not void_df.empty else 0,
            "void_area_px": s.void_area,
            "void_fraction_of_region": round(s.void_fraction, 4),
            "peak_intensity_Ip": s.peak_intensity,
            "cutoff_intensity_Ic": round(s.cutoff_intensity, 2),
            "mean_intensity": round(s.mean_intensity, 2),
            "std_intensity": round(s.std_intensity, 2),
            "tip_pixels": s.tip_pixels,
            "extension_threshold_Iv": s.extension_threshold,
            "classification_reason": s.reason,
        })
    region_df = pd.DataFrame(region_rows)
    region_df.to_csv(out / "measurements" / "region_measurements.csv", index=False)
    files.append(str(out / "measurements" / "region_measurements.csv"))

    fracs = area_fractions(cls.class_image)
    summary = summary_statistics(voids, cls.class_image, px)
    summary.update({
        "image_id": image_id,
        "relative_path": str(rec.path),
        "threshold_global": thr.threshold,
        "I_t": tip.threshold,
        "k_percent": cfg.classification.cutoff_percentile,
        "j": cfg.classification.global_sigma_multiplier,
        "mu_g": cls.diagnostics["mu_g"],
        "sigma_g": cls.diagnostics["sigma_g"],
    })
    _flat_summary(summary).to_csv(out / "measurements" / "summary_statistics.csv", index=False)
    files.append(str(out / "measurements" / "summary_statistics.csv"))

    # ---- Stage 8: figures ---------------------------------------------------
    seg_rgb = viz.class_rgb_image(gray, cls.class_image, ext.void_mask)
    if make_figures:
        viz.figure_original_and_histogram(
            gray, hist_counts, hist_centres, thr.threshold,
            f"{image_id}: input and histogram", out / "figures" / "03_histogram.png", cfg.output.fig_dpi,
        )
        viz.figure_intermediates(
            gray, initial.binary, initial.thinned, initial.linked, initial.labels,
            f"{image_id}: stage 2.1 initial contour", out / "figures" / "04_initial_contour.png",
            cfg.output.fig_dpi,
        )
        viz.figure_refinement(
            gray, initial.labels, refined.labels, refined.refined_regions,
            f"{image_id}: stage 2.2 refinement", out / "figures" / "05_refinement.png",
            cfg.output.fig_dpi,
        )
        viz.figure_void_detection(
            gray, tip.counts if tip.counts is not None else hist_counts,
            tip.centres if tip.centres is not None else hist_centres,
            tip.threshold, tip.tip_mask, ext.void_mask,
            f"{image_id}: stage 2.3 void detection", out / "figures" / "06_void_detection.png",
            cfg.output.fig_dpi,
        )
        viz.figure_final_segmentation(
            gray, cls.class_image, cls.diagnostics["counts"], fracs,
            f"{image_id}: region classification", out / "figures" / "07_final_segmentation.png",
            cfg.output.fig_dpi, tiff=export_tiff, void_mask=ext.void_mask,
        )
        viz.figure_side_by_side(
            gray, cls.class_image, f"{image_id}: original vs segmentation",
            out / "figures" / "08_side_by_side.png", cfg.output.fig_dpi,
            void_mask=ext.void_mask,
        )
        counts, edges, _ = size_distribution(voids)
        viz.figure_void_size_distribution(
            counts, edges, voids, f"{image_id}: void size and shape",
            out / "figures" / "09_void_size_distribution.png", cfg.output.fig_dpi, px,
        )
        viz.figure_summary(
            fracs, summary, f"{image_id}: quantitative summary",
            out / "figures" / "10_summary.png", cfg.output.fig_dpi,
        )
        viz.figure_composite(
            gray, initial.linked, ext.void_mask, seg_rgb, (hist_counts, hist_centres),
            f"{image_id}: overview", out / "figures" / "00_composite_overview.png",
            cfg.output.fig_dpi,
        )
        for name in ("00_composite_overview", "03_histogram", "04_initial_contour",
                     "05_refinement", "06_void_detection", "07_final_segmentation",
                     "08_side_by_side", "09_void_size_distribution", "10_summary"):
            files.append(str(out / "figures" / f"{name}.png"))

    viz.save_class_overlay(gray, cls.class_image, out / "masks" / "04_class_overlay.png",
                           tiff=export_tiff, void_mask=ext.void_mask)
    files.append(str(out / "masks" / "04_class_overlay.png"))

    # ---- Metadata -----------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    metadata = {
        "image": rec.summary(),
        "image_id": image_id,
        "processing": {
            "started": started,
            "finished": _now(),
            "elapsed_seconds": round(elapsed, 2),
            "software_version": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "outputs_complete": True,
        },
        "parameters": cfg.to_dict(),
        "parameter_documentation": {
            k: {
                "purpose": v[0], "default": v[1], "unit": v[2],
                "constraint": v[3], "provenance": v[4], "effect": v[5],
            }
            for k, v in PARAMETER_DOCS.items()
        },
        "preprocessing": {
            "applied": pre.applied,
            "operations": pre.operations,
        },
        "thresholds": {
            "global_th": thr.threshold,
            "global_mu": thr.mean,
            "global_sigma": thr.std,
            "intermediate_range": [thr.range_low, thr.range_high],
            "void_tip_I_t": tip.threshold,
            "void_tip_status": tip.status,
            "refinement_cut_off": refined.diagnostics["candidate_cut_off"],
        },
        "calibration": {
            "pixel_size_um": px,
            "source": cfg.calibration.source,
            "physical_measurements_available": px is not None,
        },
        "stages": {
            "contour_detection": initial.diagnostics,
            "refinement": refined.diagnostics,
            "void_detection": ext.diagnostics,
            "classification": cls.diagnostics,
        },
        "measurements": {
            "n_individual_voids": len(voids),
            "class_area_fractions": fracs,
            "counts": cls.diagnostics["counts"],
        },
        "warnings": warnings,
        "output_files": files,
    }
    meta_path = out / "run_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    files.append(str(meta_path))

    LOGGER.info("Run complete in %.2f s -> %s", elapsed, out)
    return PipelineOutputs(
        out_dir=out,
        image=rec,
        class_image=cls.class_image,
        void_mask=ext.void_mask,
        region_labels=refined.labels,
        summary=summary,
        files=files,
        warnings=warnings,
    )


def _flat_summary(summary: dict[str, Any]) -> pd.DataFrame:
    """Tidy summary table.  The unit is a separate column, never mixed into a value."""
    rows = [{"metric": key, "value": _as_number(value), "unit": _unit_for(key)}
            for key, value in summary.items()]
    return pd.DataFrame(rows, columns=["metric", "value", "unit"])


def _as_number(value: Any) -> float | int | None:
    """Coerce a metric to a numeric CSV cell (None -> empty cell).

    Booleans become 1/0 so the ``value`` column keeps a numeric dtype instead
    of being promoted to strings by a single True/False entry.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if isinstance(value, (float, np.floating)) else int(value)
    return None       # non-numeric identifiers (e.g. image_id) go to metadata only


def _unit_for(key: str) -> str:
    """Explicit unit label for a summary metric.

    ``"n/a"`` is never returned: pandas parses it as a missing value on read, so
    the column would silently lose entries.  ``"dimensionless"`` and ``"count"``
    carry the meaning without ambiguity.
    """
    if "_px2" in key:
        return "px^2"
    if "_um2" in key:
        return "um^2"
    if "per_mm2" in key:
        return "1/mm^2"
    if "_px" in key:
        return "px"
    if "_um" in key:
        return "um"
    if key.endswith("_percent"):
        return "%"
    if "fraction" in key:
        return "proportion"
    if key.startswith("n_") or "count" in key:
        return "count"
    if key in ("I_t", "mu_g", "sigma_g", "threshold_global", "k_percent"):
        return "intensity level"
    return "dimensionless"
