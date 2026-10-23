"""Quantitative fractography measurements.

All measurements are computed from the actual segmentation products:

* the refined region labelling,
* the extended void mask,
* the per-region classification.

Definitions (standard, compatible with the project brief and the paper):

Equivalent circular diameter  ``D_eq = sqrt(4A/pi)``
Aspect ratio                  ``AR = L_major / L_minor``
Circularity                   ``C = 4*pi*A / P^2`` (1 for a disc)
Void area fraction            ``f_v = A_void / A_analysed``

Individual voids are counted as connected components of the restricted void
mask.  The number of void *regions* is not the number of individual voids: a
mixed region may hold several shallow voids, each of which is counted
separately when the mask separates them.  Touching voids that the mask joins
are counted as one component; this is reported, not hidden.

Physical units: the paper reports no pixel-to-micrometre calibration, so all
physical columns are left empty unless the user supplies a pixel size.  A
pixel size is never inferred from magnification.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .region_classification import CLASS_MIXED, CLASS_NAMES, CLASS_SURFACE, CLASS_VOID

LOGGER = logging.getLogger(__name__)

#: Minimum area (px) for a connected void component to be counted individually.
DEFAULT_MIN_VOID_AREA = 10


@dataclass
class VoidMeasurement:
    """Shape and size of one connected void component."""

    image_id: str
    void_id: int
    parent_region_id: int
    pixel_area: int
    equivalent_diameter_px: float
    major_axis_px: float
    minor_axis_px: float
    aspect_ratio: float
    perimeter_px: float
    circularity: float
    centroid_row: float
    centroid_col: float
    touches_border: bool

    def as_row(self, pixel_size_um: float | None) -> dict[str, Any]:
        """CSV row; physical columns are ``None`` when uncalibrated."""
        row: dict[str, Any] = {
            "image_id": self.image_id,
            "void_id": self.void_id,
            "parent_region_id": self.parent_region_id,
            "pixel_area": self.pixel_area,
            "equivalent_diameter_px": round(self.equivalent_diameter_px, 4),
            "major_axis_px": round(self.major_axis_px, 4),
            "minor_axis_px": round(self.minor_axis_px, 4),
            "aspect_ratio": round(self.aspect_ratio, 4),
            "perimeter_px": round(self.perimeter_px, 4),
            "circularity": round(self.circularity, 4),
            "centroid_row_px": round(self.centroid_row, 2),
            "centroid_col_px": round(self.centroid_col, 2),
            "touches_border": self.touches_border,
            "calibration_available": pixel_size_um is not None,
            "physical_area_um2": None,
            "equivalent_diameter_um": None,
        }
        if pixel_size_um is not None:
            row["physical_area_um2"] = round(self.pixel_area * pixel_size_um ** 2, 4)
            row["equivalent_diameter_um"] = round(
                self.equivalent_diameter_px * pixel_size_um, 4
            )
        return row


def equivalent_diameter(area: float) -> float:
    """``D_eq = sqrt(4A/pi)``."""
    if area < 0:
        raise ValueError("area must be non-negative")
    return math.sqrt(4.0 * area / math.pi)


def circularity(area: float, perimeter: float) -> float:
    """``C = 4*pi*A/P^2``; 0 when the perimeter is 0."""
    if area < 0 or perimeter < 0:
        raise ValueError("area and perimeter must be non-negative")
    if perimeter == 0:
        return 0.0
    return 4.0 * math.pi * area / (perimeter ** 2)


def measure_void_components(
    void_mask: np.ndarray,
    labels: np.ndarray,
    image_id: str,
    min_area: int = DEFAULT_MIN_VOID_AREA,
) -> list[VoidMeasurement]:
    """Split the void mask into connected components and measure each.

    Parameters
    ----------
    void_mask:
        Boolean extended void mask.
    labels:
        Refined region labelling, used to attribute each void to a parent region.
    image_id:
        Identifier written to every row.
    min_area:
        Components smaller than this are excluded as speckle.

    Returns
    -------
    list[VoidMeasurement]
        Ordered by descending area (void_id starts at 1).
    """
    if void_mask.shape != labels.shape:
        raise ValueError("void_mask and labels must have the same shape")
    if void_mask.ndim != 2:
        raise ValueError("void_mask must be 2-D")
    if min_area < 1:
        raise ValueError("min_area must be >= 1")

    binary = void_mask.astype(np.uint8)
    n, comp, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h, w = binary.shape
    out: list[VoidMeasurement] = []
    for cid in range(1, n):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[cid, cv2.CC_STAT_LEFT])
        y = int(stats[cid, cv2.CC_STAT_TOP])
        bw = int(stats[cid, cv2.CC_STAT_WIDTH])
        bh = int(stats[cid, cv2.CC_STAT_HEIGHT])
        comp_mask = (comp[y:y + bh, x:x + bw] == cid).astype(np.uint8)
        rect = cv2.minAreaRect(cv2.findNonZero(comp_mask))
        (_, _), (rw, rh), _ = rect
        major, minor = max(rw, rh), min(rw, rh)
        aspect = major / minor if minor > 0 else float("inf")
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
        parent = int(np.bincount(labels[y:y + bh, x:x + bw][comp_mask > 0]).argmax())
        touches = bool(x == 0 or y == 0 or x + bw >= w or y + bh >= h)
        out.append(VoidMeasurement(
            image_id=image_id,
            void_id=0,
            parent_region_id=parent,
            pixel_area=area,
            equivalent_diameter_px=equivalent_diameter(area),
            major_axis_px=major,
            minor_axis_px=minor,
            aspect_ratio=aspect,
            perimeter_px=perimeter,
            circularity=circularity(area, perimeter),
            centroid_row=float(centroids[cid][1]),
            centroid_col=float(centroids[cid][0]),
            touches_border=touches,
        ))

    out.sort(key=lambda m: (-m.pixel_area, m.centroid_row, m.centroid_col))
    for i, m in enumerate(out, start=1):
        m.void_id = i
    LOGGER.info("Measured %d individual voids", len(out))
    return out


def area_fractions(
    class_image: np.ndarray, analysed_area: int | None = None
) -> dict[str, float]:
    """Pixel area fractions of each class over the analysed area."""
    if class_image.size == 0:
        raise ValueError("Empty class image")
    total = int(analysed_area if analysed_area is not None else class_image.size)
    if total <= 0:
        raise ValueError("analysed_area must be positive")
    return {
        CLASS_NAMES[CLASS_SURFACE]: float((class_image == CLASS_SURFACE).sum() / total),
        CLASS_NAMES[CLASS_MIXED]: float((class_image == CLASS_MIXED).sum() / total),
        CLASS_NAMES[CLASS_VOID]: float((class_image == CLASS_VOID).sum() / total),
    }


def summary_statistics(
    voids: list[VoidMeasurement],
    class_image: np.ndarray,
    pixel_size_um: float | None,
) -> dict[str, Any]:
    """Aggregate measurements with explicit units in every key."""
    total = int(class_image.size)
    void_pixels = int((class_image == CLASS_VOID).sum())
    mixed_pixels = int((class_image == CLASS_MIXED).sum())
    surface_pixels = int((class_image == CLASS_SURFACE).sum())
    fracs = area_fractions(class_image)

    d_eq = np.array([m.equivalent_diameter_px for m in voids], dtype=float)
    ar = np.array([m.aspect_ratio for m in voids], dtype=float)
    circ = np.array([m.circularity for m in voids], dtype=float)
    areas = np.array([m.pixel_area for m in voids], dtype=float)

    def _stat(arr: np.ndarray, name: str, unit: str) -> dict[str, Any]:
        if arr.size == 0:
            return {f"{name}_{s}_{unit}": None for s in
                    ("mean", "std", "min", "max", "median")}
        return {
            f"{name}_mean_{unit}": float(arr.mean()),
            f"{name}_std_{unit}": float(arr.std()),
            f"{name}_min_{unit}": float(arr.min()),
            f"{name}_max_{unit}": float(arr.max()),
            f"{name}_median_{unit}": float(np.median(arr)),
        }

    out: dict[str, Any] = {
        "n_individual_voids": len(voids),
        "total_void_area_px": int(areas.sum()) if len(areas) else 0,
        "analysed_area_px": total,
        "void_area_fraction": fracs["VOID"],
        "surface_area_fraction": fracs["SURFACE"],
        "mixed_area_fraction": fracs["MIXED"],
        "void_region_area_px": void_pixels,
        "surface_region_area_px": surface_pixels,
        "mixed_region_area_px": mixed_pixels,
        "calibration_available": pixel_size_um is not None,
        "pixel_size_um": pixel_size_um,
        "void_number_density_per_mm2": None,
    }
    out.update(_stat(d_eq, "equivalent_diameter", "px"))
    out.update(_stat(ar, "aspect_ratio", "dimensionless"))
    out.update(_stat(circ, "circularity", "dimensionless"))
    out.update(_stat(areas, "void_area", "px2"))

    if pixel_size_um is not None:
        analysed_mm2 = total * (pixel_size_um ** 2) / 1e6
        if analysed_mm2 > 0:
            out["void_number_density_per_mm2"] = len(voids) / analysed_mm2
        out["void_area_fraction_percent"] = fracs["VOID"] * 100.0
        out["surface_area_fraction_percent"] = fracs["SURFACE"] * 100.0
        out["mixed_area_fraction_percent"] = fracs["MIXED"] * 100.0
        out["total_void_area_um2"] = (
            float(areas.sum() * pixel_size_um ** 2) if len(areas) else 0.0
        )
        if d_eq.size:
            um_d = d_eq * pixel_size_um
            out["equivalent_diameter_um_mean_um"] = float(um_d.mean())
            out["equivalent_diameter_um_median_um"] = float(np.median(um_d))
    return out


def void_measurements_dataframe(
    voids: list[VoidMeasurement], pixel_size_um: float | None
) -> pd.DataFrame:
    """One row per individual void."""
    columns = [
        "image_id", "void_id", "parent_region_id", "pixel_area",
        "equivalent_diameter_px", "major_axis_px", "minor_axis_px",
        "aspect_ratio", "perimeter_px", "circularity",
        "centroid_row_px", "centroid_col_px", "touches_border",
        "calibration_available", "physical_area_um2", "equivalent_diameter_um",
    ]
    if not voids:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([v.as_row(pixel_size_um) for v in voids])[columns]


def size_distribution(
    voids: list[VoidMeasurement], bins: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Histogram of equivalent diameters (px).

    Returns ``(counts, bin_edges, bin_centres)``.  With fewer than two voids an
    empty distribution is returned so the caller can skip plotting.
    """
    d = np.array([v.equivalent_diameter_px for v in voids], dtype=float)
    if d.size < 2:
        return np.array([]), np.array([]), np.array([])
    if bins is None:
        bins = np.histogram_bin_edges(d, bins=min(30, max(5, d.size // 3)))
    counts, edges = np.histogram(d, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return counts, edges, centres
