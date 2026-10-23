"""Region characterization - reference paper section 2.3, final part.

Three classes:

``SURFACE``
    The region contains no void (no void tip, hence nothing to extend).
``VOID``
    A region consisting of a void of considerable depth (type-1 void).
``MIXED``
    A region comprising surface plus one or more voids of insignificant depth
    (type-2 voids).

Decision procedure taken verbatim from the paper:

1. Regions with no void pixel are ``SURFACE``.
2. A region fully covered by void is ``VOID``.
3. Any remaining region (partially covered by void) is examined further:

   * ``I_p`` is the intensity of the peak of the region's intensity histogram.
     For a void region ``I_p <= I_t``; this is unlikely to hold for a mixed
     region because mixed regions contain surface pixels of higher intensity.
   * ``I_c`` is the **minimum cutoff intensity such that k% of the pixels in the
     region have intensity lower than the cutoff**.  To qualify as a void region
     ``I_c < mu_g + j * sigma_g``, where ``mu_g`` and ``sigma_g`` are the mean
     and standard deviation of the whole image.  The paper uses ``k = 80`` and
     ``j = 1``.

   A partially covered region is ``VOID`` only when **both** conditions hold,
   otherwise ``MIXED``.

The ``k%`` cutoff is computed with ``numpy.percentile(method="inverted_cdf")``,
which returns the smallest value such that at least k% of the samples are less
than or equal to it - the exact wording of the paper.  The method is configurable
because library percentiles differ at the boundary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import ClassificationConfig, VoidConfig

LOGGER = logging.getLogger(__name__)

#: Integer class identifiers (an engineering choice: the paper names the classes
#: but assigns no numbers).  These values are stable and appear in the
#: class-label image and in every CSV.
CLASS_UNASSIGNED = 0
CLASS_SURFACE = 1
CLASS_MIXED = 2
CLASS_VOID = 3

CLASS_NAMES: dict[int, str] = {
    CLASS_UNASSIGNED: "UNASSIGNED",
    CLASS_SURFACE: "SURFACE",
    CLASS_MIXED: "MIXED",
    CLASS_VOID: "VOID",
}

#: Class id used in the segmentation image for a pixel labelled VOID.
CLASS_IDS = (CLASS_SURFACE, CLASS_MIXED, CLASS_VOID)


@dataclass
class RegionStats:
    """Per-region statistics used by the classification and written to CSV."""

    region_id: int
    area: int
    has_void: bool
    void_area: int
    void_fraction: float
    peak_intensity: int
    cutoff_intensity: float
    mean_intensity: float
    std_intensity: float
    tip_pixels: int
    extension_threshold: float | None
    reason: str = ""
    cls: int = CLASS_UNASSIGNED


@dataclass
class ClassificationResult:
    """Outcome of region characterization."""

    class_map: np.ndarray                 # int32 region label -> class id (index)
    region_class: dict[int, int]          # region id -> class id
    class_image: np.ndarray               # uint8 class id per pixel (0 = boundary)
    stats: list[RegionStats] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def global_statistics(gray: np.ndarray) -> tuple[float, float]:
    """Mean and standard deviation of intensity over the whole image."""
    if gray.size == 0:
        raise ValueError("Empty image")
    return float(gray.mean()), float(gray.std())


def histogram_peak_intensity(
    values: np.ndarray, bin_width: int
) -> tuple[int, np.ndarray, np.ndarray]:
    """Intensity of the histogram peak of ``values`` (``I_p``).

    The histogram uses bins of ``bin_width`` levels; the returned intensity is
    the centre of the most populated bin.  A tie is resolved toward the lower
    intensity, which is deterministic and conservative for a void test.
    """
    if values.size == 0:
        raise ValueError("Cannot compute a histogram peak of an empty region")
    if bin_width < 1:
        raise ValueError("bin_width must be >= 1")
    edges = np.arange(0, 256 + bin_width, bin_width, dtype=np.int64)
    edges = edges[edges <= 255]
    if edges[-1] != 256:
        edges = np.append(edges, 256)
    counts, edges = np.histogram(values, bins=edges)
    centres = ((edges[:-1] + edges[1:] - 1) // 2).astype(int)
    peak = int(np.argmax(counts))
    return int(centres[peak]), counts, centres


def cutoff_intensity(values: np.ndarray, percentile: float, method: str) -> float:
    """``I_c``: minimum intensity such that ``percentile``% of pixels are below it."""
    if values.size == 0:
        raise ValueError("Cannot compute a cutoff of an empty region")
    if not (0 < percentile <= 100):
        raise ValueError("percentile must be in (0, 100]")
    return float(np.percentile(values, percentile, method=method))


def classify_region(
    values: np.ndarray,
    void_area: int,
    has_tip: bool,
    i_t: float | None,
    mu_g: float,
    sigma_g: float,
    cls_cfg: ClassificationConfig,
    void_cfg: VoidConfig,
) -> tuple[int, str]:
    """Classify one region following the paper's decision sequence.

    Returns
    -------
    (class_id, reason)
        ``reason`` records which rule fired, for the diagnostic CSV.
    """
    area = int(values.size)
    if area == 0:
        raise ValueError("Region contains no pixels")
    if not has_tip or void_area == 0:
        return CLASS_SURFACE, "no void detected"
    if void_area >= area:
        return CLASS_VOID, "region fully covered by void"
    if i_t is None:
        return CLASS_MIXED, "void present but I_t unavailable"

    ip, _, _ = histogram_peak_intensity(values, void_cfg.hist_bin_width)
    ic = cutoff_intensity(values, cls_cfg.cutoff_percentile, cls_cfg.percentile_method)
    limit = mu_g + cls_cfg.global_sigma_multiplier * sigma_g

    ip_ok = ip <= i_t
    ic_ok = ic < limit
    if ip_ok and ic_ok:
        return CLASS_VOID, f"I_p={ip} <= I_t and I_c={ic:.1f} < {limit:.1f}"
    why = []
    if not ip_ok:
        why.append(f"I_p={ip} > I_t={i_t:.0f}")
    if not ic_ok:
        why.append(f"I_c={ic:.1f} >= {limit:.1f}")
    return CLASS_MIXED, "partially void; " + "; ".join(why)


def classify_regions(
    gray: np.ndarray,
    labels: np.ndarray,
    void_per_region: dict[int, dict[str, Any]],
    i_t: float | None,
    cls_cfg: ClassificationConfig,
    void_cfg: VoidConfig,
    n_regions: int | None = None,
) -> ClassificationResult:
    """Classify every labelled region and build the class image.

    Parameters
    ----------
    gray:
        Grey-scale image.
    labels:
        Refined region labelling (1..N, 0 = boundary).
    void_per_region:
        Per-region void statistics from :mod:`void_detection`.
    i_t:
        Global void-tip cutoff, or ``None`` if tip detection failed (in that case
        no region can be characterised as VOID; all are ``SURFACE``/``MIXED``
        according to the void evidence).
    cls_cfg, void_cfg:
        Parameter sets.
    n_regions:
        Optional upper bound for the label id; inferred from ``labels`` if absent.

    Returns
    -------
    ClassificationResult
        ``class_map`` indexed by region id, ``class_image`` per pixel.
    """
    if gray.shape != labels.shape:
        raise ValueError("gray and labels must have the same shape")
    if n_regions is None:
        n_regions = int(labels.max())

    mu_g, sigma_g = global_statistics(gray)
    class_map = np.zeros(n_regions + 1, dtype=np.int32)
    stats: list[RegionStats] = []
    class_image = np.zeros(gray.shape, dtype=np.uint8)

    for rid in range(1, n_regions + 1):
        mask = labels == rid
        area = int(mask.sum())
        if area == 0:
            continue
        values = gray[mask]
        rec = void_per_region.get(rid, {})
        void_area = int(rec.get("void_area", 0))
        has_tip = bool(rec.get("has_tip", False))
        tip_pixels = int(rec.get("tip_pixels", 0))
        ext_thr = rec.get("extension_threshold")

        cls_id, reason = classify_region(
            values, void_area, has_tip, i_t, mu_g, sigma_g, cls_cfg, void_cfg
        )
        class_map[rid] = cls_id
        class_image[mask] = cls_id

        ip, _, _ = histogram_peak_intensity(values, void_cfg.hist_bin_width)
        ic = cutoff_intensity(values, cls_cfg.cutoff_percentile, cls_cfg.percentile_method)
        stats.append(RegionStats(
            region_id=rid,
            area=area,
            has_void=has_tip,
            void_area=void_area,
            void_fraction=void_area / area,
            peak_intensity=ip,
            cutoff_intensity=ic,
            mean_intensity=float(values.mean()),
            std_intensity=float(values.std()),
            tip_pixels=tip_pixels,
            extension_threshold=None if ext_thr is None else float(ext_thr),
            reason=reason,
            cls=cls_id,
        ))

    counts = {CLASS_NAMES[c]: 0 for c in CLASS_IDS}
    for s in stats:
        counts[CLASS_NAMES[s.cls]] += 1
    LOGGER.info("Classification: %s", counts)

    return ClassificationResult(
        class_map=class_map,
        # Only ids that actually appear in the labelling are reported; an id that
        # was never assigned a class must not appear as UNASSIGNED.
        region_class={int(r): int(c) for r, c in enumerate(class_map) if r > 0 and c != CLASS_UNASSIGNED},
        class_image=class_image,
        stats=stats,
        diagnostics={
            "mu_g": mu_g,
            "sigma_g": sigma_g,
            "I_t": i_t,
            "cutoff_limit": mu_g + cls_cfg.global_sigma_multiplier * sigma_g,
            "k_percent": cls_cfg.cutoff_percentile,
            "j": cls_cfg.global_sigma_multiplier,
            "counts": counts,
        },
    )
