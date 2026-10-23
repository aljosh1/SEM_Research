"""Void detection - reference paper section 2.3, two steps.

**Step 1 - detection of void tips.**  Both void types (deep type-1, shallow
type-2) contain a zone of very low intensity: the *tip*, the bottom-most area of
the void.  The intensity histogram is formed by splitting the intensity scale
into bins of small size, and a local maximum is determined by comparing a bin
with its previous and next bin.  ``I_t`` is the **upper bound of the bin
corresponding to the first local maximum** (scanning upward from zero intensity).
Pixels with intensity below ``I_t`` form the tip.

**Step 2 - extension of the void tip.**  Extension runs only on regions that
contain a tip.  Pixels surrounding the tip have higher intensity but still belong
to the void.  The local threshold is

    I_v = max(mu_l - 0.5 * sigma_l, I_t)

where ``mu_l`` and ``sigma_l`` are the mean and standard deviation of intensity
**in the region**.  Pixels with intensity below ``I_v`` join the void.

Failure handling: if no local maximum exists in the low range (flat histogram, a
single impulse bin, an all-dark image) the function returns ``None`` plus a
diagnostic reason.  It never invents a threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import VoidConfig
from .contour_detection import binned_histogram

LOGGER = logging.getLogger(__name__)


@dataclass
class VoidTipResult:
    """Outcome of the void-tip detection step."""

    threshold: float | None            # I_t, or None when detection failed
    tip_mask: np.ndarray               # bool - pixels below I_t
    status: str                        # "ok" or a failure reason
    peak_bin_index: int | None = None
    peak_bin_bounds: tuple[int, int] | None = None
    counts: np.ndarray | None = None
    centres: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class VoidExtensionResult:
    """Outcome of the void-tip extension step."""

    void_mask: np.ndarray                     # bool - extended void pixels
    per_region: dict[int, dict[str, Any]]     # region id -> statistics
    diagnostics: dict[str, Any] = field(default_factory=dict)


def find_first_local_maximum(counts: np.ndarray) -> int | None:
    """Index of the first (lowest-intensity) local maximum of the histogram.

    The paper: "local maxima is determined by considering the previous and next
    bin", and the peak of interest is "the local peak in the low range".

    A bin qualifies when its count exceeds *both* neighbours.  Bin 0 has no
    previous bin; the half-bin before it is treated as empty (the intensity
    scale begins at zero), so bin 0 qualifies when ``counts[0] > counts[1]``.
    This convention matters in practice: a saturated dark void bottom produces
    a dominant spike in bin 0, and skipping bin 0 would make the program return
    a spurious peak in the body of the histogram instead (observed on real SEM
    data during development).

    Plateaus (equal adjacent counts) do not qualify, which prevents a flat
    background from being mistaken for a peak.

    Returns
    -------
    int or None
        Bin index, or ``None`` if no local maximum exists.
    """
    if counts.size < 3:
        return None
    if counts[0] > counts[1]:
        return 0
    for i in range(1, len(counts) - 1):
        if counts[i] > counts[i - 1] and counts[i] >= counts[i + 1]:
            return i
    return None


def detect_void_tips(
    gray: np.ndarray,
    cfg: VoidConfig,
    bin_upper_bound: np.ndarray | None = None,
) -> VoidTipResult:
    """Compute ``I_t`` and the void-tip mask.

    Parameters
    ----------
    gray:
        2-D ``uint8`` image.
    cfg:
        Void settings (histogram bin width).
    bin_upper_bound:
        Optional per-bin upper intensity bound.  When omitted it is derived from
        ``cfg.hist_bin_width`` as ``(i + 1) * bin_width - 1`` of the bin centre.

    Returns
    -------
    VoidTipResult
        ``threshold`` is ``None`` when no local maximum exists.
    """
    if gray.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {gray.shape}")
    if gray.size == 0:
        raise ValueError("Cannot detect void tips in an empty image")

    counts, centres = binned_histogram(gray, cfg.hist_bin_width)
    if counts.sum() == 0 or counts.max() == 0:
        return _failed_tip("empty histogram", counts, centres)
    if len(np.unique(gray)) < 3:
        return _failed_tip("uniform or near-uniform image", counts, centres)

    idx = find_first_local_maximum(counts)
    if idx is None:
        return _failed_tip("no local maximum in the histogram", counts, centres)

    # "The upper bound of the intensity bin corresponding to the first local
    # maxima is taken as I_t."  For a bin of width w whose centre is c, the
    # intensities it covers are [c - (w-1)//2, c + w//2]; the upper bound is the
    # top of that inclusive range.
    w = cfg.hist_bin_width
    upper = int(centres[idx] + w // 2)
    if bin_upper_bound is not None:
        upper = int(bin_upper_bound[idx])
    upper = int(np.clip(upper, 0, 255))

    tip_mask = gray < upper
    return VoidTipResult(
        threshold=float(upper),
        tip_mask=tip_mask,
        status="ok",
        peak_bin_index=int(idx),
        peak_bin_bounds=(int(centres[idx] - cfg.hist_bin_width // 2), upper),
        counts=counts,
        centres=centres,
        diagnostics={
            "bin_width": cfg.hist_bin_width,
            "peak_count": int(counts[idx]),
            "tip_pixels": int(tip_mask.sum()),
            "tip_fraction": float(tip_mask.mean()),
        },
    )


def _failed_tip(reason: str, counts: np.ndarray, centres: np.ndarray) -> VoidTipResult:
    LOGGER.warning("Void-tip detection failed: %s", reason)
    return VoidTipResult(
        threshold=None,
        tip_mask=np.zeros(0, dtype=bool),
        status=reason,
        counts=counts,
        centres=centres,
        diagnostics={"reason": reason},
    )


def extension_threshold(region_values: np.ndarray, i_t: float, cfg: VoidConfig) -> float:
    """``I_v = max(mu_l - coeff * sigma_l, I_t)`` for one region."""
    if region_values.size == 0:
        raise ValueError("Region contains no pixels")
    mu = float(region_values.mean())
    sd = float(region_values.std())
    return max(mu - cfg.extension_sigma_coeff * sd, float(i_t))


def extend_voids(
    gray: np.ndarray,
    labels: np.ndarray,
    tip: VoidTipResult,
    cfg: VoidConfig,
) -> VoidExtensionResult:
    """Extend tips into full voids, region by region.

    Only regions containing at least one tip pixel are processed.  Within such a
    region every pixel below ``I_v`` joins the void.  When
    ``cfg.restrict_to_region`` is True the void never leaves its region, so
    surface pixels of a *neighbouring* region cannot be absorbed merely because
    their intensity is similar.

    Parameters
    ----------
    gray, labels:
        Image and refined region labelling of equal shape.
    tip:
        Result of :func:`detect_void_tips`.
    cfg:
        Void settings.

    Returns
    -------
    VoidExtensionResult
        ``void_mask`` (bool) and per-region statistics.
    """
    if gray.shape != labels.shape:
        raise ValueError("gray and labels must have the same shape")
    if tip.threshold is None:
        return VoidExtensionResult(
            void_mask=np.zeros(gray.shape, dtype=bool),
            per_region={},
            diagnostics={"status": "skipped", "reason": tip.status},
        )

    void_mask = np.zeros(gray.shape, dtype=bool)
    per_region: dict[int, dict[str, Any]] = {}
    region_ids = np.unique(labels)
    for rid in region_ids:
        rid = int(rid)
        if rid == 0:
            continue
        mask = labels == rid
        n_tip = int((mask & tip.tip_mask).sum())
        values = gray[mask]
        record: dict[str, Any] = {
            "region_id": rid,
            "area": int(mask.sum()),
            "tip_pixels": n_tip,
            "mean_intensity": float(values.mean()),
            "std_intensity": float(values.std()),
        }
        if n_tip > 0:
            i_v = extension_threshold(values, tip.threshold, cfg)
            void_pixels = mask & (gray < i_v)
            void_mask |= void_pixels
            record.update({
                "has_tip": True,
                "extension_threshold": float(i_v),
                "void_area": int(void_pixels.sum()),
                "void_fraction_of_region": float(void_pixels.sum() / max(mask.sum(), 1)),
            })
        else:
            record.update({"has_tip": False, "extension_threshold": None, "void_area": 0})
        per_region[rid] = record

    n_with_tip = sum(1 for r in per_region.values() if r["has_tip"])
    LOGGER.info("Void extension: %d/%d regions had tips, %d void pixels",
                n_with_tip, len(per_region), int(void_mask.sum()))
    return VoidExtensionResult(
        void_mask=void_mask,
        per_region=per_region,
        diagnostics={
            "status": "ok",
            "i_t": tip.threshold,
            "regions_with_tip": n_with_tip,
            "void_pixels": int(void_mask.sum()),
            "void_fraction": float(void_mask.mean()),
        },
    )


def detect_voids(
    gray: np.ndarray, labels: np.ndarray, cfg: VoidConfig
) -> tuple[VoidTipResult, VoidExtensionResult]:
    """Convenience wrapper: tip detection then extension."""
    tip = detect_void_tips(gray, cfg)
    if tip.threshold is None:
        tip.tip_mask = np.zeros(gray.shape, dtype=bool)
    ext = extend_voids(gray, labels, tip, cfg)
    return tip, ext
