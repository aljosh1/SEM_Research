"""Detection of region contour - reference paper section 2.1.

Pipeline reproduced from Banerjee et al., *Segmentation of SEM Fractograph into
Void and Surface Regions*, section 2.1 (which in turn cites the authors' 2011
paper for the threshold detail):

1. **Binarize** ``B(i,j) = 1 if F(i,j) > th`` with ``th = mu + 2*sigma`` over the
   *intermediate* (bell-shaped) part of the binned intensity histogram, ignoring
   the peaks in the two extreme intensity ranges.
2. **Thin** the boundary segments to single-pixel width.
3. **Edge linking**: each boundary pixel carries the Sobel gradient *direction*
   of the fractograph as its edge property.  A pixel is linked to the unlinked
   boundary pixel with the closest gradient direction inside its ``k x k``
   neighbourhood, ``k`` starting at 5 and growing to 7 when nothing is found.
   Linking iterates until no segment grows; the four scan directions of the
   paper are applied per iteration, and the result is re-thinned.
4. **Pruning**: region growing from surface/void pixels; the outer contour of
   each marked region is the closed contour, so open and false curves are
   discarded.

Interpretation notes (documented in ``docs/technical_note.md``)
---------------------------------------------------------------
* The paper's region growing marks *interior* pixels until a boundary pixel is
  met; the outer contour of that marked region is the region boundary.  This
  module maps that to connected components of the complement of the boundary
  network, then assigns each boundary pixel to the region whose seed reaches it
  first (a watershed over Euclidean distance).  Products therefore tile the
  image exactly and never leave a label ambiguous.
* Region ids are 1..N.  A region that touches the image frame is retained and
  flagged ``"border"``: a fractograph is not a closed scene, so the surrounding
  surface is itself a legitimate region.  Fully enclosed regions are
  ``"closed"``.  This choice is recorded as ``border_policy`` in diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .config import BinarizationConfig, ContourConfig

LOGGER = logging.getLogger(__name__)

#: Eight-connected neighbour offsets.
NEIGHBOURS_8: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1),
)

#: Number of angular classes used to compare gradient directions.
N_ANGLE_CLASSES = 12


@dataclass
class ThresholdResult:
    """Result of the global binarization of section 2.1."""

    threshold: float
    mean: float
    std: float
    range_low: int
    range_high: int
    binary: np.ndarray            # bool, True = boundary pixel


@dataclass
class LabelResult:
    """Result of pruning / region labelling."""

    labels: np.ndarray            # int32, 1..N region territory, 0 = unlabelled
    n_regions: int
    areas: np.ndarray             # area per label id, index 0 unused (0)
    statuses: list[str]           # status per label id, index 0 = "background"
    seeds: np.ndarray             # int32, seed-pixel ids before territory growth


@dataclass
class ContourResult:
    """Result of the complete initial contour detection."""

    binary: np.ndarray
    thinned: np.ndarray
    linked: np.ndarray
    labels: np.ndarray
    n_regions: int
    areas: np.ndarray
    statuses: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 1. Binarization (section 2.1)
# --------------------------------------------------------------------------- #
def global_threshold(gray: np.ndarray, cfg: BinarizationConfig) -> ThresholdResult:
    """Compute ``th = mu + 2*sigma`` over the intermediate histogram range.

    The intensity scale is split into bins of ``hist_bin_width`` levels; mean and
    standard deviation are evaluated over the bins that lie between the extreme
    low and extreme high bands.  This is the paper's "ignoring the peaks (if any)
    in two extreme intensity ranges" - in a fractograph those extremes are the
    deep-void minima and the bright boundary shell.

    Parameters
    ----------
    gray:
        2-D ``uint8`` image.
    cfg:
        Binarization settings.

    Returns
    -------
    ThresholdResult
        Threshold plus the statistics that produced it.
    """
    if gray.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {gray.shape}")
    if gray.size == 0:
        raise ValueError("Cannot threshold an empty image")
    if cfg.hist_bin_width < 1:
        raise ValueError("hist_bin_width must be >= 1")

    counts, centres = binned_histogram(gray, cfg.hist_bin_width)
    total = counts.sum()
    if total == 0:
        raise ValueError("Empty histogram")

    cdf = np.cumsum(counts) / total
    lo = int(centres[np.searchsorted(cdf, cfg.extreme_low_fraction)])
    hi_idx = min(int(np.searchsorted(cdf, 1.0 - cfg.extreme_high_fraction)), len(centres) - 1)
    hi = int(centres[hi_idx])
    if hi <= lo:  # degenerate distribution; fall back to the full range
        lo, hi = int(centres[0]), int(centres[-1])

    mask = (centres >= lo) & (centres <= hi)
    weights = counts[mask].astype(np.float64)
    if weights.sum() == 0:
        raise ValueError("The intermediate histogram range contains no pixels")
    mu = float((centres[mask] * weights).sum() / weights.sum())
    sigma = float(np.sqrt(((centres[mask] - mu) ** 2 * weights).sum() / weights.sum()))
    th = mu + cfg.sigma_multiplier * sigma

    LOGGER.info("Global threshold th=%.2f (mu=%.2f, sigma=%.2f, range [%d,%d])",
                th, mu, sigma, lo, hi)
    return ThresholdResult(th, mu, sigma, lo, hi, gray > th)


def binned_histogram(gray: np.ndarray, bin_width: int) -> tuple[np.ndarray, np.ndarray]:
    """Histogram counts and bin centres for bins of ``bin_width`` intensity levels."""
    if bin_width < 1:
        raise ValueError("bin_width must be >= 1")
    edges = np.arange(0, 256 + bin_width, bin_width, dtype=np.int64)
    edges = edges[edges <= 256]
    if edges[-1] != 256:
        edges = np.append(edges, 256)
    counts, edges = np.histogram(gray, bins=edges)
    centres = ((edges[:-1] + edges[1:] - 1) // 2).astype(np.int64)
    return counts.astype(np.int64), centres


# --------------------------------------------------------------------------- #
# 2. Thinning
# --------------------------------------------------------------------------- #
def thin(binary: np.ndarray) -> np.ndarray:
    """Reduce boundary segments to one-pixel width.

    ``skimage.morphology.skeletonize`` (Zhang-Suen) is topology preserving and
    deterministic, which the later connectivity checks depend on.
    """
    if binary.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {binary.shape}")
    binary = binary.astype(bool)
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    return skeletonize(binary)


def count_neighbours(binary: np.ndarray) -> np.ndarray:
    """Number of 8-connected neighbours of each foreground pixel."""
    if binary.ndim != 2:
        raise ValueError("binary must be 2-D")
    b = binary.astype(np.int16)
    # Pad rather than roll: rolling would wrap the frame and create false links.
    padded = np.pad(b, 1)
    n = np.zeros_like(b, dtype=np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            n += padded[1 + dy:1 + dy + b.shape[0], 1 + dx:1 + dx + b.shape[1]]
    return n


def endpoint_mask(skeleton: np.ndarray) -> np.ndarray:
    """Pixels of a one-pixel skeleton that have exactly one neighbour."""
    sk = skeleton.astype(bool)
    return sk & (count_neighbours(sk) == 1)


# --------------------------------------------------------------------------- #
# 3. Edge linking (section 2.1)
# --------------------------------------------------------------------------- #
def gradient_direction(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient direction in radians, folded to the orientation axis [0, pi).

    An edge has no sign, so ``d`` and ``d + pi`` describe the same edge and are
    treated as equal when comparing "closest gradient direction".
    """
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    ang = np.arctan2(gy, gx)
    return np.mod(ang, np.pi)


def _direction_classes(direction: np.ndarray, n_classes: int = N_ANGLE_CLASSES) -> np.ndarray:
    """Quantise folded direction into ``n_classes`` bins of ``pi / n_classes``."""
    cls = np.floor(np.mod(direction, np.pi) / np.pi * n_classes).astype(np.int16)
    return np.clip(cls, 0, n_classes - 1)


def link_edges(
    thinned: np.ndarray,
    direction: np.ndarray,
    cfg: ContourConfig,
    k_start: int | None = None,
    k_max: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Link broken boundary segments whose gradient directions agree.

    For each open segment end, the unlinked segment end with the closest gradient
    direction inside the ``k x k`` neighbourhood is joined to it.  ``k`` starts at
    ``k_start`` (paper: 5) and grows to ``k_max`` (paper: 7) when the smaller
    neighbourhood contains no suitable target.  Linking repeats until no segment
    grows; the grown result is re-thinned after every iteration.

    Parameters
    ----------
    thinned:
        Boolean one-pixel skeleton.
    direction:
        Folded gradient direction image, same shape as ``thinned``.
    cfg:
        Contour settings (provides iteration cap and angle tolerance).
    k_start, k_max:
        Neighbourhood side overrides; the refinement stage reduces them.

    Returns
    -------
    (linked, diagnostics)
        ``diagnostics`` holds ``links``, ``iterations`` and the per-direction
        link counts.
    """
    k_start = cfg.link_k_start if k_start is None else k_start
    k_max = cfg.link_k_max if k_max is None else k_max
    if k_start < 3 or k_start % 2 == 0:
        raise ValueError("k_start must be odd and >= 3")
    if k_max < k_start:
        raise ValueError("k_max must be >= k_start")
    if thinned.shape != direction.shape:
        raise ValueError("boundary and direction images must have the same shape")

    linked = thinned.astype(bool).copy()
    if not linked.any():
        return linked, {"links": 0, "iterations": 0, "links_by_scan": {}}

    max_delta = int(cfg.link_max_angle_rad / (np.pi / N_ANGLE_CLASSES))
    offsets = neighbourhood_offsets(k_start, k_max)
    # Restrict every per-iteration array to the boundary's bounding box.  The
    # linking decision is local (k <= 7), so pixels far from every boundary can
    # never take part; processing them cost most of the runtime when refinement
    # called this function once per candidate region.
    rows = np.flatnonzero(linked.any(axis=1))
    cols = np.flatnonzero(linked.any(axis=0))
    pad = k_max // 2 + 1
    r0 = max(0, int(rows[0]) - pad)
    r1 = min(linked.shape[0], int(rows[-1]) + pad + 1)
    c0 = max(0, int(cols[0]) - pad)
    c1 = min(linked.shape[1], int(cols[-1]) + pad + 1)
    work = linked[r0:r1, c0:c1].copy()
    cls = _direction_classes(direction[r0:r1, c0:c1])

    total_links = 0
    iterations = 0
    for iterations in range(1, cfg.link_max_iterations + 1):
        # Open ends are where a segment can still grow.
        ends = endpoint_mask(work)
        if ends.sum() < 2:
            break
        best_delta, src_r, src_c = _nearest_direction_match(ends, cls, offsets)
        accept = ends & (src_r >= 0) & (best_delta <= max_delta)
        n_accept = int(accept.sum())
        if n_accept == 0:
            break
        ys, xs = np.nonzero(accept)
        drawn = np.zeros_like(work)
        for y, x in zip(ys, xs):
            cv2.line(drawn, (int(x), int(y)), (int(src_c[y, x]), int(src_r[y, x])), 1, 1)
        total_links += n_accept
        # "In case, the boundary segments gain width, once again thinning process
        # is carried out."
        new = skeletonize(work | drawn.astype(bool))
        if np.array_equal(new, work):
            break
        work = new

    linked[r0:r1, c0:c1] = work

    LOGGER.info("Edge linking: %d links, %d iterations", total_links, iterations)
    return linked, {"links": int(total_links), "iterations": int(iterations)}


def neighbourhood_offsets(k_start: int, k_max: int) -> np.ndarray:
    """Offsets of every ``k x k`` search square, ordered by increasing ``k``.

    Returns ``(n, 2)`` array of ``(dy, dx)``.  Ordering by ``k`` makes the search
    try the smallest neighbourhood first, which is the paper's gradual growth.
    """
    offsets: list[tuple[int, int]] = []
    for k in range(k_start, k_max + 1, 2):
        r = k // 2
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy == 0 and dx == 0:
                    continue
                offsets.append((dy, dx))
    return np.asarray(offsets, dtype=np.int64)


def _nearest_direction_match(
    candidates: np.ndarray, cls: np.ndarray, offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each pixel, find the candidate neighbour with the closest direction class.

    Returns ``(best_delta, src_row, src_col)``; ``best_delta`` is 99 where no
    candidate exists and ``src_row/src_col`` are -1 there.
    """
    h, w = candidates.shape
    best_delta = np.full((h, w), 99, dtype=np.int16)
    src_r = np.full((h, w), -1, dtype=np.int32)
    src_c = np.full((h, w), -1, dtype=np.int32)
    c = candidates.astype(bool)
    for dy, dx in offsets:
        shifted = np.zeros_like(c)
        ys_dst = slice(max(0, -dy), h - max(0, dy))
        ys_src = slice(max(0, dy), h - max(0, -dy))
        xs_dst = slice(max(0, -dx), w - max(0, dx))
        xs_src = slice(max(0, dx), w - max(0, -dx))
        shifted[ys_dst, xs_dst] = c[ys_src, xs_src]
        if not shifted.any():
            continue
        shifted_cls = np.zeros_like(cls)
        shifted_cls[ys_dst, xs_dst] = cls[ys_src, xs_src]
        delta = np.abs(shifted_cls - cls)
        delta = np.minimum(delta, N_ANGLE_CLASSES - delta)
        better = shifted & (delta < best_delta)
        if better.any():
            ys, xs = np.nonzero(better)
            best_delta[ys, xs] = delta[ys, xs]
            src_r[ys, xs] = ys + dy
            src_c[ys, xs] = xs + dx
    return best_delta, src_r, src_c


# --------------------------------------------------------------------------- #
# 4. Pruning and region labelling (section 2.1, final paragraph)
# --------------------------------------------------------------------------- #
def label_regions(contour: np.ndarray, min_area: int = 1) -> LabelResult:
    """Mark enclosed regions by region growing and return contiguous territories.

    Region growing starts from every surface/void pixel not separated from the
    rest by a boundary pixel.  Each resulting component is a region; components
    smaller than ``min_area`` are discarded.  Boundary pixels are then assigned to
    the nearest region by Euclidean distance, so the returned labels tile the
    image and every pixel belongs to exactly one region.  Open curves inside a
    region do not separate it, which is exactly the pruning effect the paper
    describes ("spurious boundary pixels will fail to form a closed contour").

    Parameters
    ----------
    contour:
        Boolean boundary network.
    min_area:
        Minimum seed-component area retained.

    Returns
    -------
    LabelResult
        With ``labels`` (int32), regions, their areas and statuses
        (``"closed"`` / ``"border"``), and the raw seed image.
    """
    if contour.ndim != 2:
        raise ValueError(f"Expected a 2-D contour image, got shape {contour.shape}")
    if min_area < 1:
        raise ValueError("min_area must be >= 1")
    c = contour.astype(bool)
    if c.size == 0:
        raise ValueError("Cannot label an empty image")

    free = (~c).astype(np.uint8)
    n_comp, comp = cv2.connectedComponents(free, connectivity=4)
    comp_areas = np.bincount(comp.ravel())

    keep = [cid for cid in range(1, n_comp) if comp_areas[cid] >= min_area]
    if not keep:
        # No region survives the area filter: assign the whole image to a single
        # region rather than returning an empty labelling.
        keep = [int(np.argmax(comp_areas[1:])) + 1]

    seeds = np.zeros(c.shape, dtype=np.int32)
    statuses: list[str] = ["background"]
    for new_id, cid in enumerate(keep, start=1):
        m = comp == cid
        seeds[m] = new_id
        touches = bool(m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any())
        statuses.append("border" if touches else "closed")

    # Every pixel not already a seed (boundary pixels and the discarded specks)
    # joins the nearest seed region, so the labelling tiles the whole image.
    _, (iy, ix) = ndi.distance_transform_edt(seeds == 0, return_indices=True)
    labels = seeds[iy, ix].astype(np.int32) if not c.all() else seeds

    areas = np.bincount(labels.ravel(), minlength=len(keep) + 1).astype(np.int64)
    areas[0] = 0
    LOGGER.info("Region labelling: %d regions from %d seed components",
                len(keep), n_comp - 1)
    return LabelResult(labels, len(keep), areas, statuses, seeds)


def detect_initial_contours(
    gray: np.ndarray,
    bin_cfg: BinarizationConfig,
    contour_cfg: ContourConfig,
) -> ContourResult:
    """Run the complete section 2.1 pipeline on a grey-scale image.

    Each intermediate stage is returned so it can be inspected and saved
    independently by the caller.
    """
    if gray.ndim != 2:
        raise ValueError(f"Expected a 2-D grey image, got shape {gray.shape}")
    thr = global_threshold(gray, bin_cfg)
    thinned = thin(thr.binary)
    direction = gradient_direction(gray)
    linked, link_diag = link_edges(thinned, direction, contour_cfg)
    lab = label_regions(linked, contour_cfg.min_region_area)

    return ContourResult(
        binary=thr.binary,
        thinned=thinned,
        linked=linked,
        labels=lab.labels,
        n_regions=lab.n_regions,
        areas=lab.areas,
        statuses=lab.statuses,
        diagnostics={
            "threshold": thr.threshold,
            "threshold_mean": thr.mean,
            "threshold_std": thr.std,
            "threshold_range": [thr.range_low, thr.range_high],
            "boundary_fraction": float(thr.binary.mean()),
            "thinned_pixels": int(thinned.sum()),
            "linked_pixels": int(linked.sum()),
            "links": link_diag["links"],
            "linking_iterations": link_diag["iterations"],
            "n_regions": int(lab.n_regions),
            "border_policy": "retained",
        },
    )
