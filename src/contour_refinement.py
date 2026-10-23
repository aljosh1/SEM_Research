"""Refinement of region contour - reference paper section 2.2.

The section 2.1 contour relies on a *global* threshold, so local boundary
segments that fail the global threshold are lost and neighbouring regions merge
into oversized ones.  Section 2.2 re-examines only the large regions:

* Candidate selection: with ``mu_S`` and ``sigma_S`` the mean and standard
  deviation of the extracted **region sizes** and ``S_i`` the size of region *i*,
  region *i* is a candidate when ``S_i > mu_S + sigma_S``.
* For each candidate the section 2.1 steps are repeated *inside the region* with
  a local threshold ``T_local = mu_l + 0.5 * sigma_l``, where ``mu_l`` and
  ``sigma_l`` are the mean and standard deviation of the intensity **in the
  candidate region**.
* During linking the neighbourhood size is reduced, "to restrict the over
  splitting of the region".  Spurious boundary pixels cannot form a closed
  contour and are pruned.

Implementation notes
--------------------
* The candidate rule uses region area in pixels (the paper's "size").
* The image-border region is retained but *excluded from the candidate
  statistics*: it is a measurement artefact of the finite field of view and would
  otherwise dominate ``mu_S`` and ``sigma_S``.  This is documented as a policy
  choice (``border_policy``) because the paper does not state it.  When the
  parameter ``exclude_border_from_statistics`` is False the paper's formula is
  applied to all regions verbatim.
* Refinement can only *split* a candidate; it never deletes a region or renames
  other regions, so the global label structure stays valid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .config import ContourConfig
from .contour_detection import (
    link_edges,
    neighbourhood_offsets,
    thin,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class RefinementResult:
    """Outcome of the local refinement pass."""

    labels: np.ndarray                  # int32 - refined region labels, 1..N
    n_regions: int
    areas: np.ndarray                   # index 0 unused
    statuses: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    #: Regions that were selected and the local threshold applied to each.
    refined_regions: list[dict[str, Any]] = field(default_factory=list)


def region_area_statistics(
    areas: np.ndarray, statuses: list[str], exclude_border: bool = True
) -> tuple[float, float]:
    """Mean and standard deviation of region areas.

    Parameters
    ----------
    areas:
        Area per label id, ``areas[0]`` unused.
    statuses:
        Status per label id (``"closed"`` / ``"border"``).
    exclude_border:
        Leave the image-frame region out of the statistics.

    Returns
    -------
    (mu_S, sigma_S)
        Zero standard deviation is possible (e.g. a single region).
    """
    ids = [i for i in range(1, len(areas)) if areas[i] > 0]
    if exclude_border:
        ids = [i for i in ids if statuses[i] != "border"]
    if not ids:
        return 0.0, 0.0
    vals = np.asarray([areas[i] for i in ids], dtype=np.float64)
    return float(vals.mean()), float(vals.std())


def select_candidates(
    areas: np.ndarray, statuses: list[str], cfg: ContourConfig,
    exclude_border: bool = True,
) -> tuple[list[int], float, float, float]:
    """Apply the paper's rule ``S_i > mu_S + sigma_S``.

    Returns
    -------
    (candidate_ids, mu_S, sigma_S, cut_off)
        ``cut_off = mu_S + sigma_S``.  An empty area list yields no candidates.
    """
    mu, sd = region_area_statistics(areas, statuses, exclude_border)
    cut = mu + cfg.area_sigma_coeff * sd
    # The paper's condition is strict: S_i > mu_S + sigma_S.
    ids = [i for i in range(1, len(areas)) if areas[i] > cut and areas[i] > 0]
    return ids, mu, sd, cut


def local_threshold(region_pixels: np.ndarray, cfg: ContourConfig) -> float:
    """``T_local = mu_l + coeff * sigma_l`` over the intensities in the region.

    Parameters
    ----------
    region_pixels:
        1-D array of intensities of the candidate region.

    Returns
    -------
    float
        The local threshold.  For a constant-intensity region ``sigma_l == 0`` and
        the threshold equals the region intensity, so no pixel exceeds it.
    """
    if region_pixels.size == 0:
        raise ValueError("Candidate region contains no pixels")
    mu = float(region_pixels.mean())
    sd = float(region_pixels.std())
    return mu + cfg.local_sigma_coeff * sd


def _refine_one_region(
    gray: np.ndarray,
    region_mask: np.ndarray,
    cfg: ContourConfig,
    direction: np.ndarray | None = None,
) -> tuple[np.ndarray, float, int]:
    """Split one candidate region using local thresholding.

    Returns ``(sub_labels, t_local, n_pieces)`` where ``sub_labels`` is 0 outside
    the region and 1..n_pieces inside it.  A region that does not split is
    returned whole as a single piece.

    ``direction`` is the image's Sobel direction.  It depends only on the image,
    not on the region, so the caller computes it once and passes it in;
    recomputing it per region dominated runtime.
    """
    values = gray[region_mask]
    t_local = local_threshold(values, cfg)

    local_binary = np.zeros(gray.shape, dtype=bool)
    local_binary[region_mask] = values > t_local
    if not local_binary.any():
        return region_mask.astype(np.int32), t_local, 1

    thinned = thin(local_binary)
    if direction is None:
        from .contour_detection import gradient_direction
        direction = gradient_direction(gray)
    # The reduced neighbourhood prevents over-splitting (section 2.2).
    linked, _ = link_edges(
        thinned, direction, cfg,
        k_start=cfg.refine_link_k,
        k_max=cfg.refine_link_k,
    )

    # Prune: only boundary curves that actually close a region survive.  Interior
    # is the region minus its local boundary; region growing then yields the
    # pieces.  Use 4-connectivity so a one-pixel diagonal gap does not merge.
    #
    # Work inside the region's bounding box: the distance transform and the
    # component labelling are then O(region), not O(image).  On a real panel this
    # reduced the refinement stage from 12.0 s to ~1 s with identical output.
    rows = np.flatnonzero(region_mask.any(axis=1))
    cols = np.flatnonzero(region_mask.any(axis=0))
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    m_roi = region_mask[r0:r1, c0:c1]
    l_roi = linked[r0:r1, c0:c1]

    interior = m_roi & ~l_roi
    n_pieces, comp = _label_within(interior, m_roi, cfg.min_region_area)
    if n_pieces <= 1:
        return region_mask.astype(np.int32), t_local, 1

    # Territory growth inside the region so the pieces tile it exactly.
    seeds = np.where(interior, comp, 0).astype(np.int32)
    if seeds.max() == 0:
        return region_mask.astype(np.int32), t_local, 1
    _, (iy, ix) = ndi.distance_transform_edt(seeds == 0, return_indices=True)
    nearest = seeds[iy, ix]
    sub = np.zeros(gray.shape, dtype=np.int32)
    roi_out = sub[r0:r1, c0:c1]
    roi_out[m_roi] = nearest[m_roi]
    sub[r0:r1, c0:c1] = roi_out
    pieces = int(len(np.unique(sub[region_mask])))
    return sub, t_local, pieces


def _label_within(
    interior: np.ndarray, region_mask: np.ndarray, min_area: int
) -> tuple[int, np.ndarray]:
    """Connected components of ``interior`` restricted to ``region_mask``."""
    import cv2

    n, comp = cv2.connectedComponents(interior.astype(np.uint8), connectivity=4)
    areas = np.bincount(comp.ravel())
    keep = [c for c in range(1, n) if areas[c] >= min_area]
    out = np.zeros(comp.shape, dtype=np.int32)
    for new_id, c in enumerate(keep, start=1):
        out[comp == c] = new_id
    return len(keep), out


def refine_contours(
    gray: np.ndarray,
    labels: np.ndarray,
    areas: np.ndarray,
    statuses: list[str],
    cfg: ContourConfig,
    exclude_border_from_statistics: bool = True,
) -> RefinementResult:
    """Run section 2.2 over an initial labelling.

    Parameters
    ----------
    gray:
        Grey-scale image (uint8).
    labels:
        Initial int32 labelling, 0 = boundary, 1..N regions.
    areas:
        Area per label id.
    statuses:
        Status per label id.
    cfg:
        Contour settings, including the local sigma coefficient and the reduced
        linking neighbourhood.
    exclude_border_from_statistics:
        Leave the frame-touching region out of ``mu_S``/``sigma_S``.

    Returns
    -------
    RefinementResult
        New contiguous labelling plus a per-region refinement record.
    """
    if gray.ndim != 2 or labels.shape != gray.shape:
        raise ValueError("gray and labels must be 2-D arrays of equal shape")
    if len(areas) != len(statuses):
        raise ValueError("areas and statuses must have one entry per label id")

    candidates, mu_s, sigma_s, cut = select_candidates(
        areas, statuses, cfg, exclude_border_from_statistics
    )

    # The Sobel direction depends only on ``gray``; compute it once for all
    # candidates instead of once per region.
    direction = None
    if candidates:
        from .contour_detection import gradient_direction
        direction = gradient_direction(gray)

    new_labels = np.zeros(labels.shape, dtype=np.int32)
    new_areas = [0]
    new_statuses = ["background"]
    refined_records: list[dict[str, Any]] = []
    next_id = 1

    for region_id in range(1, len(areas)):
        mask = labels == region_id
        if not mask.any():
            continue
        if region_id not in candidates:
            new_labels[mask] = next_id
            new_areas.append(int(mask.sum()))
            new_statuses.append(statuses[region_id])
            next_id += 1
            continue

        sub, t_local, pieces = _refine_one_region(gray, mask, cfg, direction=direction)
        piece_ids = [int(p) for p in np.unique(sub[mask]) if p > 0]
        # Deterministic ordering so repeated runs give the same labels.
        for p in piece_ids:
            pm = mask & (sub == p)
            new_labels[pm] = next_id
            new_areas.append(int(pm.sum()))
            new_statuses.append("refined")
            next_id += 1
        refined_records.append({
            "region_id": int(region_id),
            "original_area": int(mask.sum()),
            "local_threshold": float(t_local),
            "regional_mean": float(gray[mask].mean()),
            "regional_std": float(gray[mask].std()),
            "pieces": len(piece_ids),
        })
        LOGGER.info("Refined region %d (area %d) into %d piece(s) at T_local=%.2f",
                    region_id, int(mask.sum()), len(piece_ids), t_local)

    res = RefinementResult(
        labels=new_labels,
        n_regions=next_id - 1,
        areas=np.asarray(new_areas, dtype=np.int64),
        statuses=new_statuses,
        refined_regions=refined_records,
        diagnostics={
            "mu_S": mu_s,
            "sigma_S": sigma_s,
            "candidate_cut_off": cut,
            "candidate_rule": f"S_i > mu_S + {cfg.area_sigma_coeff} * sigma_S",
            "n_candidates": len(candidates),
            "border_excluded_from_statistics": exclude_border_from_statistics,
            "n_regions_before": int(len(areas) - 1),
            "n_regions_after": int(next_id - 1),
        },
    )
    LOGGER.info("Refinement: %d candidates computed, %d regions refined, %d -> %d regions",
                len(candidates), len(refined_records), len(areas) - 1, next_id - 1)
    return res
