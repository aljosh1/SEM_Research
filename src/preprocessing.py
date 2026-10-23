"""Optional preprocessing of the grey-scale fractograph.

The reference paper applies **no** preprocessing: the raw grey-scale image is
binarized directly (section 2.1).  Therefore preprocessing is disabled by
default.  When enabled, every operation is recorded in a provenance dict so the
run metadata states exactly which image the algorithm saw.

Conservative choices are deliberate: the void-detection logic depends on the
intensity gradient from void bottom to surface, and aggressive smoothing or
contrast clipping destroys that gradient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .config import PreprocessingConfig


@dataclass
class PreprocessResult:
    image: np.ndarray                       # uint8 0..255
    operations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def applied(self) -> bool:
        return bool(self.operations)


def preprocess(gray: np.ndarray, cfg: PreprocessingConfig) -> PreprocessResult:
    """Apply the configured operations in a fixed order.

    Order: median filter -> Gaussian smoothing -> percentile contrast stretch.
    Each step is skipped when its parameter is at the neutral value, so a
    disabled configuration returns a copy of the input unchanged.

    Parameters
    ----------
    gray:
        2-D ``uint8`` grey image (0..255).
    cfg:
        Preprocessing settings.

    Returns
    -------
    PreprocessResult
        Processed image plus a list of applied operations.

    Raises
    ------
    ValueError
        If ``gray`` is not a 2-D array or contains values outside 0..255.
    """
    if gray.ndim != 2:
        raise ValueError(f"Expected 2-D grey image, got shape {gray.shape}")
    if gray.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {gray.dtype}")
    if gray.size == 0:
        raise ValueError("Cannot preprocess an empty image")

    result = PreprocessResult(image=gray.copy())
    if not cfg.enabled:
        return result

    if cfg.median_kernel and cfg.median_kernel >= 3:
        k = int(cfg.median_kernel)
        if k % 2 == 0:
            raise ValueError("median_kernel must be odd")
        result.image = cv2.medianBlur(result.image, k)
        result.operations.append({"operation": "median_blur", "kernel": k})

    if cfg.gaussian_sigma > 0:
        sigma = float(cfg.gaussian_sigma)
        result.image = cv2.GaussianBlur(result.image, (0, 0), sigmaX=sigma)
        result.operations.append({"operation": "gaussian_blur", "sigma": sigma})

    if cfg.percentile_clip is not None:
        lo_p, hi_p = cfg.percentile_clip
        if not (0 <= lo_p < hi_p <= 100):
            raise ValueError("percentile_clip must satisfy 0 <= lo < hi <= 100")
        lo, hi = np.percentile(result.image, [lo_p, hi_p])
        if hi > lo:
            stretched = (result.image.astype(np.float64) - lo) * (255.0 / (hi - lo))
            result.image = np.clip(stretched, 0, 255).astype(np.uint8)
            result.operations.append(
                {"operation": "percentile_stretch", "low": lo_p, "high": hi_p,
                 "intensity_low": float(lo), "intensity_high": float(hi)}
            )

    return result
