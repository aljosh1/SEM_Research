"""Image loading and normalisation for SEM fractographs.

Design rules (see ``docs/technical_note.md``):

* The *original* array is never modified.  Grayscale conversion, bit-depth
  conversion and preprocessing all produce new arrays.
* The segmentation algorithm works on ``uint8`` intensity in 0..255, which is
  the scale used in the reference paper (histogram axis 0..250+).  Converted
  arrays are recorded in :class:`ImageRecord` together with the source range so
  that no information is silently lost.
* 16-bit TIFF is preserved verbatim in ``record.original`` and converted once,
  with the conversion factor written to the run metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

#: Extensions the loader accepts (lower-case, including the dot).
SUPPORTED_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


class ImageLoadError(RuntimeError):
    """Raised when a path cannot be turned into a usable intensity image."""


@dataclass
class ImageRecord:
    """Container describing one loaded image and its derived representations."""

    path: Path
    original: np.ndarray
    gray: np.ndarray            # uint8, 0..255 - the array used by the algorithm
    is_grayscale_source: bool
    source_dtype: str
    source_shape: tuple[int, ...]
    source_min: float
    source_max: float
    channels: int
    bit_depth: int
    conversion_note: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.gray.shape[0])

    @property
    def width(self) -> int:
        return int(self.gray.shape[1])

    def summary(self) -> dict[str, Any]:
        """Human/JSON readable description of the loaded image."""
        return {
            "path": str(self.path),
            "dimensions": {"height": self.height, "width": self.width},
            "channels": self.channels,
            "source_dtype": self.source_dtype,
            "source_shape": list(self.source_shape),
            "source_intensity_range": [self.source_min, self.source_max],
            "bit_depth": self.bit_depth,
            "is_grayscale_source": self.is_grayscale_source,
            "conversion_note": self.conversion_note,
            "algorithm_intensity_range": [int(self.gray.min()), int(self.gray.max())],
        }


def _validate_path(path: str | Path) -> Path:
    p = Path(path)
    if not str(path).strip():
        raise ImageLoadError("Empty image path supplied.")
    if not p.exists():
        raise ImageLoadError(f"Image file does not exist: {p}")
    if not p.is_file():
        raise ImageLoadError(f"Path is not a file: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(
            f"Unsupported image format '{p.suffix}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return p


def _scale_to_uint8(arr: np.ndarray, src_min: float, src_max: float) -> tuple[np.ndarray, str]:
    """Map an integer/float array linearly to uint8 without clipping real data."""
    if src_max <= src_min:
        # Uniform image: a single intensity.  Map to mid-grey deterministically.
        value = 0 if src_max <= 0 else 127
        out = np.full(arr.shape, value, dtype=np.uint8)
        return out, f"uniform image mapped to {value}"
    if arr.dtype == np.uint8:
        return arr.copy(), "8-bit, used directly"
    scale = 255.0 / (src_max - src_min)
    out = np.clip((arr.astype(np.float64) - src_min) * scale, 0, 255).astype(np.uint8)
    note = (
        f"{np.dtype(arr.dtype).name} source range [{src_min:.1f}, {src_max:.1f}] "
        f"linearly mapped to 0..255 (scale {scale:.4f} levels/level); "
        "original preserved in ImageRecord.original"
    )
    return out, note


def load_image(path: str | Path) -> ImageRecord:
    """Load an image, validate it and produce a ``uint8`` grey representation.

    Parameters
    ----------
    path:
        Image file (PNG, TIFF/TIF, JPEG/JPG, BMP).

    Returns
    -------
    ImageRecord
        With ``original`` (untouched) and ``gray`` (uint8 used by the algorithm).

    Raises
    ------
    ImageLoadError
        If the path is invalid, the extension is unsupported, or the file cannot
        be decoded.
    """
    p = _validate_path(path)

    # cv2.imread with UNCHANGED keeps 16-bit data intact (IMREAD_GRAYSCALE would
    # silently down-convert, which the brief forbids).
    original = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if original is None:
        raise ImageLoadError(f"Could not decode image (corrupt or unreadable): {p}")

    if original.ndim == 2:
        channels = 1
        is_gray = True
    elif original.ndim == 3:
        channels = int(original.shape[2])
        is_gray = channels == 1
    else:
        raise ImageLoadError(f"Unsupported array shape {original.shape} for {p}")

    src_min = float(np.min(original))
    src_max = float(np.max(original))
    bit_depth = int(np.iinfo(original.dtype).bits) if np.issubdtype(
        original.dtype, np.integer
    ) else 0

    if channels == 1:
        plane = original[:, :, 0] if original.ndim == 3 else original
        gray, note = _scale_to_uint8(plane, src_min, src_max)
    else:
        # Colour image: convert to grey.  OpenCV's BGR2GRAY uses Rec.601 weights.
        if original.dtype == np.uint8:
            planes = original[:, :, :3]
        else:
            planes, _ = _scale_to_uint8(original, src_min, src_max)
            planes = planes[:, :, :3]
        gray = cv2.cvtColor(planes, cv2.COLOR_BGR2GRAY)
        note = (
            f"colour source with {channels} channels converted with "
            "cv2.COLOR_BGR2GRAY (Rec.601); original colour data preserved"
        )
        if channels == 4:
            note += "; alpha channel ignored"

    if gray.ndim != 2:
        raise ImageLoadError(f"Grayscale conversion produced shape {gray.shape}")

    LOGGER.info(
        "Loaded %s: %dx%d, %d channel(s), dtype=%s, range [%s, %s]",
        p.name, gray.shape[1], gray.shape[0], channels,
        original.dtype, src_min, src_max,
    )

    return ImageRecord(
        path=p,
        original=original,
        gray=gray,
        is_grayscale_source=is_gray,
        source_dtype=str(original.dtype),
        source_shape=tuple(int(s) for s in original.shape),
        source_min=src_min,
        source_max=src_max,
        channels=channels,
        bit_depth=bit_depth,
        conversion_note=note,
    )


def intensity_histogram(
    image: np.ndarray, bin_width: int = 1, max_value: int = 255
) -> tuple[np.ndarray, np.ndarray]:
    """Return smoothed histogram counts and bin centres.

    The reference paper forms histograms by "dividing the intensity scale into
    bins of small size" and, for the binarization threshold, uses the *average
    frequency* per bin.  This function returns the sum per bin; dividing by
    ``bin_width`` gives the average without changing any threshold computed from
    the distribution shape.

    Parameters
    ----------
    image:
        2-D array of intensities (uint8 expected).
    bin_width:
        Width of each histogram bin in intensity levels (>= 1).
    max_value:
        Inclusive upper bound of the intensity scale.

    Returns
    -------
    (counts, centres)
        ``counts`` has length ``ceil((max_value + 1) / bin_width)``; ``centres``
        holds the integer bin centre used as the bin's representative intensity.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {image.shape}")
    if bin_width < 1:
        raise ValueError("bin_width must be >= 1")
    if image.size == 0:
        raise ValueError("Cannot build a histogram of an empty image")
    edges = np.arange(0, max_value + bin_width + 1, bin_width, dtype=np.int64)
    edges[-1] = max_value + 1
    counts, _ = np.histogram(image, bins=edges)
    centres = (edges[:-1] + edges[1:] - 1) // 2
    return counts.astype(np.int64), centres.astype(np.int64)
