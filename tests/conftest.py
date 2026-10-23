"""Shared fixtures.

Synthetic images here exist only to test *software behaviour*.  They are not
SEM data and no scientific claim is made from them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_SAMPLE = ROOT / "data" / "sample"


@pytest.fixture
def tmp_image(tmp_path):
    """Factory: write a grey array to a temporary PNG/TIFF/JPEG and return path."""

    def _make(arr: np.ndarray, name: str = "img.png") -> Path:
        p = tmp_path / name
        cv2.imwrite(str(p), arr)
        return p

    return _make


@pytest.fixture
def synthetic_field() -> np.ndarray:
    """Dark background, mid-grey plateaus and bright rims.

    Structurally similar to a fractograph (bright boundaries, darker
    interiors) so that thresholding and region labelling can be exercised.
    """
    h, w = 240, 320
    img = np.full((h, w), 40, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (140, 110), 80, -1)
    cv2.rectangle(img, (180, 130), (300, 220), 70, -1)
    cv2.rectangle(img, (20, 20), (140, 110), 200, 2)
    cv2.rectangle(img, (180, 130), (300, 220), 200, 2)
    cv2.circle(img, (80, 170), 30, 10, -1)
    cv2.circle(img, (80, 170), 30, 210, 2)
    noise = np.random.default_rng(0).integers(-4, 5, size=img.shape)
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def uniform_image() -> np.ndarray:
    return np.full((64, 64), 128, dtype=np.uint8)


@pytest.fixture
def real_image_path() -> Path:
    """A real SEM fractograph extracted from the reference paper, if present."""
    p = DATA_SAMPLE / "sem_dimples_deep.jpeg"
    if not p.exists():
        pytest.skip("real SEM sample not available")
    return p
