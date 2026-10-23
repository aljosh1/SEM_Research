"""Image loading, grayscale conversion and histogram generation."""

from __future__ import annotations

import numpy as np
import pytest

from src.image_io import (
    ImageLoadError,
    intensity_histogram,
    load_image,
)


def test_loads_grayscale_png_and_preserves_original(tmp_image, synthetic_field):
    path = tmp_image(synthetic_field, "g.png")
    rec = load_image(path)
    assert rec.gray.shape == synthetic_field.shape
    assert rec.gray.dtype == np.uint8
    assert rec.channels == 1
    assert rec.is_grayscale_source
    # original must be untouched
    np.testing.assert_array_equal(rec.original, synthetic_field)


def test_reports_dimensions_and_intensity_range(tmp_image, synthetic_field):
    rec = load_image(tmp_image(synthetic_field, "g.png"))
    s = rec.summary()
    assert s["dimensions"] == {"height": synthetic_field.shape[0], "width": synthetic_field.shape[1]}
    assert s["source_intensity_range"] == [float(synthetic_field.min()), float(synthetic_field.max())]


def test_converts_colour_image_to_grayscale(tmp_image):
    colour = np.zeros((32, 48, 3), dtype=np.uint8)
    colour[:, :, 2] = 255  # red in BGR
    rec = load_image(tmp_image(colour, "c.png"))
    assert rec.channels == 3
    assert not rec.is_grayscale_source
    assert rec.gray.shape == (32, 48)
    # red -> Rec.601 luma ~76
    assert 60 < int(rec.gray[0, 0]) < 90
    assert rec.original.shape == (32, 48, 3)


def test_loads_jpeg_and_tiff(tmp_image, synthetic_field):
    for name in ("a.jpg", "b.tiff", "c.tif", "d.png"):
        rec = load_image(tmp_image(synthetic_field, name))
        assert rec.gray.ndim == 2


def test_16bit_tiff_is_preserved_and_scaled(tmp_image):
    a16 = (np.linspace(0, 4095, 64 * 64).reshape(64, 64)).astype(np.uint16)
    rec = load_image(tmp_image(a16, "deep.tiff"))
    assert rec.source_dtype == "uint16"
    assert rec.bit_depth == 16
    assert rec.original.dtype == np.uint16
    assert rec.gray.dtype == np.uint8
    assert rec.gray.min() == 0 and rec.gray.max() == 255
    assert "linearly mapped" in rec.conversion_note


def test_missing_file_raises():
    with pytest.raises(ImageLoadError, match="does not exist"):
        load_image("does/not/exist.png")


def test_unsupported_extension_raises(tmp_path):
    p = tmp_path / "x.gif"
    p.write_bytes(b"GIF89a")
    with pytest.raises(ImageLoadError, match="Unsupported"):
        load_image(p)


def test_empty_path_raises():
    with pytest.raises(ImageLoadError):
        load_image("")


def test_corrupt_file_raises(tmp_path):
    p = tmp_path / "corrupt.png"
    p.write_bytes(b"not an image")
    with pytest.raises(ImageLoadError, match="Could not decode"):
        load_image(p)


def test_uniform_image_loads_without_error(tmp_image, uniform_image):
    rec = load_image(tmp_image(uniform_image, "u.png"))
    assert rec.gray.min() == rec.gray.max()


def test_histogram_counts_sum_to_pixel_count(synthetic_field):
    counts, centres = intensity_histogram(synthetic_field, bin_width=1)
    assert counts.sum() == synthetic_field.size
    assert len(counts) == len(centres) == 256


def test_histogram_bin_width_groups_levels(synthetic_field):
    counts4, centres4 = intensity_histogram(synthetic_field, bin_width=4)
    assert len(counts4) == 64
    assert counts4.sum() == synthetic_field.size
    assert np.all(np.diff(centres4) == 4)


def test_histogram_rejects_bad_input():
    with pytest.raises(ValueError):
        intensity_histogram(np.zeros((2, 2, 3), np.uint8))
    with pytest.raises(ValueError):
        intensity_histogram(np.zeros((0, 0), np.uint8))
    with pytest.raises(ValueError):
        intensity_histogram(np.zeros((4, 4), np.uint8), bin_width=0)
