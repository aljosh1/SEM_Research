"""Figures and result export.

Class colours are fixed here and used everywhere (figures, side-by-side view,
composite overview), so a reviewer can compare panels across runs:

===== ======== =========
Class Colour   RGB
===== ======== =========
VOID  red      (220, 30, 30)
SURFACE green  (30, 160, 60)
MIXED yellow   (235, 200, 20)
===== ======== =========

Overlays blend the class colour with the grey-scale image at
``OVERLAY_ALPHA`` so the underlying SEM texture stays visible.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")  # headless-safe; works in VS Code without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .region_classification import CLASS_MIXED, CLASS_NAMES, CLASS_SURFACE, CLASS_VOID  # noqa: E402

LOGGER = logging.getLogger(__name__)

#: BGR colours for OpenCV drawing.
CLASS_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    CLASS_VOID: (30, 30, 220),
    CLASS_SURFACE: (60, 160, 30),
    CLASS_MIXED: (20, 200, 235),
}
#: RGB colours for matplotlib.
CLASS_COLORS_RGB: dict[int, tuple[float, float, float]] = {
    CLASS_VOID: (220 / 255, 30 / 255, 30 / 255),
    CLASS_SURFACE: (30 / 255, 160 / 255, 60 / 255),
    CLASS_MIXED: (235 / 255, 200 / 255, 20 / 255),
}
OVERLAY_ALPHA = 0.55

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
})


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save(fig: plt.Figure, path: Path, dpi: int, tiff: bool = False) -> None:
    fig.savefig(path, dpi=dpi, format="png")
    if tiff:
        fig.savefig(path.with_suffix(".tiff"), dpi=dpi, format="tiff")
    plt.close(fig)
    LOGGER.debug("Saved %s", path)


def _binary_rgb(mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    img = np.ones((*mask.shape, 3), dtype=float)
    img[mask.astype(bool)] = color
    return img


def save_class_overlay(
    gray: np.ndarray,
    class_image: np.ndarray,
    path: Path,
    dpi: int = 200,
    tiff: bool = False,
    void_mask: np.ndarray | None = None,
) -> None:
    """Grey image tinted by class colour (png; optional tiff).

    With ``void_mask`` the fill is pixel accurate (see :func:`class_rgb_image`).
    """
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    out = base.copy()
    if void_mask is not None:
        void_px = void_mask.astype(bool) & np.isin(class_image, (CLASS_VOID, CLASS_MIXED))
        mixed_px = (class_image == CLASS_MIXED) & ~void_px
        surface_px = class_image == CLASS_SURFACE
        for m, cls_id in ((void_px, CLASS_VOID), (mixed_px, CLASS_MIXED), (surface_px, CLASS_SURFACE)):
            if m.any():
                out[m] = ((1 - OVERLAY_ALPHA) * base[m]
                          + OVERLAY_ALPHA * np.array(CLASS_COLORS_BGR[cls_id], dtype=np.float32))
    else:
        for cls_id, color in CLASS_COLORS_BGR.items():
            m = class_image == cls_id
            if m.any():
                out[m] = (1 - OVERLAY_ALPHA) * base[m] + OVERLAY_ALPHA * np.array(color, dtype=np.float32)
    img = np.clip(out, 0, 255).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    if tiff:
        cv2.imwrite(str(Path(path).with_suffix(".tiff")), img)


def figure_original_and_histogram(
    gray: np.ndarray, hist_counts: np.ndarray, hist_centres: np.ndarray,
    threshold: float | None, title: str, path: Path, dpi: int = 200,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Grey-scale fractograph")
    axes[0].axis("off")
    axes[1].bar(hist_centres, hist_counts, width=1.0, color="0.3")
    if threshold is not None:
        axes[1].axvline(threshold, color="red", ls="--", lw=1.2,
                        label=f"th = {threshold:.1f}")
        axes[1].legend(loc="upper right")
    axes[1].set_xlabel("intensity level")
    axes[1].set_ylabel("pixel count")
    axes[1].set_title("Intensity histogram")
    fig.suptitle(title)
    _save(fig, path, dpi)


def figure_intermediates(
    gray: np.ndarray,
    binary: np.ndarray,
    thinned: np.ndarray,
    linked: np.ndarray,
    labels: np.ndarray,
    title: str,
    path: Path,
    dpi: int = 200,
) -> None:
    """Binary -> thinned -> linked -> initial regions, plus overlay."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    panels = [
        ("(a) binary (th = mu + 2 sigma)", _binary_rgb(binary, (0.1, 0.1, 0.1)), None),
        ("(b) thinned", _binary_rgb(thinned, (0.1, 0.1, 0.4)), None),
        ("(c) linked", _binary_rgb(linked, (0.6, 0.0, 0.0)), None),
        ("(d) initial region labels", labels, "nipy_spectral"),
        ("(e) initial contour on image", None, None),
    ]
    for ax, (t, data, cmap) in zip(axes, panels):
        if data is None:
            ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
            ys, xs = np.nonzero(linked)
            ax.scatter(xs, ys, s=0.05, c="red", marker="s")
        else:
            ax.imshow(data, cmap=cmap, vmin=0, vmax=None)
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    _save(fig, path, dpi)


def figure_refinement(
    gray: np.ndarray,
    initial_labels: np.ndarray,
    refined_labels: np.ndarray,
    refined_regions: list[dict[str, Any]],
    title: str,
    path: Path,
    dpi: int = 200,
) -> None:
    """Original | initial regions | refined regions, with a refinement table."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("original")
    axes[0].axis("off")
    axes[1].imshow(initial_labels, cmap="nipy_spectral")
    axes[1].set_title(f"initial regions (n={int(initial_labels.max())})")
    axes[1].axis("off")
    axes[2].imshow(refined_labels, cmap="nipy_spectral")
    axes[2].set_title(f"refined regions (n={int(refined_labels.max())})")
    axes[2].axis("off")
    if refined_regions:
        info = "\n".join(
            f"region {r['region_id']}: {r['original_area']} px -> {r['pieces']} piece(s), "
            f"T_local={r['local_threshold']:.1f}"
            for r in sorted(refined_regions, key=lambda r: -r["original_area"])[:8]
        )
        fig.text(0.01, -0.02, "Refined regions:\n" + info, va="top", fontsize=7, family="monospace")
    fig.suptitle(title)
    _save(fig, path, dpi)


def figure_void_detection(
    gray: np.ndarray,
    hist_counts: np.ndarray,
    hist_centres: np.ndarray,
    i_t: float | None,
    tip_mask: np.ndarray,
    void_mask: np.ndarray,
    title: str,
    path: Path,
    dpi: int = 200,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    axes[0].bar(hist_centres, hist_counts, width=1.0, color="0.3")
    if i_t is not None:
        axes[0].axvline(i_t, color="red", ls="--", lw=1.2, label=f"I_t = {i_t:.0f}")
        axes[0].legend()
    axes[0].set_title("(a) histogram and void-tip cutoff")
    axes[0].set_xlabel("intensity")
    axes[0].set_ylabel("count")
    axes[1].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("(b) original")
    axes[1].axis("off")
    axes[2].imshow(tip_mask, cmap="Reds")
    axes[2].set_title("(c) void-tip mask")
    axes[2].axis("off")
    ov = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    ov[void_mask] = 0.4 * ov[void_mask] + 0.6 * np.array((30, 30, 220), np.float32)
    axes[3].imshow(cv2.cvtColor(np.clip(ov, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    axes[3].set_title("(d) extended void mask")
    axes[3].axis("off")
    fig.suptitle(title)
    _save(fig, path, dpi)


def figure_final_segmentation(
    gray: np.ndarray,
    class_image: np.ndarray,
    counts: dict[str, int],
    fractions: dict[str, float],
    title: str,
    path: Path,
    dpi: int = 200,
    tiff: bool = True,
    void_mask: np.ndarray | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("original")
    axes[0].axis("off")

    rgb = class_rgb_image(gray, class_image, void_mask)
    axes[1].imshow(rgb)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS_RGB[cls_id],
                      label=f"{CLASS_NAMES[cls_id]} ({counts.get(CLASS_NAMES[cls_id], 0)} regions, "
                            f"{fractions.get(CLASS_NAMES[cls_id], 0) * 100:.1f}% area)")
        for cls_id in (CLASS_VOID, CLASS_SURFACE, CLASS_MIXED)
    ]
    axes[1].legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.85)
    axes[1].set_title("three-class segmentation (void pixels filled)")
    axes[1].axis("off")
    fig.suptitle(title)
    _save(fig, path, dpi, tiff=tiff)


def figure_side_by_side(
    gray: np.ndarray,
    class_image: np.ndarray,
    title: str,
    path: Path,
    dpi: int = 200,
    void_mask: np.ndarray | None = None,
) -> None:
    """Compact original | segmentation panel used by the CLI's quick-look export."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original SEM image")
    axes[0].axis("off")
    axes[1].imshow(class_rgb_image(gray, class_image, void_mask))
    axes[1].set_title("Segmentation: VOID / SURFACE / MIXED")
    axes[1].axis("off")
    fig.suptitle(title)
    _save(fig, path, dpi)


def figure_void_size_distribution(
    counts: np.ndarray,
    edges: np.ndarray,
    voids: list[Any],
    title: str,
    path: Path,
    dpi: int = 200,
    pixel_size_um: float | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    scale = pixel_size_um if pixel_size_um is not None else 1.0
    unit = "um" if pixel_size_um is not None else "px"
    if counts.size:
        axes[0].bar(0.5 * (edges[:-1] + edges[1:]) * scale, counts,
                    width=np.diff(edges).mean() * scale * 0.9, color="0.35")
    axes[0].set_xlabel(f"equivalent circular diameter [{unit}]")
    axes[0].set_ylabel("number of voids")
    axes[0].set_title(f"Void-size distribution (n={len(voids)})")
    if voids:
        ar = [v.aspect_ratio for v in voids if np.isfinite(v.aspect_ratio)]
        circ = [v.circularity for v in voids]
        axes[1].scatter(ar, circ, s=14, alpha=0.7, color="tab:red", edgecolors="none")
        axes[1].set_xlabel("aspect ratio (major/minor)")
        axes[1].set_ylabel("circularity (4*pi*A/P^2)")
        axes[1].set_title("Void shape: aspect ratio vs circularity")
        axes[1].grid(alpha=0.3)
    fig.suptitle(title)
    _save(fig, path, dpi)


def figure_summary(
    fractions: dict[str, float], summary: dict[str, Any], title: str, path: Path, dpi: int = 200
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = ["VOID", "MIXED", "SURFACE"]
    vals = [fractions.get(n, 0) * 100 for n in names]
    colors = [CLASS_COLORS_RGB[CLASS_VOID], CLASS_COLORS_RGB[CLASS_MIXED], CLASS_COLORS_RGB[CLASS_SURFACE]]
    axes[0].bar(names, vals, color=colors)
    axes[0].set_ylabel("area fraction [%]")
    axes[0].set_title("Class area fractions")
    for i, v in enumerate(vals):
        axes[0].text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    keys = [
        ("n_individual_voids", "individual voids"),
        ("total_void_area_px", "void area [px]"),
        ("equivalent_diameter_mean_px", "mean D_eq [px]"),
        ("equivalent_diameter_median_px", "median D_eq [px]"),
        ("aspect_ratio_mean_dimensionless", "mean aspect ratio"),
        ("circularity_mean_dimensionless", "mean circularity"),
    ]
    axes[1].axis("off")
    lines = []
    for k, label in keys:
        v = summary.get(k)
        if v is None:
            continue
        lines.append(f"{label:<24s} {v:,.2f}" if isinstance(v, float) else f"{label:<24s} {v:,}")
    axes[1].text(0.0, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=9)
    axes[1].set_title("Summary statistics")
    fig.suptitle(title)
    _save(fig, path, dpi)


def figure_composite(
    gray: np.ndarray,
    linked: np.ndarray,
    void_mask: np.ndarray,
    rgb_segmentation: np.ndarray,
    histogram: tuple[np.ndarray, np.ndarray],
    title: str,
    path: Path,
    dpi: int = 180,
) -> None:
    """One-page overview: original, contour, void mask, result, histogram."""
    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 3, hspace=0.18, wspace=0.08)
    specs = [
        (gs[0, 0], gray, "gray", None, "(a) original SEM fractograph"),
        (gs[0, 1], _binary_rgb(linked, (0.6, 0.0, 0.0)), None, None, "(b) detected contour network"),
        (gs[0, 2], None, None, None, "(c) histogram"),
        (gs[1, 0], void_mask, "Reds", None, "(d) extended void mask"),
        (gs[1, 1:], rgb_segmentation, None, None, "(e) final class segmentation"),
    ]
    for spec, data, cmap, _, ttl in specs:
        ax = fig.add_subplot(spec)
        if ttl.startswith("(c)"):
            ax.bar(histogram[1], histogram[0], width=1.0, color="0.35")
            ax.set_xlabel("intensity")
            ax.set_ylabel("count")
        else:
            ax.imshow(data, cmap=cmap)
        ax.set_title(ttl)
        ax.axis("off") if ttl != "(c)" else None
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def class_rgb_image(
    gray: np.ndarray,
    class_image: np.ndarray,
    void_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Tinted RGB segmentation.

    ``class_image`` colours whole regions.  Because a VOID region is usually only
    *partially* covered by void (measured void fraction ~0.32 on real data), the
    region fill alone would paint surface pixels red.  When ``void_mask`` is
    supplied, pixel-accurate rendering is used instead:

    * void pixels inside a VOID or MIXED region -> red;
    * MIXED region pixels that are not void -> yellow;
    * SURFACE region pixels -> green;
    * unclassified pixels -> untouched grey-scale.
    """
    base = np.stack([gray] * 3, axis=-1).astype(float) / 255.0
    rgb = base.copy()
    if void_mask is not None:
        if void_mask.shape != class_image.shape:
            raise ValueError("void_mask and class_image must have the same shape")
        void_px = void_mask.astype(bool) & np.isin(class_image, (CLASS_VOID, CLASS_MIXED))
        surface_px = (class_image == CLASS_SURFACE)
        mixed_px = (class_image == CLASS_MIXED) & ~void_px
        for m, cls_id in ((void_px, CLASS_VOID), (mixed_px, CLASS_MIXED), (surface_px, CLASS_SURFACE)):
            if m.any():
                rgb[m] = (1 - OVERLAY_ALPHA) * base[m] + OVERLAY_ALPHA * np.array(CLASS_COLORS_RGB[cls_id])
        return rgb
    for cls_id, color in CLASS_COLORS_RGB.items():
        m = class_image == cls_id
        if m.any():
            rgb[m] = (1 - OVERLAY_ALPHA) * base[m] + OVERLAY_ALPHA * np.array(color)
    return rgb
