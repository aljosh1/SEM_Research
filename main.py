"""Command-line application for SEM fractograph segmentation.

Run from the project root in VS Code (PowerShell):

    .\\.venv\\Scripts\\python.exe main.py --input data\\sample\\sem_fig4_panel_b.jpeg --output outputs\\run1

Adjust parameters, e.g. a stricter global threshold and a calibration:

    .\\.venv\\Scripts\\python.exe main.py --input data\\sample\\x.png --output outputs\\x \\
        --sigma-multiplier 2.5 --pixel-size-um 0.25 --calibration-source "SEM metadata"

See ``--help`` for the complete list.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import (
    PARAMETER_DOCS,
    SegmentationConfig,
    default_config,
)
from src.image_io import ImageLoadError, SUPPORTED_EXTENSIONS
from src.pipeline import run_pipeline

EPILOG = """\
examples
--------
  Run with defaults (reproduces the paper's parameter values):
    python main.py -i data/sample/sem_fig4_panel_b.jpeg -o outputs/run1

  Tune the global threshold and print the parameter reference:
    python main.py -i image.tif -o outputs/tuned --sigma-multiplier 2.5 --list-parameters

  Supply a calibration (pixel size) so physical measurements are reported:
    python main.py -i image.png -o outputs/cal --pixel-size-um 0.25 \\
        --calibration-source "verified 20 um scale bar"

  Disable all figures for a fast batch run:
    python main.py -i image.png -o outputs/fast --no-figures
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Segment SEM fractographs into VOID, SURFACE and MIXED regions using "
            "the conventional image-processing method of Banerjee et al. "
            "(hole_extract.pdf), and export masks, figures and CSV measurements."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = p.add_argument_group("input / output")
    g.add_argument("-i", "--input", required=True, type=Path,
                   help=f"input image ({', '.join(sorted(SUPPORTED_EXTENSIONS))})")
    g.add_argument("-o", "--output", default=Path("outputs/run"), type=Path,
                   help="output directory (created if missing) [default: outputs/run]")
    g.add_argument("--image-id", default=None,
                   help="identifier written into every CSV row [default: input file stem]")
    g.add_argument("--no-figures", action="store_true",
                   help="skip figure generation (faster; CSVs and masks are still written)")
    g.add_argument("--no-tiff", action="store_true",
                   help="do not write extra TIFF copies of masks/final segmentation")
    g.add_argument("-v", "--verbose", action="store_true", help="verbose logging")

    g = p.add_argument_group("preprocessing (optional, disabled by default)")
    g.add_argument("--preprocess", action="store_true",
                   help="enable preprocessing (the paper applies none)")
    g.add_argument("--gaussian-sigma", type=float, default=0.0,
                   help="Gaussian smoothing sigma in px [default: 0.0 = off]")
    g.add_argument("--median-kernel", type=int, default=0,
                   help="median filter kernel size, odd, 0 = off [default: 0]")
    g.add_argument("--percentile-clip", type=float, nargs=2, metavar=("LOW", "HIGH"),
                   default=None, help="contrast stretch percentile clip, e.g. 1 99")

    g = p.add_argument_group("stage 2.1 - global contour detection")
    g.add_argument("--sigma-multiplier", type=float, default=2.0,
                   help="th = mu + a*sigma (paper: 2.0) [default: 2.0]")
    g.add_argument("--hist-bin-width", type=int, default=5,
                   help="bin width for the smoothed histogram [default: 5]")
    g.add_argument("--extreme-low-fraction", type=float, default=0.10,
                   help="low-intensity tail excluded from mu/sigma [default: 0.10]")
    g.add_argument("--extreme-high-fraction", type=float, default=0.10,
                   help="high-intensity tail excluded from mu/sigma [default: 0.10]")
    g.add_argument("--link-k-start", type=int, default=5,
                   help="initial linking neighbourhood side (paper: 5)")
    g.add_argument("--link-k-max", type=int, default=7,
                   help="maximum linking neighbourhood side (paper: 7)")
    g.add_argument("--min-region-area", type=int, default=50,
                   help="discard regions smaller than this many pixels [default: 50]")

    g = p.add_argument_group("stage 2.2 - local refinement")
    g.add_argument("--area-sigma-coeff", type=float, default=1.0,
                   help="candidate rule S_i > mu_S + c*sigma_S (paper: 1.0)")
    g.add_argument("--local-sigma-coeff", type=float, default=0.5,
                   help="T_local = mu_l + c*sigma_l (paper: 0.5)")
    g.add_argument("--refine-link-k", type=int, default=3,
                   help="reduced linking neighbourhood during refinement [default: 3]")

    g = p.add_argument_group("stage 2.3 - void detection and classification")
    g.add_argument("--void-hist-bin-width", type=int, default=2,
                   help="bin width for the void-tip histogram [default: 2]")
    g.add_argument("--extension-sigma-coeff", type=float, default=0.5,
                   help="I_v = max(mu_l - c*sigma_l, I_t) (paper: 0.5)")
    g.add_argument("--cutoff-percentile", type=float, default=80.0,
                   help="k%% cutoff intensity I_c (paper: 80)")
    g.add_argument("--global-sigma-multiplier", type=float, default=1.0,
                   help="j in I_c < mu_g + j*sigma_g (paper: 1)")
    g.add_argument("--percentile-method", default="inverted_cdf",
                   choices=["inverted_cdf", "linear", "lower", "higher",
                            "midpoint", "nearest"],
                   help="numpy percentile method for I_c [default: inverted_cdf]")

    g = p.add_argument_group("calibration (optional)")
    g.add_argument("--pixel-size-um", type=float, default=None,
                   help="physical pixel size in micrometres; omit for pixel-only results")
    g.add_argument("--calibration-source", default="unavailable",
                   help="provenance of the pixel size, e.g. 'verified scale bar'")

    g = p.add_argument_group("reference")
    g.add_argument("--list-parameters", action="store_true",
                   help="print the documented parameter table and exit")
    return p


def config_from_args(args: argparse.Namespace) -> SegmentationConfig:
    """Build a validated configuration from parsed arguments."""
    cfg = default_config()
    cfg.preprocessing.enabled = bool(args.preprocess)
    cfg.preprocessing.gaussian_sigma = args.gaussian_sigma
    cfg.preprocessing.median_kernel = args.median_kernel
    if args.percentile_clip is not None:
        lo, hi = args.percentile_clip
        cfg.preprocessing.percentile_clip = (float(lo), float(hi))

    cfg.binarization.sigma_multiplier = args.sigma_multiplier
    cfg.binarization.hist_bin_width = args.hist_bin_width
    cfg.binarization.extreme_low_fraction = args.extreme_low_fraction
    cfg.binarization.extreme_high_fraction = args.extreme_high_fraction

    cfg.contour.link_k_start = args.link_k_start
    cfg.contour.link_k_max = args.link_k_max
    cfg.contour.min_region_area = args.min_region_area
    cfg.contour.area_sigma_coeff = args.area_sigma_coeff
    cfg.contour.local_sigma_coeff = args.local_sigma_coeff
    cfg.contour.refine_link_k = args.refine_link_k

    cfg.void.hist_bin_width = args.void_hist_bin_width
    cfg.void.extension_sigma_coeff = args.extension_sigma_coeff

    cfg.classification.cutoff_percentile = args.cutoff_percentile
    cfg.classification.global_sigma_multiplier = args.global_sigma_multiplier
    cfg.classification.percentile_method = args.percentile_method

    cfg.calibration.pixel_size_um = args.pixel_size_um
    cfg.calibration.source = args.calibration_source

    cfg.validate()
    return cfg


def print_parameter_table() -> None:
    print(f"{'parameter':<42} {'default':<14} {'unit':<12} provenance")
    print("-" * 110)
    for name, (purpose, default, unit, constraint, provenance, _effect) in PARAMETER_DOCS.items():
        print(f"{name:<42} {str(default):<14} {unit:<12} {provenance}")
        print(f"    purpose: {purpose}")
        print(f"    valid:   {constraint}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    if args.list_parameters:
        print_parameter_table()
        return 0

    if not args.input.exists():
        parser.error(f"input image not found: {args.input}")

    try:
        cfg = config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    print("=" * 72)
    print("SEM fractograph segmentation - Banerjee et al. methodology")
    print("=" * 72)
    print(f"input      : {args.input}")
    print(f"output     : {args.output}")
    print(f"preprocess : {'enabled' if cfg.preprocessing.enabled else 'disabled'}")
    print(f"calibration: {'pixel size %.4g um (%s)' % (cfg.calibration.pixel_size_um, cfg.calibration.source) if cfg.calibration.pixel_size_um else 'not supplied - pixel units only'}")
    print("-" * 72)

    try:
        result = run_pipeline(
            image_path=args.input,
            out_dir=args.output,
            cfg=cfg,
            image_id=args.image_id,
            export_tiff=not args.no_tiff,
            make_figures=not args.no_figures,
        )
    except ImageLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"ERROR: invalid input or parameters: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - unexpected failures must be visible
        print(f"ERROR: processing failed: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        return 1

    s = result.summary
    print()
    print("RESULTS")
    print(f"  regions classified : VOID={s.get('void_region_area_px', 0)} px, "
          f"MIXED={s.get('mixed_region_area_px', 0)} px, "
          f"SURFACE={s.get('surface_region_area_px', 0)} px")
    print(f"  area fractions     : VOID {s['void_area_fraction'] * 100:.2f}%  "
          f"MIXED {s['mixed_area_fraction'] * 100:.2f}%  "
          f"SURFACE {s['surface_area_fraction'] * 100:.2f}%")
    print(f"  individual voids   : {s['n_individual_voids']}")
    print(f"  total void area    : {s['total_void_area_px']:,} px"
          + (f"  ({s.get('total_void_area_um2', 0):,.2f} um^2)" if cfg.calibration.pixel_size_um else ""))
    if s.get("equivalent_diameter_mean_px") is not None:
        print(f"  mean void D_eq     : {s['equivalent_diameter_mean_px']:.2f} px"
              + (f"  ({s.get('equivalent_diameter_um_mean_um', 0):.2f} um)" if cfg.calibration.pixel_size_um else ""))
    if not cfg.calibration.pixel_size_um:
        print("  physical units     : unavailable (no calibration supplied)")
    if result.warnings:
        print()
        for w in result.warnings:
            print(f"  WARNING: {w}")

    print()
    print(f"Run completed. {len(result.files)} files written.")
    print(f"Output directory: {result.out_dir.resolve()}")
    print("  masks/         : void-tip mask, extended void mask, class image, overlay")
    print("  contours/      : binary, thinned, linked, initial and refined labels")
    print("  figures/       : histogram, intermediate stages, final segmentation, summary")
    print("  measurements/  : void_measurements.csv, region_measurements.csv, summary_statistics.csv")
    print("  run_metadata.json : parameters, thresholds, warnings and output list")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
