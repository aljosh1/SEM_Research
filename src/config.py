"""Configuration for SEM fractograph segmentation.

Every parameter that influences the segmentation is declared here together with
its provenance.  The provenance is one of:

``PAPER``
    The value is stated in the reference paper (Banerjee et al., "Segmentation
    of SEM Fractograph into Void and Surface Regions", section 2).  The paper
    section is recorded in ``paper_ref``.
``IMPLEMENTATION``
    The paper specifies the *operation* but not the exact numeric setting; the
    value below is an engineering default chosen for a faithful reproduction and
    is explicitly labelled as such.  These are the only places where tuning is
    permitted without changing the published method.
``EXTENSION``
    Optional functionality that is *not* part of the paper (for example
    preprocessing) and is disabled by default or clearly isolated.

The class identifiers themselves are documented as an engineering decision: the
paper names three classes (void region, surface, mixed region) but assigns no
integers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


#: Provenance tags used by :data:`PARAMETER_DOCS`.
PAPER = "paper"
IMPLEMENTATION = "implementation"
EXTENSION = "extension"


@dataclass
class PreprocessingConfig:
    """Optional preprocessing.  Disabled by default (the paper applies none)."""

    enabled: bool = False
    #: Gaussian smoothing sigma in pixels applied to a copy of the image.
    gaussian_sigma: float = 0.0
    #: Median filter kernel size (odd); 0 or 1 disables.
    median_kernel: int = 0
    #: Clip limits for contrast stretching on percentiles, e.g. (1, 99).
    percentile_clip: tuple[float, float] | None = None


@dataclass
class BinarizationConfig:
    """Section 2.1 - binarization of the grey-scale fractograph."""

    #: th = mu + 2 sigma, taken over the intermediate histogram range.
    sigma_multiplier: float = 2.0
    #: Intensity scale is divided into bins of small width (sec. 2.1).
    hist_bin_width: int = 5
    #: Fraction of the scale treated as "extreme low" band (excluded from mu, sigma).
    extreme_low_fraction: float = 0.10
    #: Fraction of the scale treated as "extreme high" band (excluded).
    extreme_high_fraction: float = 0.10


@dataclass
class ContourConfig:
    """Section 2.1 (continued) + 2.2 - initial contour and refinement."""

    #: Start neighbourhood side k of the k x k linking search (paper: k = 5).
    link_k_start: int = 5
    #: Maximum neighbourhood side (paper: gradually increased up to 7).
    link_k_max: int = 7
    #: Maximum linking iterations (paper: iterate until no growth).
    link_max_iterations: int = 25
    #: Max gradient-direction difference (radians) allowed when linking.
    #: Paper says "closest gradient direction"; a cap of pi/2 is an
    #: engineering guard against joining orthogonal features.
    link_max_angle_rad: float = 1.5708
    #: Neighbourhood for local linking during refinement.  The paper states
    #: "during linking, neighbourhood size is reduced to restrict over splitting"
    #: without giving the number; 3 is the smallest meaningful reduction from 5.
    refine_link_k: int = 3
    #: Regional threshold T_local = mu_l + coeff * sigma_l (paper: 0.5).
    local_sigma_coeff: float = 0.5
    #: Candidate region rule S_i > mu_S + sigma_S (paper: coefficient 1.0).
    area_sigma_coeff: float = 1.0
    #: Regions smaller than this many pixels are discarded as noise specks.
    min_region_area: int = 50


@dataclass
class VoidConfig:
    """Section 2.3 - void tip detection and extension."""

    #: Bin width for the regional/global intensity histogram (small bins).
    hist_bin_width: int = 2
    #: Void-extension threshold I_v = max(mu_l - coeff*sigma_l, I_t) (paper .5).
    extension_sigma_coeff: float = 0.5
    #: Restrict extension to pixels inside the region (paper: local rule).
    restrict_to_region: bool = True


@dataclass
class ClassificationConfig:
    """Section 2.3 - region classification."""

    #: k percent cut-off used for I_c (paper: k = 80).
    cutoff_percentile: float = 80.0
    #: j used in mu_g + j*sigma_g (paper: j = 1).
    global_sigma_multiplier: float = 1.0
    #: Interpolation used for the k% cut-off. "inverted_cdf" is the strict
    #: definition: the smallest intensity such that >= k% of pixels lie below it.
    percentile_method: str = "inverted_cdf"


#: Documented parameter table: name -> (purpose, default, unit, constraint,
#: provenance, expected effect).  Kept next to the dataclasses so the README,
#: the technical note and the run metadata never drift apart.
PARAMETER_DOCS: dict[str, tuple[str, Any, str, str, str, str]] = {
    "preprocessing.enabled": (
        "Enable optional smoothing/contrast preprocessing",
        False, "bool", "True/False", EXTENSION,
        "Paper applies no preprocessing; enabling may blur weak boundaries.",
    ),
    "preprocessing.gaussian_sigma": (
        "Gaussian smoothing sigma", 0.0, "px", ">= 0", EXTENSION,
        "Higher values suppress noise but destroy the intensity gradient in voids.",
    ),
    "preprocessing.median_kernel": (
        "Median filter kernel", 0, "px", "odd, 0 or >= 3", EXTENSION,
        "Removes salt-and-pepper speckle; can erase thin boundary pixels.",
    ),
    "binarization.sigma_multiplier": (
        "Coefficient multiplying sigma in th = mu + a*sigma", 2.0, "-", "> 0",
        PAPER + " (sec. 2.1: th = mu + 2*sigma)",
        "Higher -> fewer boundary pixels, porous contours; lower -> false contours.",
    ),
    "binarization.hist_bin_width": (
        "Bin width for smoothed histogram", 5, "intensity levels", ">= 1",
        "\"bins of small width\" (sec. 2.1, number not given)",
        "Larger bins smooth jaggedness but bias mu, sigma.",
    ),
    "binarization.extreme_low_fraction": (
        "Low-intensity fraction excluded from mu/sigma", 0.10, "fraction", "0-0.5",
        "\"ignoring the peaks (if any) in two extreme intensity ranges\" (2011 paper)",
        "Excluding the dark void-tip mode keeps mu, sigma representative of the body.",
    ),
    "binarization.extreme_high_fraction": (
        "High-intensity fraction excluded from mu/sigma", 0.10, "fraction", "0-0.5",
        "same as above", "Excluding the bright boundary mode.",
    ),
    "contour.link_k_start": (
        "Initial linking neighbourhood side", 5, "px", "odd, >= 3",
        PAPER + " (sec. 2.1: k = 5)", "Larger -> links across wider gaps, more false closure.",
    ),
    "contour.link_k_max": (
        "Maximum linking neighbourhood side", 7, "px", "odd, >= link_k_start",
        PAPER + " (sec. 2.1: up to 7)", "Larger -> fewer breaks, more spurious links.",
    ),
    "contour.link_max_iterations": (
        "Maximum linking iterations", 25, "iterations", ">= 1",
        "paper: 'till no further growth' (cap is an implementation safeguard)",
        "Affects only runtime once convergence is reached.",
    ),
    "contour.link_max_angle_rad": (
        "Max gradient-direction difference for linking", 1.5708, "rad", "0..pi",
        IMPLEMENTATION, "Smaller -> straighter, more broken contours.",
    ),
    "contour.refine_link_k": (
        "Neighbourhood side used during refinement linking", 3, "px", "odd, >= 3",
        "paper: 'neighbourhood size is reduced' (value not stated)",
        "Smaller -> prevents over-splitting of large regions.",
    ),
    "contour.local_sigma_coeff": (
        "Coefficient in T_local = mu_l + c*sigma_l", 0.5, "-", ">= 0",
        PAPER + " (sec. 2.2: mu_l + .5 sigma_l)",
        "Higher -> fewer refined boundaries; lower -> over-segmentation.",
    ),
    "contour.area_sigma_coeff": (
        "Coefficient in candidate rule S_i > mu_S + c*sigma_S", 1.0, "-", ">= 0",
        PAPER + " (sec. 2.2: S_i > mu_S + sigma_S)",
        "Higher -> fewer regions refined.",
    ),
    "contour.min_region_area": (
        "Minimum region area kept", 50, "px^2", ">= 1",
        IMPLEMENTATION, "Discards single-pixel speckle regions.",
    ),
    "void.hist_bin_width": (
        "Bin width for void-tip histogram", 2, "intensity levels", ">= 1",
        "\"bins of small size\" (sec. 2.3, number not given)",
        "Coarser bins merge the low-intensity mode.",
    ),
    "void.extension_sigma_coeff": (
        "Coefficient in I_v = max(mu_l - c*sigma_l, I_t)", 0.5, "-", ">= 0",
        PAPER + " (sec. 2.3: mu_l - .5 sigma_l)",
        "Higher c -> tighter void extension.",
    ),
    "void.restrict_to_region": (
        "Restrict void extension to the parent region", True, "bool", "True/False",
        IMPLEMENTATION, "False can leak void pixels into neighbouring regions.",
    ),
    "classification.cutoff_percentile": (
        "k% cut-off intensity I_c", 80.0, "percent", "0-100",
        PAPER + " (sec. 2.3: k = 80)", "Higher k -> more regions qualify as void.",
    ),
    "classification.global_sigma_multiplier": (
        "j in the rule I_c < mu_g + j*sigma_g", 1.0, "-", ">= 0",
        PAPER + " (sec. 2.3: j = 1)", "Higher j -> more regions qualify as void.",
    ),
    "classification.percentile_method": (
        "Percentile definition for I_c", "inverted_cdf", "-",
        "numpy method name",
        "Paper gives no library; inverted_cdf matches the strict wording.",
        "Alternative methods shift thresholds by at most one intensity level.",
    ),
}


@dataclass
class CalibrationConfig:
    """Optional physical calibration.

    The paper reports no pixel-to-micrometre scale, so physical measurements
    stay unavailable unless the user supplies a value.
    """

    pixel_size_um: float | None = None
    source: str = "unavailable"


@dataclass
class OutputConfig:
    fig_dpi: int = 200
    save_tiff: bool = True
    class_colors: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: {
            "VOID": (220, 30, 30),
            "SURFACE": (30, 160, 60),
            "MIXED": (235, 200, 20),
        }
    )


@dataclass
class SegmentationConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    binarization: BinarizationConfig = field(default_factory=BinarizationConfig)
    contour: ContourConfig = field(default_factory=ContourConfig)
    void: VoidConfig = field(default_factory=VoidConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable flat-or-nested dict of all settings."""
        return asdict(self)

    def validate(self) -> None:
        """Raise :class:`ValueError` for inconsistent or out-of-range settings."""
        errors: list[str] = []
        if self.binarization.sigma_multiplier <= 0:
            errors.append("binarization.sigma_multiplier must be > 0")
        if self.binarization.hist_bin_width < 1:
            errors.append("binarization.hist_bin_width must be >= 1")
        lo = self.binarization.extreme_low_fraction
        hi = self.binarization.extreme_high_fraction
        if not (0 <= lo < 0.5) or not (0 <= hi < 0.5):
            errors.append("extreme band fractions must lie in [0, 0.5)")
        if lo + hi >= 0.9:
            errors.append("extreme low + high fractions must leave an intermediate range")
        if self.contour.link_k_start < 3 or self.contour.link_k_start % 2 == 0:
            errors.append("contour.link_k_start must be odd and >= 3")
        if self.contour.link_k_max < self.contour.link_k_start:
            errors.append("contour.link_k_max must be >= contour.link_k_start")
        if self.contour.link_max_iterations < 1:
            errors.append("contour.link_max_iterations must be >= 1")
        if self.contour.refine_link_k < 3 or self.contour.refine_link_k % 2 == 0:
            errors.append("contour.refine_link_k must be odd and >= 3")
        if not (0 < self.classification.cutoff_percentile <= 100):
            errors.append("classification.cutoff_percentile must be in (0, 100]")
        if self.contour.min_region_area < 1:
            errors.append("contour.min_region_area must be >= 1")
        if self.output.fig_dpi < 50:
            errors.append("output.fig_dpi must be >= 50")
        if self.preprocessing.median_kernel not in (0, 1) and (
            self.preprocessing.median_kernel < 3
            or self.preprocessing.median_kernel % 2 == 0
        ):
            errors.append("preprocessing.median_kernel must be 0 or an odd number >= 3")
        if self.preprocessing.gaussian_sigma < 0:
            errors.append("preprocessing.gaussian_sigma must be >= 0")
        if self.calibration.pixel_size_um is not None and self.calibration.pixel_size_um <= 0:
            errors.append("calibration.pixel_size_um must be > 0 when supplied")
        if errors:
            raise ValueError("Invalid configuration: " + "; ".join(errors))


def default_config() -> SegmentationConfig:
    """Return a validated default configuration reproducing the paper."""
    cfg = SegmentationConfig()
    cfg.validate()
    return cfg
