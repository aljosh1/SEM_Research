# Technical Note — Automated Segmentation of SEM Fractographs into Void and Surface Regions

**Implementation of:** S. Banerjee, K. Basu, P. C. Chakraborti, S. K. Saha,
*Segmentation of SEM Fractograph into Void and Surface Regions* (LNCS-style manuscript, 10 pp.,
supplied as `data/reference/hole_extract.pdf`), with its contour-extraction predecessor
S. Banerjee et al., *Extraction of Region Contour in SEM Fractographs*, IEEE CS Press, EAIT 2011,
pp. 98–101 (supplied as `data/reference/extraction_region_contour_2011.pdf`).

Code: `src/`. Every section below names the function that implements it.

---

## 1. Objective and scientific motivation

Quantitative fractography estimates fracture-surface features so that they can be correlated with
microstructure and loading history. For ductile fracture the surface carries dimples — the halves of
nucleated and coalesced voids — and the size, shape and distribution of those voids reflect the
underlying damage process (paper §1; refs. therein).

Manual image processing of SEM fractographs is ad hoc and analyst-dependent: the paper explicitly
motivates automation because "the outcome depends on the acumen of the individual." The objective of
this project is therefore an automated, reproducible segmentation of a fractograph into **void**,
**surface** and **mixed** regions, plus quantitative measurements of the void population.

## 2. Physical interpretation of the three classes

The paper distinguishes regions by *whether a void of considerable depth* is present (paper §2.3):

| Class | Paper wording | Physical meaning |
|---|---|---|
| **VOID** | "a void of considerable depth (type-1 void)" | Deep cavity; intensity rises gradually from a dark bottom to the bright rim where the void meets the surface, so the void subtends a large share of the region. |
| **SURFACE** | "a surface region has no void" | Fracture surface only. |
| **MIXED** | "surface and void(s) of insignificant depth (type-2 void)" | Surface densely populated with shallow dimples. Each dimple is dark but has no gradual depth gradient. |

A critical assumption, stated in the paper: a deep void shows a **gradient** in intensity with a
minimum at its bottom; a shallow void is "only a small patch of dark intensity without any noticeable
depth." Intensity is thus an *indicator* of depth, not independent proof of it — this note follows
the paper in treating grayscale as evidence, and §8 records where that inference is weak.

## 3. Reference methodology

Three stages (paper §2):

1. **Detection of region contour** (§2.1)
2. **Refinement of region contour** (§2.2)
3. **Region characterization** (§2.3)

### 3.1 Stage 1 — contour detection (`src/contour_detection.py`)

**(a) Binarization.** Boundary pixels are brighter than surfaces:

$$B(i,j) = \begin{cases} 1 & F(i,j) > th \\ 0 & \text{otherwise}\end{cases}$$

The threshold is derived from the *intermediate, bell-shaped* part of the intensity histogram:

$$th = \mu + 2\sigma$$

where $\mu,\sigma$ are the mean and standard deviation of the binned histogram over the intermediate
range, after "ignoring the peaks (if any) in two extreme intensity ranges." The paper's rationale:
choosing the upper local peak as $th$ yields a "highly porous boundary," while a low $th$ produces
"plenty of candidate pixels around the actual boundary" and splits a surface into smaller regions.

*Implementation:* `global_threshold()`. The intensity scale is binned (`hist_bin_width`, default 5);
the extreme bands are removed as the lowest/highest `extreme_low_fraction` / `extreme_high_fraction`
of the cumulative distribution (default 0.10 each); $\mu,\sigma$ are then computed over the remaining
bins, and $th = \mu + 2\sigma$. `histogram_peak_intensity()` is not used here — the paper requires the
*shape* of the intermediate range, not a peak.

**(b) Thinning.** Boundary segments are "made of single pixel width" so that linking is simpler.
*Implementation:* `thin()` using `skimage.morphology.skeletonize` (Zhang–Suen; topology preserving,
deterministic).

**(c) Edge linking.** "A Pixel is linked with another in a predefined neighbourhood having closest
gradient direction. In case of non-availability of any pixel in the said neighbourhood, then its size
is increased up to a maximum." The edge property is the **Sobel gradient direction** at each boundary
pixel, computed on the fractograph.

*Implementation:* `link_edges()`, with `gradient_direction()` for the Sobel direction folded onto the
orientation axis $[0,\pi)$ (an edge has no sign: $d$ and $d+\pi$ are the same edge). Directions are
quantised into 12 classes of 15°, matching the resolution at which "closest" is meaningful for 8-bit
data. Each iteration:

1. find open segment ends (`endpoint_mask()` — skeleton pixels with exactly one neighbour);
2. for each end, search the $k\times k$ neighbourhood, $k = 5$ then $7$, and select the candidate end
   whose direction class differs least;
3. accept a link only if the difference is within `link_max_angle_rad`;
4. draw a raster line between the two ends and re-thin the result.

Linking repeats until no new link is accepted, which implements "goes on iteratively till no further
growth in the contour segments is possible" and the instruction that "the boundary segments gain
width, once again thinning process is carried out." This is the **paper's parameter values** (k = 5
→ 7); the reach is 3 px, so wider gaps correctly remain open (verified by
`test_linking_cannot_bridge_gaps_wider_than_k_max`).

**(d) Pruning.** "we select a surface/void pixel in a detected region as a seed and region growing
algorithm is followed. It recursively marks the pixels in the regions till the boundary pixels are
encountered. The outer contour of the marked region provides the desired closed contour." The paper
gives no numeric pruning criterion; segmentation into **classical regions** (seed interior + nearest-distance
boundary assignment) is used instead of an arbitrary "reject large regions" rule, because such a rule
would confound thresholding failures with genuine deep voids. Border-touching regions are still
fruitful for candidate selection and are labelled `"border"`; non-border regions are `"closed"`.

### 3.2 Stage 2 — contour refinement (`src/contour_refinement.py`)

Refinement re-runs stage 1 *inside* the large regions with local statistics.

$$
S_i > \mu_S + \sigma_S
$$

with $S_i$ the size of region $i$ and $\mu_S,\sigma_S$ the mean and standard deviation of region sizes.
$a$ is configurable but **1.0 by default, as in the paper.**

For a candidate region, the local threshold is

$$
T_\text{local} = \mu_l + 0.5\,\sigma_l
$$

with $\mu_l,\sigma_l$ the mean and standard deviation of intensity **in that region**. The paper notes
it is "quite liberal to exclude the possibility of missing any likely boundary pixels"; to
compensate, "during linking, neighbourhood size is reduced to restrict the over splitting" — this
implementation uses $k = 3$ (`refine_link_k`), the smallest meaningful reduction from 5, because the
paper does not give the number.

*Implementation:* `select_candidates()`, `local_threshold()`, `refine_contours()`. Refinement only
splits; it never deletes a region or renames others, so label consistency is preserved. If a candidate
does not split, it is returned whole.

### 3.3 Stage 3 — void detection and region characterization (`src/void_detection.py`, `src/region_classification.py`)

**(a) Void tips.** "The intensity histogram is formed by dividing the intensity scale into bins of
small size and local maxima is determined by considering the previous and next bin. The upper bound of
the intensity bin corresponding to the first local maxima is taken as $I_t$. Pixels with intensity
value smaller than $I_t$ form the tip."

*Implementation:* `detect_void_tips()` → `find_first_local_maximum()`. Bins of `hist_bin_width`
(default 2); a bin is a local maximum if it strictly exceeds both neighbours; **bin 0 is eligible**
(the half-bin before zero is empty), because a saturated dark void bottom produces its mode there.
$I_t$ is the **upper bound** of that bin.

**(b) Void extension.** Pixels surrounding the tip belong to the void but are brighter:

$$
I_v = \max(\mu_l - 0.5\,\sigma_l,\; I_t)
$$

Pixels with intensity below $I_v$ join the void. Extension runs only on regions containing a tip.
The paper's justification: for type-1 voids "$\mu_l$ is very close to $I_t$," so $I_v \approx I_t$ and
extension is limited; for type-2 voids $\mu_l > I_t$ and $I_v = \mu_l - 0.5\sigma_l$ relaxes the
threshold to grow the tips.

*Implementation:* `extension_threshold()`, `extend_voids()`.

**(c) Classification.** "The regions without any void can be readily identified as the surface. A
region fully covered by a void can be treated as a void region... for the regions partially covered by
the void have to be analyzed further." For a partially covered region, with
$I_p$ = intensity of the region's histogram peak and $I_t$ as above:

$$
\text{VOID} \iff I_p \le I_t \;\wedge\; I_c < \mu_g + j\,\sigma_g
$$

otherwise **MIXED**. Here $I_c$ is "the minimum cut-off intensity value such that $k$% pixels in the
region have the intensity lower than the cut-off," and $\mu_g,\sigma_g$ are the whole-image mean and
standard deviation. **$k = 80$ and $j = 1$** in the paper, and are the defaults here.

*Implementation:* `classify_region()`, `classify_regions()`, `cutoff_intensity()` with
`numpy.percentile(method="inverted_cdf")`, which is the strict reading of "minimum cut-off such that
k% are below it."

## 4. Mathematical summary

| Quantity | Formula | Source |
|---|---|---|
| Global threshold | $th = \mu + 2\sigma$ over intermediate histogram range | paper §2.1 |
| Refinement candidate | $S_i > \mu_S + \sigma_S$ | paper §2.2 |
| Local threshold | $T_\text{local} = \mu_l + 0.5\sigma_l$ | paper §2.2 |
| Void-tip cutoff | $I_t$ = upper bound of first low-range histogram local maximum | paper §2.3 |
| Void extension | $I_v = \max(\mu_l - 0.5\sigma_l, I_t)$ | paper §2.3 |
| Void test (partial) | $I_p \le I_t$ **and** $I_c < \mu_g + j\sigma_g$, $j=1$ | paper §2.3 |
| Void percentile | $I_c$: min intensity with $k\%$ of region pixels below, $k=80$ | paper §2.3 |
| Equivalent diameter | $D_\text{eq} = \sqrt{4A/\pi}$ | project brief |
| Aspect ratio | $AR = L_\text{major}/L_\text{minor}$ | project brief |
| Circularity | $C = 4\pi A/P^2$ | project brief |
| Void area fraction | $f_v = A_\text{void}/A_\text{analysed}$ | project brief |

## 5. Image-processing operations used, and why

| Operation | Purpose | Risk if mis-set |
|---|---|---|
| Binned histogram + tail exclusion | Isolate the intermediate bell-shaped range so $\mu,\sigma$ describe surface pixels, not deep voids or bright rims | Including the void mode lowers $th$, producing false contours. |
| Global binarization | Candidate boundary pixels (bright shell) | Too high → porous, open contour; too low → over-segmentation. |
| Skeletonization | One-pixel boundary for linking and region derivation | Skeleton spurs add noise; pruned by the area filter. |
| Sobel direction | Edge property for linking | Noisy directions in texture; the 15° quantisation stabilises matching. |
| Region growing / connected components | Derive closed regions and discard open curves | Leakage through gaps merges regions (see §8). |
| Percentile ($k$=80) | Compare region's dark share against global intensity scale | Off-by-one percentile definitions change $I_c$ slightly. |

## 6. Algorithm assumptions

1. Boundaries are brighter than the surfaces they separate (paper §2).
2. The intermediate histogram range is approximately unimodal/bell-shaped (§2.1).
3. Regions are of comparable size, with deep voids producing *larger* regions (§2.2).
4. For a deep void, $\mu_l \approx I_t$ (§2.3) — i.e. the region's mean intensity is near the
   void-tip cutoff. **This assumption is the least secure; see §8.**
5. Void regions contain few bright surface pixels, so $I_p \le I_t$ holds for them.

## 7. Parameters requiring tuning

All parameters are in `src/config.py` with provenance (`paper` / `implementation` / `extension`),
documented in `PARAMETER_DOCS` and printed by `python main.py --list-parameters`.

| Parameter | Default | Provenance | Note |
|---|---|
| `binarization.sigma_multiplier` | 2.0 | paper | paper is explicit |
| `binarization.hist_bin_width` | 5 | paper says "small width", number not given | |
| `binarization.extreme_*_fraction` | 0.10 | "ignore peaks in extreme ranges" | operationalises an instruction |
| `contour.link_k_start` / `link_k_max` | 5 / 7 | paper | paper is explicit |
| `contour.link_max_angle_rad` | π/2 | implementation | guard, not in paper |
| `contour.refine_link_k` | 3 | paper says "reduced", value not given | |
| `contour.local_sigma_coeff` | 0.5 | paper | paper is explicit |
| `contour.area_sigma_coeff` | 1.0 | paper | paper is explicit |
| `void.hist_bin_width` | 2 | "bins of small size", number not given | |
| `void.extension_sigma_coeff` | 0.5 | paper | paper is explicit |
| `classification.cutoff_percentile` (k) | 80 | paper | paper is explicit |
| `classification.global_sigma_multiplier` (j) | 1 | paper | paper is explicit |

The paper reports no pixel-to-micrometre calibration, so physical measurements are unavailable unless
the user supplies `--pixel-size-um`. Magnification is never converted to a pixel size.

## 8. Known limitations (verified empirically, see `docs/validation_report.md`)

1. **First-stage contour can leak.** On real panels the k ≤ 7 linking reach is often smaller than the
   gaps in the binarized boundary, so neighbouring dimples merge. This is the paper's own design
   constraint, not a regression.
2. **Region-size statistics are strongly skewed.** Measured $\mu_S = 307$ px vs $\sigma_S = 576$ px on
   a real panel, so $\mu_S + \sigma_S$ selects only the largest regions. The paper's "regions are of
   uniform size" premise does not hold on every fractograph.
3. **Void extension over-reaches when $\sigma_l$ is large.** Because $\mu_l - 0.5\sigma_l$ is computed
   over the whole region (as the paper states), a region containing both void depths and bright rim has
   a high $I_v$ (measured 22.6–56.8 at $\mu_g = 64.8$, $I_t = 1$), so off-void surface shadow above
   $I_t$ is also admitted. VOID regions consequently averaged ~35 % actual void coverage on the
   validation image. **The resulting VOID area fraction (49 %) is not physically validated.**
4. **No ground truth.** No annotated masks exist for these images, so no IoU/Dice/precision/recall was
   computed. Software correctness is verified; scientific accuracy is **not**.
5. **Saturation of the void bottom.** $I_t = 1$ on these panels because void bottoms clip to 0. Any
   change in SEM brightness/contrast shifts $I_t$ directly.

## 9. Traceability to code

| Paper location | Function |
|---|---|
| §2.1 binarization, $th=\mu+2\sigma$ | `global_threshold`, `binned_histogram` |
| §2.1 thinning | `thin` |
| §2.1 edge linking, k = 5→7 | `link_edges`, `gradient_direction`, `endpoint_mask`, `neighbourhood_offsets` |
| §2.1 pruning / region growing | `label_regions` |
| §2.2 candidate rule | `select_candidates`, `region_area_statistics` |
| §2.2 local threshold | `local_threshold`, `refine_contours` |
| §2.3 void tips, $I_t$ | `detect_void_tips`, `find_first_local_maximum` |
| §2.3 extension, $I_v$ | `extension_threshold`, `extend_voids` |
| §2.3 classification | `classify_region`, `classify_regions`, `cutoff_intensity` |
| Fig. 4 output description | `visualization.figure_final_segmentation` |

## 10. References

1. S. Banerjee, K. Basu, P. C. Chakraborti, S. K. Saha. *Segmentation of SEM Fractograph into Void and
   Surface Regions.* Manuscript (`data/reference/hole_extract.pdf`).
2. S. Banerjee, K. Basu, P. C. Chakraborti, S. K. Saha. *Extraction of Region Contour in SEM
   Fractographs.* Proc. 2nd Int. Conf. on Emerging Applications of Information Technology, IEEE CS
   Press, 2011, pp. 98–101. DOI: 10.1109/EAIT.2011.20.
3. Zhang, T. Y., Suen, C. Y. *A fast parallel algorithm for thinning digital patterns.*
   Commun. ACM 27(3), 1984.
4. Virtanen, P. et al. *SciPy 1.0.* Nature Methods 17, 2020. (percentile definitions)
