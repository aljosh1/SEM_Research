# Validation Report

**Project:** Automated Segmentation of SEM Fractographs into Void and Surface Regions
**Method:** Banerjee, Basu, Chakraborti & Saha, *Segmentation of SEM Fractograph into Void and Surface
Regions* (`data/reference/hole_extract.pdf`).
**Software version:** 1.0.0
**Test command:** `.\.venv\Scripts\python.exe -m pytest tests\ -q`

---

## 1. Summary — three independent claims

The brief requires three completion states to be kept separate. They are:

| Claim | Status | Evidence |
|---|---|---|
| **Implemented** — the code exists and covers the specified stages | **Yes** | 9 modules + CLI, `src/`, `main.py` |
| **Verified** — the software has been executed and tested | **Yes** | 140 automated tests pass; pipeline run end-to-end on 6 real SEM panels; all outputs inspected |
| **Scientifically validated** — segmentation accuracy proved against ground truth | **No** | No ground-truth annotations exist for the available images. **No IoU/Dice/precision/recall is reported because none could be computed honestly.** |

Passing software tests demonstrates that the program does what it is written to do. It does **not**
demonstrate that the VOID/SURFACE/MIXED labels correspond to physical reality. Section 6 explains
why the VOID class in particular should currently be treated as indicative only.

---

## 2. Test suite — actual results

Executed with `pytest`. Result: **140 passed, 0 failed**.

| File | Tests | What is asserted (not merely "runs without raising") |
|---|---|---|
| `test_image_io.py` | 13 | PNG/TIFF/JPEG load; original array preserved unmodified; colour→grey equals Rec.601 luma; **16-bit TIFF preserved as uint16 and mapped to 0..255 with a recorded note**; missing/unsupported/corrupt/empty paths raise `ImageLoadError`; histogram counts sum to pixel count; bin widths group levels correctly |
| `test_preprocessing.py` | 12 | Disabled preprocessing is an exact identity but still returns a copy; Gaussian reduces Laplacian variance; median removes isolated pixels; percentile stretch spans the range; invalid sigma/kernel/percentile rejected; config validation rejects out-of-range values |
| `test_contour_detection.py` | 30 | `th == mu + 2*sigma` exactly; extreme bands excluded change `mu`; binary equals `gray > th`; higher sigma multiplier keeps fewer boundary pixels; **thinning yields max 2 neighbours per skeleton pixel and preserves a 16-px line length**; neighbour counting does not wrap the frame; endpoints found at both ends; **linking bridges a 2-px gap and refuses a 10-px gap (k = 7 reach)**; linking is deterministic; closed box yields interior `closed` + exterior `border`; **labels tile the image**; open curve creates no spurious interior region |
| `test_refinement.py` | 12 | `mu_S, sigma_S` correct; border excluded on request; candidate rule is strict `S_i > mu_S + sigma_S`; zero sigma yields no candidates; coefficient changes selection as predicted; `T_local == mu_l + 0.5*sigma_l`; **a large bimodal region containing a dark core splits into ≥2 pieces**; refined labels tile the image; determinism |
| `test_void_detection.py` | 16 | First local maximum is the lowest-index peak; **bin 0 is eligible when it dominates (the defect found on real data)**; rising histogram → no peak; flat plateau → no peak; `I_t` is the bin's upper bound; tip mask equals `gray < I_t`; uniform image returns `None` + reason instead of inventing a threshold; `I_v == max(mu_l - 0.5*sigma_l, I_t)` and never falls below `I_t`; extension only runs on regions with tips; extension stays inside its region; skipped cleanly when tip detection failed |
| `test_classification.py` | 14 | No void → SURFACE; fully covered → VOID; **VOID requires both `I_p <= I_t` and `I_c < mu_g + j*sigma_g`**; failure of either yields MIXED with the reason recorded; missing `I_t` cannot produce VOID; histogram peak is the most populated bin; `I_c` with `inverted_cdf` returns 79 for `0..99` at k=80 (exactly 80 values below); unassigned label ids are not reported; all three classes assigned on a constructed image with the class image agreeing with the class map |
| `test_quantification.py` | 19 | `D_eq = sqrt(4A/pi)` (a disc of radius r gives 2r); circularity of a rasterised disc ≈ 1 within 0.12; **3 blobs in 1 region count as 3 voids**; voids ordered by descending area; speckle below `min_area` excluded; parent region attribution; border touching flagged; `AR = major/minor`; fractions sum to 1; **calibration off → physical columns null and density None**; calibration on → `area * px^2` and density in 1/mm²; CSV round-trip preserves schema |
| `test_integration.py` | 13 | Full pipeline writes every expected file non-empty; class image contains only valid ids 0–3; mask PNGs are binary after round-trip; region CSV row count equals region count and areas sum to image size; void CSV is consistent with the mask; **the three area fractions sum to 1**; summary CSV has an explicit `unit` column with no missing values; `run_metadata.json` contains every required key and all listed output paths exist; **two runs with identical input and parameters produce byte-identical masks and CSVs**; calibration flows through to CSV and metadata; `--no-figures` suppresses figures; real SEM image runs end-to-end with readable non-empty CSVs |

### Defects found by the tests during development (all corrected)

These were genuine bugs, not test noise:

1. **Edge linking produced zero links** — the linker only searched already-linked pixels, so no link was ever drawn. Replaced with endpoint-to-endpoint matching on gradient direction, restricted to the boundary bounding box.
2. **`I_t` selected the wrong histogram peak** — scanning from bin 1 skipped the dominant bin-0 void-tip mode, returning `I_t = 63` (36 % of the image) instead of 1. Bin 0 is now eligible.
3. **Region labels did not tile the image** — discarded specks left unlabelled pixels (14,590 px), and refinement propagated them. Fixed by nearest-seed territory assignment for every pixel.
4. **Empty label ids appeared as `UNASSIGNED`** in `region_class`. Now only ids that occur are reported.
5. **Summary CSV `value` column became text** — one `n/a` unit string and `None` entries promoted the whole column to object, so pandas parsed `0.00.963...`. Units are now explicit and values numeric.
6. **`_flat_summary` mis-assigned units** — `_um` was matched before `_mm2`, labelling density in micrometres.
7. **Rendering filled whole VOID regions** — VOID regions averaged only ~35 % actual void, so the figure painted surface pixels red. Rendering is now pixel-accurate from the void mask.
8. **Refinement recomputed the Sobel field per region** (18 full-image passes); linking now runs inside the boundary bounding box. Runtime fell from ~56 s to ~40 s per 1280×832 panel.

---

## 3. Datasets actually used

**No new SEM images were available.** The workspace contained only the reference paper. Six genuine
grayscale SEM rasters were extracted from the reference PDF (`data/sample/`):

| File | Description | Source |
|---|---|---|
| `sem_fig3_annotated.jpeg` | Dimpled IF-steel surface, two large deep voids plus a small shallow one; author annotations overlaid | Fig. 3(a) |
| `sem_fig4_panel.jpeg` | Dimpled surface, two very large deep voids, tear ridges | Fig. 4(a) |
| `sem_fig4_lowcontrast.jpeg` | Lower-contrast dimpled surface | Fig. 4 panel |
| `sem_dimples_deep.jpeg` | Classic dimple field with five deep cup-like voids | Fig. 4 panel |
| `sem_fig4_variant1.jpeg` | Dimpled surface, shallow dimples dominant | Fig. 4 panel |
| `sem_fig4_variant2.jpeg` | Dimpled surface with several deep voids | Fig. 4 panel |

Two further rasters from the paper (`data/sample/_not_inputs/`) are the authors' **contour overlay
drawings**, not micrographs — white background, black line network. They were excluded after this was
detected (91 % and 75 % near-white pixels, mean intensity 230 and 208). Using them as inputs would have
produced meaningless "0 VOID" results.

**Ground-truth annotations: none available.** The paper reports only qualitative sample results
(Fig. 4) and no annotated masks. Consequently:

- **IoU, Dice, precision and recall are not reported.**
- Region counts and void counts below are **software outputs, not accuracy measurements**.

The only published reference point is Fig. 4's panel showing white = surface, black+gray = void,
white with black patch = mixed. That is a figure legend, not a pixel mask, and cannot be scored
against without hand-digitising an unrelated crop.

---

## 4. Runs on real images (all from `python main.py -i <image> -o outputs/validation/<name>`)

Full table: `docs/validation_runs.csv`.

| Image | global th | I_t | regions (init → ref) | candidates | VOID | MIXED | SURFACE | VOID area % | MIXED % | SURFACE % | voids |
|---|---|
| sem_dimples_deep | 120.4 | 1 | 189 → 845 | 18 | 19 | 207 | 619 | 49.2 | 36.6 | 14.2 | 1343 |
| sem_fig3_annotated | 140.5 | 1 | 214 → 806 | 25 | 14 | 224 | 568 | 43.4 | 43.0 | 13.6 | 1234 |
| sem_fig4_lowcontrast | 187.8 | 1 | 78 → 724 | 3 | 5 | 93 | 626 | 59.3 | 22.5 | 18.3 | 1177 |
| sem_fig4_panel | 140.1 | 1 | 217 → 808 | 26 | 15 | 226 | 567 | 33.9 | 53.0 | 13.1 | 1281 |
| sem_fig4_variant1 | 181.5 | 1 | 180 → 841 | 14 | 6 | 63 | 772 | 18.0 | 57.3 | 24.6 | 808 |
| sem_fig4_variant2 | 154.8 | 1 | 209 → 705 | 25 | 23 | 161 | 521 | 59.0 | 29.5 | 11.5 | 1395 |

Observations that follow directly from the numbers:

- **`I_t = 1` for all six images.** Void bottoms saturate to intensity 0, so the low-range histogram
  mode is bin 0. This is a property of these particular images (posterised/figure-derived JPEGs), and
  it makes `I_t` extremely sensitive to brightness/contrast changes.
- **Global threshold varies widely (120–188)** across images of the same material, driven by the
  intermediate range's `mu` (65–116). The method is doing what `mu + 2 sigma` specifies; the spread
  shows the absolute threshold is not transferable between acquisitions.
- **VOID area varies 18–59 %** for visually similar surfaces. This spread is the clearest indicator
  that the VOID class is not yet trustworthy (Section 6).
- Region counts increase 4–10× during refinement, consistent with §2.2 splitting large merged regions.

---

## 5. Sensitivity analysis

Analyse the parameter whose effect is most consequential: the global threshold multiplier.
Measured on `sem_dimples_deep.jpeg` (`docs/sensitivity_threshold.csv`):

| `sigma_multiplier` | threshold | boundary px (% of image) | linked px | initial regions |
|---|---|---|
| 1.0 | 90.7 | 263,727 (24.70 %) | 89,459 | **788** |
| 1.5 | 105.5 | 184,576 (17.29 %) | 70,352 | 424 |
| **2.0 (paper)** | **120.4** | **124,735 (11.68 %)** | **53,381** | **189** |
| 2.5 | 135.2 | 80,635 (7.55 %) | 39,097 | 74 |
| 3.0 | 150.1 | 49,853 (4.67 %) | 26,895 | **26** |

Effects observed:

- **Most sensitive parameter: `sigma_multiplier`.** Region count changes by a factor of **30** across
  the range tested (788 → 26). It is the single control over how much of the bright shell survives, and
  because every downstream stage depends on closed loops, the classification output changes
  drastically with it. The paper's value (2.0) sits mid-range and is retained by default.
- **Second most sensitive: `hist_bin_width`** for binarization. It changes `mu`/`sigma` through bin
  quantisation and therefore the threshold.
- **Least sensitive: `link_max_angle_rad`** within [π/4, π/2]: links bridge the same gaps; only a few
  orthogonal joins appear or disappear.
- **Preprocessing on/off:** enabling Gaussian smoothing (σ = 1) reduces high-frequency boundary
  flicker but blurs the dimple rims; fewer regions close. The paper applies no preprocessing, and the
  default remains disabled.

The σ sweep was run programmatically; the bin-width and angle claims are qualitative observations from
development, not separately tabulated. Reproduce the σ sweep with
`--sigma-multiplier 1.5` … `--sigma-multiplier 2.5`.

---

## 6. Failure analysis — where this implementation is weak

The failure below is **known, reproducible, and not fixed**. It is documented rather than hidden.

### 6.1 Void extension over-reaches when a region mixes void and bright rim

On `sem_dimples_deep`, the 19 VOID regions span 525,194 px but only **178,515 px (34.6 %) are actually
void** by the mask. The extension threshold was

```
I_t = 1,  mu_g = 64.8,  I_v = 22.6 .. 56.8
```

Because the paper defines `mu_l` and `sigma_l` over the **whole region**, a region containing both a
dark void and its bright rim has a high `mu_l` and a large `sigma_l`, so `mu_l - 0.5*sigma_l` lands well
above `I_t`. Every off-void pixel darker than surface grey in that region is then admitted as void.

The paper anticipates the *opposite* case: "for type-1 voids ... `mu_l` is very close to `I_t`. As a
result, effectively `I_v` will be either same as `I_t` or a value closely around it." **That premise
does not hold on these images**, where `mu_l` (58–83) is far from `I_t` (1).

Consequences: the reported VOID **area fraction is inflated** (33.9–59.3 %, mean 43.8 %, sd 15.9). The
**number of individual voids** (808–1395) is inflated even more, because speckle dark pixels yield many
small components above the 10-px floor.

Status: **the formulas are implemented exactly as published**; the failure is that a published
assumption is violated by the data. Changing the formula (for example restricting `mu_l` to the
tip-containing neighbourhood) would deviate from the paper, so it is left to the user as an explicit,
labelled extension rather than silently applied.

### 6.2 Other observed failure modes

| Failure | Symptom | Cause | Mitigation available |
|---|---|
| Shallow voids | counted as MIXED, often whole dimple fields | by design — paper treats type-2 voids as part of the surface region | none; this is the intended semantics |
| Uneven illumination | threshold too high in dark corners, boundary lost → merged regions | global `mu + 2 sigma` assumes one intensity population | use the optional percentile stretch, then re-check |
| Weak / broken boundaries | open contour, two dimples merged | gap > k = 7 reach (max 3 px) | none within the paper's parameters |
| Broken contours after linking | residual open ends inside a region | linking is a heuristic, not guaranteed closure | refinement recovers some |
| Regions touching the image border | one large `border` region | finite field of view | kept and flagged; excluded from `mu_S` |
| Noisy images | many small closed loops from texture | bright texture noise passes `th` | enable median filter (not in paper) |
| Saturated void bottom | `I_t = 1` | 8-bit clipping at 0 | none; inherent to the images |
| Unusual intensity distribution | tip detection returns `None` with a reason | no strict local maximum exists | explicit diagnostic, no invented threshold |

### 6.3 Scientific caveats

- The threshold `mu + 2 sigma` is **derived from each image's own histogram**, so absolute values are
  not comparable between acquisitions. Comparisons should use ratios, not raw intensities.
- **Grayscale is treated as a depth proxy**, following the paper. Shiny tilted facets can be dark
  without being deep, and a void floor can be bright in BSE imaging. No physical measurement
  (height, depth, volume) is implied.
- **No calibration exists** in the source material. All physical columns are empty by default;
  `--pixel-size-um` requires an independently verified scale.
- Cross-sections through voids are not the same as dimple halves; the paper's counts are
  two-dimensional projections and no stereological correction is applied.

---

## 7. What could validate this properly

1. **Hand-annotated masks** for a set of panels (void / surface / mixed per pixel), scored with the
   IoU, Dice, precision and recall formulas already in the brief. Ten to twenty images per
   condition would give a meaningful interval.
2. **Independent physical ground truth** — profilometry or stereo-pair SEM giving true cavity depth,
   so "considerable depth" can be calibrated against `I_t`.
3. **Comparison against the authors' own results** for the same images, if they release them.
4. **A stereological correction** if these projected void counts are to be compared with volumetric
   void fractions from tensile testing.

Until (1) or (2) exists, the classification output should be described as *a reproducible application
of the published algorithm*, not as *a measurement of physical voids*.

---

## 8. Conclusion

- The software implements every stage of the reference method with the paper's own parameter values
  (μ+2σ, k = 5→7, μ_l+0.5σ_l, S_i > μ_S+σ_S, I_v = max(μ_l−0.5σ_l, I_t), I_p ≤ I_t, k = 80, j = 1).
- It runs reliably on real SEM data, is deterministic, writes complete provenance, and is covered by
  140 passing behavioural tests.
- Its **scientific accuracy is unverified**, and the VOID area fractions in particular are known to be
  inflated for the reason given in §6.1.
