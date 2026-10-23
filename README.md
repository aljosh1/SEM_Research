# Automated Segmentation of SEM Fractographs into Void and Surface Regions

A reproducible Python implementation of the conventional image-processing method of
**Banerjee, Basu, Chakraborti & Saha** for segmenting scanning-electron-microscope fractographs
into **VOID**, **SURFACE** and **MIXED** regions, with quantitative fractography output (void sizes,
shapes, area fractions) and complete provenance for every run.

No deep learning is used anywhere: the pipeline is intensity histograms, thresholding,
skeletonization, direction-guided edge linking, region growing and explicit decision rules.

> **Scientific status.** The software is implemented and verified (159 automated tests pass, and it has
> been run end-to-end on six real SEM panels). It is **not** scientifically validated: no ground-truth
> annotations exist for these images, so no accuracy metric is claimed — although the evaluation code
> that computes IoU/Dice/precision/recall is implemented and ready (`evaluate.py`, see §16). The VOID
> area fractions are known to over-estimate, for reasons documented in
> [`docs/validation_report.md`](docs/validation_report.md) §6.1. Read that section before interpreting
> results.

---

## Contents

1. [Scientific motivation](#1-scientific-motivation)
2. [Objective](#2-objective)
3. [The three region classes](#3-the-three-region-classes)
4. [Algorithm summary](#4-algorithm-summary)
5. [Technology stack](#5-technology-stack)
6. [Project structure](#6-project-structure)
7. [Windows installation](#7-windows-installation)
8. [Opening in VS Code](#8-opening-in-vs-code)
9. [Command-line usage](#9-command-line-usage)
10. [Supported image formats](#10-supported-image-formats)
11. [Configuration options](#11-configuration-options)
12. [Generated outputs](#12-generated-outputs)
13. [Measurement definitions](#13-measurement-definitions)
14. [Physical calibration](#14-physical-calibration)
15. [Running the tests](#15-running-the-tests)
16. [Validation methodology](#16-validation-methodology)
17. [Known limitations](#17-known-limitations)
18. [Reference paper](#18-reference-paper)
19. [Adding your own sample images](#19-adding-your-own-sample-images)
20. [Troubleshooting](#20-troubleshooting)

---

## 1. Scientific motivation

Ductile fracture proceeds by void nucleation, growth and coalescence; the halves of those voids are the
dimples visible on the fracture surface. Their size, shape and distribution encode the material's
damage history and correlate with tensile properties. Quantifying them by hand in an image editor is
slow and observer-dependent, which is why the reference paper proposes an automated scheme.

## 2. Objective

Given one SEM fractograph, produce:

- an integer region labelling,
- a three-class map (VOID / SURFACE / MIXED),
- a pixel-accurate void mask,
- per-void and per-region measurement tables,
- figures and a run-metadata file sufficient to reproduce the result.

## 3. The three region classes

| Class | Paper definition | Physical meaning |
|---|---|---|
| **VOID** | "a void of considerable depth (type-1 void)" | Deep cavity. Intensity rises gradually from a dark bottom to the bright rim, so the void occupies a large share of the region. |
| **SURFACE** | "a surface region has no void" | Fracture surface with no detected void. |
| **MIXED** | "surface and void(s) of insignificant depth (type-2 void)" | Surface carrying one or more shallow dimples with no depth gradient. |

The three classes are never collapsed into two. Note that in the paper's own figures, shallow dimples
are treated as part of the enclosing surface region — so a dense dimple field legitimately reads as
MIXED.

## 4. Algorithm summary

**Stage 1 — initial contour** (paper §2.1)

1. Binarize with `th = μ + 2σ`, where μ and σ are computed over the *intermediate* bins of the
   intensity histogram, ignoring the extreme low/high tails.
2. Thin the boundary to one-pixel width.
3. Link broken segments: each boundary pixel carries its **Sobel gradient direction**; an open end is
   joined to the nearby end with the closest direction, searching `k=5` then `k=7`. Iterate until no
   segment grows, re-thinning each time.
4. Prune by region growing: enclosed components are regions; open curves do not separate anything and
   are discarded. Boundary pixels are assigned to their nearest region, so the labelling tiles the image.

**Stage 2 — local refinement** (paper §2.2)

Regions with `S_i > μ_S + σ_S` are re-thresholded *inside themselves* at `T_local = μ_l + 0.5σ_l`,
then thinned, linked with a reduced neighbourhood (`k=3`) and re-segmented into pieces.

**Stage 3 — void detection and classification** (paper §2.3)

- `I_t` = upper bound of the bin of the **first local maximum in the low-intensity range**.
- Pixels below `I_t` are void tips; tips are extended inside their region with
  `I_v = max(μ_l − 0.5σ_l, I_t)`.
- A region with no void is SURFACE; fully covered is VOID; otherwise it is VOID only if
  `I_p ≤ I_t` **and** `I_c < μ_g + jσ_g` (k = 80, j = 1), else MIXED.

Full equations, assumptions and traceability: [`docs/technical_note.md`](docs/technical_note.md).
Flowchart and pseudocode: [`docs/algorithm_flowchart.md`](docs/algorithm_flowchart.md).

## 5. Technology stack

| Component | Version | Why |
|---|---|---|
| Python | 3.11–3.14 | target runtime |
| numpy | ≥ 1.26 | array maths |
| opencv-python | ≥ 4.9 | TIFF/PNG/JPEG I/O, connected components, Sobel, contours |
| scikit-image | ≥ 0.22 | topology-preserving skeletonization |
| scipy | ≥ 1.11 | distance transforms for region territories |
| matplotlib | ≥ 3.8 | figures (Agg backend, headless-safe) |
| pandas | ≥ 2.1 | CSV output |
| pytest | ≥ 8.0 | test suite |

Version pins in `requirements.txt`. No deep-learning framework is installed or needed.

## 6. Project structure

```
SEM_RESEARCH/
├── data/
│   ├── raw/                    # your own inputs (git-ignored)
│   ├── sample/                 # real SEM panels extracted from the paper
│   │   └── _not_inputs/        # paper figures that are drawings, not micrographs
│   └── reference/              # the two reference PDFs
├── docs/
│   ├── annotations_guide.md   # how to label masks for scoring
│   ├── technical_note.md       # methodology, equations, traceability
│   ├── algorithm_flowchart.md  # flowchart + pseudocode
│   ├── validation_report.md    # tests, real-image runs, failures
│   ├── validation_runs.csv     # per-image metrics (actual)
│   ├── sensitivity_threshold.csv
│   └── presentation_outline.md
├── outputs/                    # created per run (git-ignored)
├── src/
│   ├── config.py               # all parameters + provenance
│   ├── image_io.py             # loading, bit-depth handling, histograms
│   ├── preprocessing.py        # optional, disabled by default
│   ├── contour_detection.py    # §2.1
│   ├── contour_refinement.py   # §2.2
│   ├── void_detection.py       # §2.3 tip detection + extension
│   ├── region_classification.py# §2.3 classification
│   ├── quantification.py       # void/region measurements
│   ├── visualization.py        # figures
│   ├── pipeline.py             # orchestration + metadata
│   └── evaluation.py           # IoU / Dice / precision / recall vs annotations
├── tests/                      # 159 pytest tests
├── main.py                     # command-line application
├── evaluate.py                 # score runs against annotations
├── requirements.txt
└── .gitignore
```

## 7. Windows installation

Requires Python 3.11+ on `PATH`. From the project root in **PowerShell**:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation with an execution-policy error, either allow it for the current user

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

or skip activation and always call the interpreter directly:

```powershell
.\.venv\Scripts\python.exe main.py --help
```

**Command Prompt (`cmd.exe`)** — activation differs:

```bat
.venv\Scripts\activate.bat
```

## 8. Opening in VS Code

```powershell
code .
```

Then select the interpreter: `Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv\Scripts\python.exe`.
The venv is git-ignored, so it must be created on each machine.

For a quick visual check inside VS Code, open `main.py` and press `F5`, or run the task:

```powershell
.\.venv\Scripts\python.exe main.py -i data\sample\sem_dimples_deep.jpeg -o outputs\demo
```

## 9. Command-line usage

Show all options:

```powershell
.\.venv\Scripts\python.exe main.py --help
```

Print the documented parameter table (name, default, unit, provenance):

```powershell
.\.venv\Scripts\python.exe main.py --list-parameters
```

**Basic run** (reproduces the paper's parameter values):

```powershell
.\.venv\Scripts\python.exe main.py -i data\sample\sem_dimples_deep.jpeg -o outputs\run1
```

**With a calibration** (only if you have a verified pixel size):

```powershell
.\.venv\Scripts\python.exe main.py -i data\sample\sem_dimples_deep.jpeg -o outputs\run_cal --pixel-size-um 0.25 --calibration-source "verified 20 um scale bar"
```

> `0.25` is an illustrative number for the command syntax only. It is **not** a value measured from
> any image in this repository.

**Tuned run** (sensitivity check on the global threshold):

```powershell
.\.venv\Scripts\python.exe main.py -i data\sample\sem_dimples_deep.jpeg -o outputs\run_thr --sigma-multiplier 2.5
```

**Fast batch run** (no figures):

```powershell
.\.venv\Scripts\python.exe main.py -i data\sample\sem_fig4_panel.jpeg -o outputs\fast --no-figures -v
```

Exit codes: `0` success, `1` unexpected processing failure, `2` invalid input or parameters.
The program never silently continues with an invalid parameter — it raises a configuration error naming
the offending field.

Runtime is roughly 30–60 s for a 1280×832 panel on a typical laptop; most of it is skeletonization.

### Adjustable parameters

| Flag | Default | Paper value |
|---|---|---|
| `--sigma-multiplier` | 2.0 | 2.0 |
| `--hist-bin-width` | 5 | "small width" (unspecified) |
| `--extreme-low-fraction` / `--extreme-high-fraction` | 0.10 | "ignore extreme peaks" |
| `--link-k-start` / `--link-k-max` | 5 / 7 | 5 → 7 |
| `--min-region-area` | 50 | not in paper |
| `--area-sigma-coeff` | 1.0 | 1.0 |
| `--local-sigma-coeff` | 0.5 | 0.5 |
| `--refine-link-k` | 3 | "reduced" (unspecified) |
| `--void-hist-bin-width` | 2 | "small size" (unspecified) |
| `--extension-sigma-coeff` | 0.5 | 0.5 |
| `--cutoff-percentile` (k) | 80 | 80 |
| `--global-sigma-multiplier` (j) | 1.0 | 1.0 |
| `--percentile-method` | `inverted_cdf` | unspecified |
| `--preprocess`, `--gaussian-sigma`, `--median-kernel`, `--percentile-clip` | off | no preprocessing |
| `--pixel-size-um`, `--calibration-source` | none | none |

## 10. Supported image formats

`.png`, `.tif`, `.tiff`, `.jpg`, `.jpeg`, `.bmp`.

- **Colour** images are converted to grey with Rec.601 weights; the colour array is retained in memory.
- **16-bit TIFF** is loaded intact as `uint16` (never down-converted by the loader). Because the
  algorithm works on 0–255 as in the paper, it is mapped linearly to `uint8`; the mapping and the
  source range are recorded in `run_metadata.json` under `image.conversion_note`.
- The **original array is never modified**; all processing happens on copies.

## 11. Configuration options

Two equivalent routes:

1. **CLI flags** (Table above) — best for one-off experiments.
2. **`src/config.py`** — best for programmatic use and for changing defaults:

```python
from src.config import default_config
from src.pipeline import run_pipeline

cfg = default_config()
cfg.binarization.sigma_multiplier = 2.0
cfg.calibration.pixel_size_um = 0.25
cfg.validate()          # raises ValueError on invalid combinations
run_pipeline("image.tif", "outputs/x", cfg)
```

Every parameter carries a provenance tag — `paper`, `implementation` (the paper specifies the operation
but not the number) or `extension` (not in the paper). `cfg.validate()` rejects inconsistent settings.

## 12. Generated outputs

Each run creates:

```
outputs/<name>/
├── contours/
│   ├── 01_binary.png                 # th = mu + 2 sigma
│   ├── 02_thinned.png
│   ├── 03_linked.png
│   ├── 04_region_labels.png          # initial regions
│   └── 05_refined_region_labels.png  # after §2.2
├── masks/
│   ├── 01_void_tip_mask.png          # pixels < I_t
│   ├── 02_extended_void_mask.png     # after I_v extension
│   ├── 03_class_image.png            # 0=boundary, 1=SURFACE, 2=MIXED, 3=VOID
│   └── 04_class_overlay.png          # class tint over the grey image
├── figures/
│   ├── 00_composite_overview.png     # one-page review figure
│   ├── 01_original.png
│   ├── 02_grayscale.png (or 02_preprocessed.png)
│   ├── 03_histogram.png
│   ├── 04_initial_contour.png
│   ├── 05_refinement.png
│   ├── 06_void_detection.png
│   ├── 07_final_segmentation.png     # + TIFF unless --no-tiff
│   ├── 08_side_by_side.png           # original | segmentation
│   ├── 09_void_size_distribution.png
│   └── 10_summary.png
├── measurements/
│   ├── void_measurements.csv
│   ├── region_measurements.csv
│   └── summary_statistics.csv
└── run_metadata.json
```

Masks are saved as 0/255 `uint8` PNG (with TIFF copies by default). Class colours are fixed across all
figures: **VOID red (220,30,30)**, **SURFACE green (30,160,60)**, **MIXED yellow (235,200,20)**.
These are visual aids, not part of the algorithm.

`run_metadata.json` records the input path and dimensions, timestamp, software and Python versions,
every parameter value with its provenance, thresholds (`th`, `I_t`, refinement cut-off), preprocessing
operations, calibration source, per-stage diagnostics, warnings, and the list of written files.

> Note: figures render the void mask **per pixel**. A VOID *region* is a classification unit and may
> contain non-void pixels (measured ~35 % void coverage on real panels), so painting the whole region
> red would misrepresent the image.

## 13. Measurement definitions

| Quantity | Definition |
|---|---|
| Equivalent circular diameter | `D_eq = sqrt(4A/π)` |
| Aspect ratio | `AR = L_major / L_minor` (from the minimum-area rectangle) |
| Circularity | `C = 4πA / P²` (1 for a disc; less for irregular shapes) |
| Void area fraction | `f_v = A_void / A_analysed` |

- **Individual voids** are connected components of the extended void mask, filtered at ≥ 10 px.
  A MIXED region containing several shallow dimples therefore contributes several voids — the number
  of void *regions* is not the number of voids.
- **Touching voids** that the mask joins are counted as one component. This is reported, not split
  artificially.
- Voids touching the image border are flagged (`touches_border`), because they are truncated by the
  field of view.

Files:

- `void_measurements.csv` — one row per void: image id, void id, parent region, pixel area, D_eq,
  major/minor axis, aspect ratio, perimeter, circularity, centroid, border flag, calibration flag,
  and physical area / diameter if calibrated.
- `region_measurements.csv` — one row per region: class, area and fraction, number of associated voids,
  void area and fraction, `I_p`, `I_c`, mean/std intensity, tip pixels, `I_v`, and the reason the
  classification fired.
- `summary_statistics.csv` — long format (`metric`, `value`, `unit`) so units are never ambiguous.

## 14. Physical calibration

**No calibration exists in the source material.** The paper reports magnification but no pixel size,
and a pixel size is **never inferred from magnification** (that requires working distance, detector
geometry and magnification calibration).

Provide it explicitly when you have it:

```powershell
.\.venv\Scripts\python.exe main.py -i image.tif -o outputs\cal --pixel-size-um 0.0834 --calibration-source "SEM metadata: 1024x768 at 2000x, 85.4 um field width"
```

Without it, every physical column is empty, `calibration_available` is `false`, and
`void_number_density_per_mm2` is `null`. Pixel-based analysis never requires calibration.

## 15. Running the tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\ -v
```

Current result: **159 passed**. Coverage by module is listed in
[`docs/validation_report.md`](docs/validation_report.md) §2, including the eight real defects the suite
exposed during development.

Run a subset:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_void_detection.py -q
```

The suite includes one test that needs `data/sample/sem_dimples_deep.jpeg`; it skips cleanly if the
sample is absent.

## 16. Validation methodology

Three states are kept strictly separate (details in `docs/validation_report.md`):

1. **Implemented** — all stages coded, parameters traceable to the paper.
2. **Verified** — executed and tested: 159 passing behavioural tests; six real panels processed; outputs
   inspected; determinism confirmed by byte-identical re-runs.
3. **Scientifically validated** — **not achieved.** No ground-truth annotations exist, so IoU, Dice,
   precision and recall are deliberately **not reported**. Segmentation accuracy is unproven.

### Scoring against your own annotations

The metrics are implemented and waiting for data. Put annotation files in `data/annotations/`, named
by image stem:

| File | Interpretation |
|---|---|
| `<stem>_void_mask.png` (or `.tif`) | binary mask: non-zero = void |
| `<stem>_classes.png` (or `.tif`) | class ids: 0 = ignore, 1 = SURFACE, 2 = MIXED, 3 = VOID |

Then:

```powershell
.\.venv\Scripts\python.exe evaluate.py --outputs outputs\validation --annotations data\annotations
```

This writes `evaluation/segmentation_metrics.csv` (one row per image: TP/FP/FN/TN, IoU, Dice,
precision, recall, plus per-class IoU/Dice) and `evaluation/evaluation_summary.json`. With no
annotations present it prints that no metric is reported and writes an empty table — it never invents
a score. When both masks are empty the metrics are `None` (undefined) rather than a misleading 1.0.

## 17. Known limitations

1. **VOID area is over-estimated.** With `μ_l` and `σ_l` taken over the whole region (as the paper
   states), regions that mix a dark void with a bright rim get a raised `I_v`, admitting off-void dark
   surface. Measured VOID coverage inside VOID regions was ~35 %, and reported VOID area fractions
   ranged 18–59 % across similar surfaces.
2. **First-stage contours can leak.** Linking reach is `k=7` ⇒ 3 px, so wider gaps stay open and
   neighbouring dimples merge. Refinement recovers some, not all.
3. **Very sensitive to `sigma_multiplier`.** Region count changed 30× between 1.0 and 3.0.
4. **`I_t` collides with saturation.** On these images void bottoms clip to 0, giving `I_t = 1`, so
   brightness/contrast settings directly shift the tip threshold.
5. **No physical units** without explicit calibration; no stereological correction.
6. **Grayscale is a depth proxy only**, following the paper. Tilt, detector type and material contrast
   can break the assumption.
7. **Runtime** is ~30–60 s per megapixel-scale panel.

## 18. Reference paper

S. Banerjee, K. Basu, P. C. Chakraborti, S. K. Saha.
*Segmentation of SEM Fractograph into Void and Surface Regions.*
Manuscript supplied at `data/reference/hole_extract.pdf`.

Contour-stage companion:
S. Banerjee, K. Basu, P. C. Chakraborti, S. K. Saha. *Extraction of Region Contour in SEM
Fractographs.* Proc. 2nd Int. Conf. on Emerging Applications of Information Technology, IEEE CS Press,
2011, pp. 98–101. DOI [10.1109/EAIT.2011.20](https://doi.org/10.1109/EAIT.2011.20).

## 19. Adding your own sample images

1. Copy images into `data/sample/` (small set) or `data/raw/` (bulk; git-ignored).
2. Prefer the **original** SEM export (8- or 16-bit TIFF over re-compressed JPEG).
3. Run:
   ```powershell
   .\.venv\Scripts\python.exe main.py -i data\sample\my_image.tif -o outputs\my_image --image-id my_image
   ```
4. Inspect `figures\00_composite_overview.png` first, then `05_refinement.png` to see whether the region
   structure is sensible before trusting any numbers.
5. If boundaries are lost, try `--sigma-multiplier 1.8`; if the image is noisy, `--median-kernel 3`
   (this is a deviation from the paper — record it).

Do not commit bulk data; `data/raw/` is ignored except for `.gitkeep`.

## 20. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Activate.ps1 cannot be loaded` | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or call `.\.venv\Scripts\python.exe` directly |
| `ModuleNotFoundError: No module named 'src'` | not run from the project root | `cd` to the folder containing `main.py` |
| `ImageLoadError: Unsupported image format` | extension outside the list | convert to PNG/TIFF/JPEG |
| `ImageLoadError: Could not decode` | truncated/corrupt file | re-export from the microscope |
| `ERROR: invalid input or parameters: ...` | out-of-range flag | the message names the field; see `--list-parameters` |
| Warning `Void-tip detection failed: ...` | histogram has no local maximum (e.g. uniform image) | expected; no VOID regions can be assigned — check the image |
| Everything is MIXED, nothing VOID | `I_t` low and `I_p > I_t` for every region | inspect `06_void_detection.png`; the surface may genuinely be all type-2 voids |
| VOID far too large | extension over-reach (see §17.1) | known limitation; consider `--extension-sigma-coeff` sweep and record it as a deviation |
| Figures fail to save | output directory not writable | choose a writable `-o` path |
| Run takes >2 minutes | large image, many regions | use `--no-figures`; expect ~30–60 s per MP |

---

## Documentation index

- [`docs/technical_note.md`](docs/technical_note.md) — equations, assumptions, traceability to the paper
- [`docs/algorithm_flowchart.md`](docs/algorithm_flowchart.md) — flowchart and pseudocode
- [`docs/validation_report.md`](docs/validation_report.md) — test results, real-image metrics, failures
- [`docs/validation_runs.csv`](docs/validation_runs.csv) — per-image measured outputs
- [`docs/sensitivity_threshold.csv`](docs/sensitivity_threshold.csv) — sigma-multiplier sweep
- [`docs/presentation_outline.md`](docs/presentation_outline.md) — slide-by-slide handover deck
