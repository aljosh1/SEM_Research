# Presentation Outline — 10 Slides

Editable format not generated (no PowerPoint generator available in this environment). Slide titles,
content bullets, and the **exact figure file** to paste into each slide are given below. All figures
referenced exist in this repository.

**Suggested demo sequence (10 min):** run the CLI live on `data/sample/sem_dimples_deep.jpeg` (≈40 s),
then walk the audience through the generated `figures/` folder in numbered order.

---

## Slide 1 — Title

**Automated Segmentation of SEM Fractographs into Void and Surface Regions**

- Implementation of Banerjee, Basu, Chakraborti & Saha
- Conventional image processing · Python · reproducible runs
- Presenter / date / institute

*Figure:* none, or `data/sample/sem_dimples_deep.jpeg` as a full-bleed background.

---

## Slide 2 — Scientific background

- Ductile fracture = void nucleation → growth → coalescence
- Dimples on the fracture surface are the void halves
- Void size/shape/distribution correlate with tensile properties
- Manual image editing is slow, subjective, and analyst-dependent — hence automation

*Figure:* `data/sample/sem_dimples_deep.jpeg` (annotate: "five deep voids", "shallow dimple field").

---

## Slide 3 — Reference-paper methodology

Three stages, in the authors' own terms:

1. **Detection of region contour** — `th = μ + 2σ` over the intermediate histogram range; thinning;
   Sobel-direction edge linking (k = 5 → 7); region-growing pruning
2. **Refinement of region contour** — only for `S_i > μ_S + σ_S`; local `T_local = μ_l + 0.5σ_l`;
   reduced linking neighbourhood
3. **Region characterization** — `I_t` from the first low-range histogram peak;
   `I_v = max(μ_l − 0.5σ_l, I_t)`; classify with `I_p ≤ I_t` and `I_c < μ_g + jσ_g` (k=80, j=1)

*Figure:* a simple 3-stage arrow diagram (build in PowerPoint from the text above).

---

## Slide 4 — Algorithm flowchart

- Image → load/preserve → optional preprocessing
- → threshold → thin → link → region-grow → label
- → candidate selection → local re-threshold → split
- → void tips → extension → classification
- → quantification → CSV + figures + metadata

*Figure:* screenshot of `docs/algorithm_flowchart.md` (render the Mermaid block, or paste its
pseudocode).

---

## Slide 5 — Software architecture

- `src/` — one module per scientific stage, each with a paper-section reference in its docstring
- `src/config.py` — every parameter tagged **paper / implementation / extension**
- `src/pipeline.py` — orchestration, output naming, provenance
- `main.py` — CLI (`--help`, `--list-parameters`, calibration flags)
- `tests/` — 159 tests
- `evaluate.py` — annotation scoring (IoU / Dice / precision / recall), ready but unscored
- No deep learning anywhere

*Figure:* directory tree (from README §6).

---

## Slide 6 — Contour detection and refinement

- `th = μ+2σ` with extreme tails excluded → boundary shell (11.7 % of pixels on the demo image)
- Thinning → 1-px skeleton; direction-matched linking, k = 5 → 7, iterative, re-thinned
- Region growing prunes open curves
- Refinement splits oversized regions at `T_local`

**Measured:** 189 initial regions → 845 after refinement (18 candidate regions).

*Figure:* `outputs/validation/sem_dimples_deep/figures/04_initial_contour.png`
then `05_refinement.png` — the "before/after" that carries the slide.

---

## Slide 7 — Void detection and classification

- `I_t = 1` on this image (void bottoms saturate to 0) → tip mask
- Extension per region with `I_v = max(μ_l − 0.5σ_l, I_t)`
- Class rules applied in the paper's order: no void → SURFACE; fully covered → VOID;
  else `I_p ≤ I_t` **and** `I_c < μ_g + jσ_g`

**Measured on demo:** VOID 19 regions / MIXED 207 / SURFACE 619.

*Figure:* `06_void_detection.png`, then `07_final_segmentation.png`.

---

## Slide 8 — Quantitative measurements

- Per void: area, `D_eq = sqrt(4A/π)`, major/minor axis, aspect ratio, perimeter, circularity
- Per region: class, area, fraction, void fraction, `I_p`, `I_c`, decision reason
- Summary: area fractions, size statistics, number density (only if calibrated)
- Physical units **only** with a verified pixel size; never inferred from magnification

**Measured (demo):** 1,343 individual voids; VOID 49.2 %, MIXED 36.6 %, SURFACE 14.2 % of area.

*Figure:* `09_void_size_distribution.png` and `10_summary.png`. Show the CSV header row.

---

## Slide 9 — Experimental results and validation

- **Verified:** 159 tests pass; deterministic re-runs; six real SEM panels processed end-to-end
- **Sensitivity:** region count changes 30× as `sigma_multiplier` goes 1.0 → 3.0
- **NOT scientifically validated:** no ground truth ⇒ no IoU / Dice / precision / recall claimed
- **Evaluator built and ready** (`evaluate.py`): drop annotated masks into `data/annotations/` and the
  metrics compute automatically. It currently scores 0 images.
- Known defect: VOID area inflated (only ~35 % of a VOID region is actually void) because the paper's
  `μ_l ≈ I_t` premise fails on these images

*Figure:* `docs/validation_runs.csv` as a table (6 rows), plus `08_side_by_side.png`.
State the negative result explicitly — it is the most important slide.

---

## Slide 10 — Limitations and future work

- Annotate 10–20 panels per condition → the evaluator computes IoU/Dice immediately
  (`docs/annotations_guide.md` gives the file format and annotation rules)
- Independent depth ground truth (profilometry / stereo SEM) to calibrate "considerable depth"
- Restrict `μ_l` to the tip neighbourhood (a labelled deviation, not a silent change)
- Add stereological correction for projected void counts
- GUI wrapper with sliders for `sigma_multiplier` and live side-by-side preview

*Figure:* `outputs/validation/sem_dimples_deep/figures/00_composite_overview.png` as a closing summary.

---

## Speaker notes — what to say about the weakness

> "The implementation follows the published equations exactly, and I can point to the line of code for
> each one. What I cannot claim is that the resulting VOID percentage is physically correct. The paper
> assumes that for a deep void the region mean intensity stays close to the void-tip cutoff; on these
> images the region mean sits 55–80 levels above it, so the extension step admits dark surface pixels
> that are not part of the void. That is why I report the software as verified but the science as
> unvalidated — and why the next step is annotation, not more tuning."
