# Annotation Guide — how to produce ground truth for scoring

This is the one outstanding item needed to turn "verified" into "scientifically validated".
The evaluator is already written (`evaluate.py`); it only needs masks.

---

## 1. What to annotate

For each image you want to score, produce **one** file in `data/annotations/`, named after the input
image's stem:

| Image | Annotation file | Kind |
|---|---|---|
| `data/sample/sem_dimples_deep.jpeg` | `sem_dimples_deep_void_mask.png` | binary void mask |
| `data/sample/sem_fig4_panel.jpeg` | `sem_fig4_panel_classes.png` | three-class map |

Two acceptable formats:

### Option A — binary void mask (quickest, scores VOID only)

- 2-D grayscale PNG or TIFF, same width and height as the input image.
- Any non-zero pixel (or any non-black colour pixel) means **void**.
- Everything else is treated as **not void**.
- Use this when your question is "how well are the deep voids found?"

### Option B — three-class map (scores VOID, MIXED and SURFACE separately)

- 2-D `uint8` PNG or TIFF, same dimensions as the input.
- Pixel values must be exactly:

| Value | Class |
|---|---|
| `0` | **ignore** — boundary pixels, ambiguous areas, image border. Excluded from every metric. |
| `1` | SURFACE |
| `2` | MIXED |
| `3` | VOID |

- Any other value is rejected with an error rather than guessed at.

> Do **not** save annotations as JPEG. JPEG compression changes pixel values and will silently corrupt
> class ids.

---

## 2. Annotation rules (make these explicit and keep them constant)

These rules matter more than the tool you use, because they define what "correct" means. Write them
down with the masks you produce.

1. **VOID = a cavity you can see going into the surface.** Not merely a dark patch. Use the shading and
   the presence of a rim/depth gradient as evidence.
2. **MIXED = surface carrying shallow dimples with no visible depth.**
3. **SURFACE = relatively smooth or ridged fracture surface with no cavity.**
4. **Tear ridges and bright edges are SURFACE**, even when strongly shadowed on one side.
5. **Ambiguous pixels → 0 (ignore).** It is better to exclude than to guess; ignoring small boundary
   bands is normal and does not inflate the score.
6. **Do not use the algorithm's own output as a starting point** unless you genuinely re-verify every
   boundary by eye. "Annotation" copied from a prediction measures nothing.
7. Annotate at the **original resolution**. Do not annotate a resized preview.
8. Decide up front whether saturated-black regions that are *out of focus* or *beyond the edge of the
   specimen* are void. Recommended: mark them **0 (ignore)** and record that decision.

---

## 3. Suggested workflow

1. Copy the image into an editor that supports layered masks (GIMP, Krita, Photoshop, or Fiji/ImageJ).
2. Fill VOID in pure red `(255,0,0)`, MIXED in pure yellow `(255,255,0)`, leave SURFACE black
   `(0,0,0)`, mark ignore in pure green `(0,255,0)` — so the colour coding is unambiguous on screen.
3. Export the mask, then convert colours to class ids:

```python
import cv2
import numpy as np

mask = cv2.imread("drawn.png", cv2.IMREAD_COLOR)          # BGR
out = np.zeros(mask.shape[:2], np.uint8)                  # default 0 = ignore
out[(mask[:, :, 2] > 200) & (mask[:, :, 1] < 80)] = 3      # red   -> VOID
out[(mask[:, :, 2] > 200) & (mask[:, :, 1] > 200)] = 2     # yellow-> MIXED
out[(mask[:, :, 2] < 60) & (mask[:, :, 1] < 60) & (mask[:, :, 0] < 60)] = 1  # black -> SURFACE
cv2.imwrite("data/annotations/my_image_classes.png", out)
```

4. Check the class histogram before trusting the file:

```python
import cv2, numpy as np
m = cv2.imread("data/annotations/my_image_classes.png", 0)
print({int(k): int(v) for k, v in zip(*np.unique(m, return_counts=True))})
# expect roughly: 0: boundary band, 1: large, 2: medium, 3: small
```

5. Run the scorer.

---

## 4. How many images?

Metrics on one image are an anecdote. For a defensible statement you need enough images to show
variance — the brief suggests ten to twenty per condition. As a practical minimum:

- **≥ 5 images** to see whether performance is stable,
- **≥ 10 per magnification** (×500, ×1000, ×2000 as in the paper) to make a claim per scale,
- include deliberately difficult cases (low contrast, heavy saturation) so the weakness is documented
  rather than avoided.

Report `n_images` alongside every mean. The evaluator already does this.

---

## 5. Interpreting the output

Run:

```powershell
.\.venv\Scripts\python.exe evaluate.py --outputs outputs\validation --annotations data\annotations
```

Outputs:

- `outputs/validation/evaluation/segmentation_metrics.csv` — one row per image.
- `outputs/validation/evaluation/evaluation_summary.json` — per-image details plus aggregate.

Read it as follows:

| Metric | What a low value means here |
|---|---|
| **IoU** | The predicted void area does not coincide with the annotated cavities. Expect this to be the harshest number, because a small boundary error on a large region costs little but a systematic over-extension costs a lot. |
| **Dice** | Same information as IoU, more forgiving of small objects. |
| **Precision** | The algorithm is marking things as void that are not. Given the known over-extension (§6.1 of the validation report), **expect precision to fail before recall**. |
| **Recall** | The algorithm is missing real cavities. |

If precision is low while recall is high, that confirms the documented over-extension rather than
contradicting it. If both are low, the contour stage is failing on that image.

---

## 6. What not to do

- Do not report the numbers in `docs/validation_report.md` §7.1 (TP 288,106 etc.) as accuracy — they
  come from a deliberately perturbed copy of a prediction, used only to prove the metric code works.
- Do not score an image whose annotation was derived from the pipeline output.
- Do not average a metric over fewer images than you report.
- Do not drop the images where the method performs badly; report them and explain why.
