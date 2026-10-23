"""Score segmentation runs against manual annotations.

Usage (PowerShell, from the project root)::

    .\\.venv\\Scripts\\python.exe evaluate.py --outputs outputs\\validation --annotations data\\annotations

Each run directory under ``--outputs`` is matched to an annotation in
``--annotations`` by the image file name (see ``src.evaluation`` for the naming
rules).  With no annotations present the command reports that plainly and exits
with code 0; it never invents a score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.evaluation import aggregate, evaluate_run, results_to_frame


def find_runs(outputs_root: Path) -> list[Path]:
    """Directories that contain a run_metadata.json."""
    return sorted(p.parent for p in outputs_root.rglob("run_metadata.json"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Compute IoU / Dice / precision / recall against annotations.",
    )
    p.add_argument("--outputs", type=Path, required=True,
                   help="directory containing one or more pipeline run folders")
    p.add_argument("--annotations", type=Path, required=True,
                   help="directory holding annotation masks")
    p.add_argument("--out", type=Path, default=None,
                   help="write a results CSV/JSON here [default: <outputs>/evaluation]")
    args = p.parse_args(argv)

    runs = find_runs(args.outputs)
    if not runs:
        print(f"No runs found under {args.outputs} (looked for run_metadata.json).")
        return 2

    results = []
    for run in runs:
        meta = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
        image_path = meta["image"]["path"]
        r = evaluate_run(run, image_path, args.annotations)
        r["run_dir"] = str(run)
        results.append(r)

    scored = [r for r in results if r.get("status") == "ok"]
    df = results_to_frame(results)
    agg = aggregate(results)

    print(f"Runs found      : {len(results)}")
    print(f"Scored against  : {agg['n_images']} annotation(s)")
    if not scored:
        print()
        print("No annotations were found, so no metric is reported.")
        print("Add masks named '<image stem>_void_mask.png' (binary) or")
        print("'<image stem>_classes.png' (0=ignore, 1=SURFACE, 2=MIXED, 3=VOID)")
        print(f"to: {args.annotations}")
        print()
        print("Metadata of the first run inspected:")
        first = results[0]
        print(f"  {first['image']}: {first.get('message', '')}")
    else:
        cols = [c for c in ("image", "void_iou", "void_dice", "void_precision",
                            "void_recall", "void_tp", "void_fp", "void_fn") if c in df.columns]
        print()
        print(df[cols].to_string(index=False))
        print()
        print("VOID mask, mean over scored images:")
        for key in ("iou", "dice", "precision", "recall"):
            v = agg[f"void_{key}_mean"]
            print(f"  {key:<10s} {v:.4f}" if v is not None else f"  {key:<10s} undefined")

    out_dir = args.out or (args.outputs / "evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "segmentation_metrics.csv", index=False)
    (out_dir / "evaluation_summary.json").write_text(
        json.dumps({"aggregate": agg, "per_image": results}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nWritten: {out_dir / 'segmentation_metrics.csv'}")
    print(f"Written: {out_dir / 'evaluation_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
