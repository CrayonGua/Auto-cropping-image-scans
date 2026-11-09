"""Evaluate auto-detection IoU success on a labelled dataset.

Ground-truth format: a .json with the same name as the image, content
{"quad": [[x, y], [x, y], [x, y], [x, y]]} in original-image pixel coords.

Usage:
    python tools/eval.py --images sample_data/images --labels sample_data/labels
    python tools/eval.py --images sample_data/images --labels sample_data/labels --out eval_report.csv

The reported success rate can go directly into the demo video and report.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_config  # noqa: E402
from app.detector import detect, load_bgr  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MASK_MAX_SIDE = 1200


def polygon_iou(predicted, truth, width: int, height: int) -> float:
    scale = min(1.0, MASK_MAX_SIDE / float(max(width, height)))
    mask_shape = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))

    def rasterize(points) -> np.ndarray:
        pts = (np.asarray(points, dtype=np.float32) * scale).astype(np.int32)
        mask = np.zeros(mask_shape, np.uint8)
        cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 1)
        return mask

    mask_pred = rasterize(predicted)
    mask_true = rasterize(truth)
    intersection = int(np.logical_and(mask_pred, mask_true).sum())
    union = int(np.logical_or(mask_pred, mask_true).sum())
    return intersection / union if union else 0.0


def evaluate(images_dir: Path, labels_dir: Path, output_csv: Path | None) -> int:
    config = load_config()
    detector_cfg = config["detector"]

    image_paths = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        print(f"No images found in: {images_dir}")
        return 1

    rows: list[dict] = []
    skipped: list[str] = []

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.json"
        if not label_path.exists():
            skipped.append(image_path.name)
            continue

        truth = json.loads(label_path.read_text(encoding="utf-8"))["quad"]
        image = load_bgr(image_path)
        height, width = image.shape[:2]

        regions, reason = detect(image, detector_cfg)
        if regions:
            best = max(regions, key=lambda region: polygon_iou(region.quad, truth, width, height))
            iou = polygon_iou(best.quad, truth, width, height)
            confidence = best.confidence
        else:
            iou = 0.0
            confidence = 0.0

        status = "ok" if iou >= 0.85 else ("partial" if iou >= 0.70 else "fail")
        rows.append(
            {
                "image": image_path.name,
                "iou": round(iou, 4),
                "status": status,
                "confidence": round(confidence, 3),
                "regions": len(regions),
                "reason": reason or "",
            }
        )

    if not rows:
        print("No images have ground-truth labels; nothing to evaluate.")
        return 1

    ok = sum(1 for row in rows if row["status"] == "ok")
    partial = sum(1 for row in rows if row["status"] == "partial")
    failed = len(rows) - ok - partial
    mean_iou = sum(row["iou"] for row in rows) / len(rows)

    print(f"{'Image':<38}{'IoU':>8}{'Status':>10}{'Conf':>10}{'Regions':>8}")
    for row in rows:
        print(
            f"{row['image']:<38}{row['iou']:>8.3f}{row['status']:>10}"
            f"{row['confidence']:>10.3f}{row['regions']:>8}"
        )

    print("-" * 76)
    print(f"Samples {len(rows)} | mean IoU {mean_iou:.3f}")
    print(f"Success (IoU>=0.85): {ok} ({ok / len(rows) * 100:.1f}%)")
    print(f"Partial (0.70-0.85): {partial}")
    print(f"Failed (<0.70): {failed}")
    if skipped:
        print(f"Skipped (no label): {len(skipped)}: {', '.join(skipped)}")

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["image", "iou", "status", "confidence", "regions", "reason"],
            )
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow({})
            writer.writerow({"image": "SUMMARY", "iou": round(mean_iou, 4),
                             "status": f"ok={ok}, partial={partial}, fail={failed}"})
        print(f"Evaluation written to: {output_csv}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate auto-detection IoU success")
    parser.add_argument("--images", default=str(ROOT / "sample_data" / "images"))
    parser.add_argument("--labels", default=str(ROOT / "sample_data" / "labels"))
    parser.add_argument("--out", default=None, help="optional: output CSV path")
    args = parser.parse_args()

    output = Path(args.out) if args.out else None
    return evaluate(Path(args.images), Path(args.labels), output)


if __name__ == "__main__":
    sys.exit(main())
