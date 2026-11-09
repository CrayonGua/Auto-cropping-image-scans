"""Evaluate detected crop candidates against labelled sample images."""

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
MATCH_THRESHOLD = 0.70


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


def _load_quads(label_path: Path) -> list[list[list[float]]]:
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    if "quads" in payload:
        quads = payload["quads"]
    elif "quad" in payload:
        quads = [payload["quad"]]
    else:
        raise ValueError(f"Label file must contain 'quad' or 'quads': {label_path}")
    if not isinstance(quads, list) or not quads:
        raise ValueError(f"Label file contains no quadrilaterals: {label_path}")
    return quads


def _match_candidates(candidate_quads, truth_quads, width: int, height: int):
    scores = [
        (polygon_iou(candidate, truth, width, height), candidate_index, truth_index)
        for candidate_index, candidate in enumerate(candidate_quads)
        for truth_index, truth in enumerate(truth_quads)
    ]
    scores.sort(reverse=True)
    matched_candidates: set[int] = set()
    matched_truth: set[int] = set()
    target_ious = [0.0] * len(truth_quads)
    for score, candidate_index, truth_index in scores:
        if score < MATCH_THRESHOLD:
            break
        if candidate_index in matched_candidates or truth_index in matched_truth:
            continue
        matched_candidates.add(candidate_index)
        matched_truth.add(truth_index)
        target_ious[truth_index] = score
    return target_ious, len(matched_candidates)


def evaluate(images_dir: Path, labels_dir: Path, output_csv: Path) -> int:
    detector_cfg = load_config()["detector"]
    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        print(f"No supported images found in: {images_dir}")
        return 1

    rows: list[dict] = []
    skipped: list[str] = []
    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.json"
        if not label_path.exists():
            skipped.append(image_path.name)
            continue

        truth_quads = _load_quads(label_path)
        image = load_bgr(image_path)
        height, width = image.shape[:2]
        regions, _reason = detect(image, detector_cfg)
        candidate_quads = [region.quad for region in regions]
        target_ious, matched = _match_candidates(candidate_quads, truth_quads, width, height)
        default_iou = (
            max(polygon_iou(candidate_quads[0], truth, width, height) for truth in truth_quads)
            if candidate_quads
            else 0.0
        )
        ok_targets = sum(score >= 0.85 for score in target_ious)
        partial_targets = sum(MATCH_THRESHOLD <= score < 0.85 for score in target_ious)
        failed_targets = len(target_ious) - ok_targets - partial_targets
        false_positives = max(0, len(candidate_quads) - matched)
        mean_iou = sum(target_ious) / len(target_ious)
        status = "ok" if ok_targets == len(target_ious) and false_positives == 0 else (
            "partial" if matched else "fail"
        )
        rows.append(
            {
                "image": image_path.name,
                "targets": len(truth_quads),
                "candidates": len(candidate_quads),
                "mean_iou": round(mean_iou, 4),
                "matched_targets": matched,
                "false_positives": false_positives,
                "default_candidate_iou": round(default_iou, 4),
                "ok_targets": ok_targets,
                "partial_targets": partial_targets,
                "failed_targets": failed_targets,
                "status": status,
            }
        )

    if not rows:
        print("No labelled images were available for evaluation.")
        return 1

    target_count = sum(row["targets"] for row in rows)
    ok_count = sum(row["ok_targets"] for row in rows)
    partial_count = sum(row["partial_targets"] for row in rows)
    failed_count = sum(row["failed_targets"] for row in rows)
    matched_count = sum(row["matched_targets"] for row in rows)
    candidate_count = sum(row["candidates"] for row in rows)
    false_positives = sum(row["false_positives"] for row in rows)
    mean_iou = sum(row["mean_iou"] * row["targets"] for row in rows) / target_count
    precision = matched_count / candidate_count if candidate_count else 0.0
    recall = matched_count / target_count if target_count else 0.0

    print(f"{'Image':<34}{'GT':>5}{'Candidates':>12}{'Mean IoU':>12}{'Default IoU':>14}{'Status':>10}")
    for row in rows:
        print(
            f"{row['image']:<34}{row['targets']:>5}{row['candidates']:>12}"
            f"{row['mean_iou']:>12.3f}{row['default_candidate_iou']:>14.3f}{row['status']:>10}"
        )
    print("-" * 87)
    print(f"Labelled targets: {target_count}; mean IoU: {mean_iou:.3f}")
    print(f"Targets: {ok_count} strong (>=0.85), {partial_count} partial (0.70-0.85), {failed_count} missed/low (<0.70)")
    print(f"Candidate precision at IoU >= 0.70: {precision:.1%}; candidate recall: {recall:.1%}; false positives: {false_positives}")
    if skipped:
        print(f"Skipped without labels ({len(skipped)}): {', '.join(skipped)}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image", "targets", "candidates", "mean_iou", "matched_targets",
        "false_positives", "default_candidate_iou", "ok_targets",
        "partial_targets", "failed_targets", "status",
    ]
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({})
        writer.writerow(
            {
                "image": "SUMMARY",
                "targets": target_count,
                "candidates": candidate_count,
                "mean_iou": round(mean_iou, 4),
                "matched_targets": matched_count,
                "false_positives": false_positives,
                "default_candidate_iou": "",
                "ok_targets": ok_count,
                "partial_targets": partial_count,
                "failed_targets": failed_count,
                "status": f"precision={precision:.4f}; recall={recall:.4f}",
            }
        )
    print(f"Evaluation report written to: {output_csv}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate crop candidates against labelled images.")
    parser.add_argument("--images", default=str(ROOT / "sample_data" / "images"))
    parser.add_argument("--labels", default=str(ROOT / "sample_data" / "labels"))
    parser.add_argument("--out", default=str(ROOT / "eval_report.csv"), help="CSV output path")
    args = parser.parse_args()
    return evaluate(Path(args.images), Path(args.labels), Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
