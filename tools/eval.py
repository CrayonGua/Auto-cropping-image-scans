"""在带真值的数据集上评估自动检测的 IoU 成功率。

真值格式：与图片同名的 .json，内容为 {"quad": [[x, y], [x, y], [x, y], [x, y]]}（原图像素坐标）。

用法：
    python tools/eval.py --images sample_data/images --labels sample_data/labels
    python tools/eval.py --images sample_data/images --labels sample_data/labels --out eval_report.csv

输出的成功率可以直接放进演示视频和报告里。
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
        print(f"没有找到图片：{images_dir}")
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
        print("所有图片都缺少真值标注，未做评估。")
        return 1

    ok = sum(1 for row in rows if row["status"] == "ok")
    partial = sum(1 for row in rows if row["status"] == "partial")
    failed = len(rows) - ok - partial
    mean_iou = sum(row["iou"] for row in rows) / len(rows)

    print(f"{'图片':<38}{'IoU':>8}{'判定':>10}{'置信度':>10}{'候选数':>8}")
    for row in rows:
        print(
            f"{row['image']:<38}{row['iou']:>8.3f}{row['status']:>10}"
            f"{row['confidence']:>10.3f}{row['regions']:>8}"
        )

    print("-" * 76)
    print(f"样本数 {len(rows)}｜平均 IoU {mean_iou:.3f}")
    print(f"成功(IoU>=0.85) {ok} ({ok / len(rows) * 100:.1f}%)")
    print(f"部分成功(0.70-0.85) {partial}")
    print(f"失败(<0.70) {failed}")
    if skipped:
        print(f"跳过（无真值标注）{len(skipped)}：{', '.join(skipped)}")

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
        print(f"评估结果已写入：{output_csv}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="评估自动检测的 IoU 成功率")
    parser.add_argument("--images", default=str(ROOT / "sample_data" / "images"))
    parser.add_argument("--labels", default=str(ROOT / "sample_data" / "labels"))
    parser.add_argument("--out", default=None, help="可选：输出 CSV 路径")
    args = parser.parse_args()

    output = Path(args.out) if args.out else None
    return evaluate(Path(args.images), Path(args.labels), output)


if __name__ == "__main__":
    sys.exit(main())
