"""ROI 检测与透视裁剪（OpenCV 实现）。

整体思路（两条候选策略，取置信度高的结果）：
    策略 A（底色差）：估计图片四周底色 -> 与底色差异大的区域 -> Otsu 二值化
                     -> 形态学闭/开运算 -> 外轮廓 -> 四边形逼近
    策略 B（边缘）：  Canny 边缘 -> 膨胀/闭运算把边连成闭环 -> 外轮廓 -> 四边形逼近
策略 B 主要用来兜底“白底上的白卡片”这种策略 A 看不见的情况。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass
class Region:
    quad: list[list[float]]
    confidence: float
    area_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "quad": [[round(float(x), 1), round(float(y), 1)] for x, y in self.quad],
            "confidence": round(float(self.confidence), 3),
            "area_ratio": round(float(self.area_ratio), 4),
        }


# ---------------------------------------------------------------- 读写图片

def load_bgr(path: str | Path) -> np.ndarray:
    """用 Pillow 读图，顺带修正 EXIF 方向。

    不用 cv2.imread 的原因：它读不了中文/非 ASCII 路径，也不处理 EXIF 旋转，
    会导致“显示方向”和“坐标”错位。
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        rgb = np.array(im)
    return rgb[:, :, ::-1].copy()


def save_png(path: str | Path, bgr: np.ndarray) -> None:
    """cv2.imencode 配合 tofile 写入，兼容非 ASCII 路径。"""
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG 编码失败")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


def fit_long_side(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return bgr
    scale = max_side / float(longest)
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- 几何工具

def order_quad(points: np.ndarray | list) -> np.ndarray:
    """把四个点整理成 左上 -> 右上 -> 右下 -> 左下。"""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError("四边形必须正好 4 个点")
    if len(np.unique(pts, axis=0)) < 4:
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        pts = pts[np.argsort(angles)]
        pts = np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)
        return pts.astype(np.float32)
    summed = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    return np.array(
        [pts[np.argmin(summed)], pts[np.argmax(diff)],
         pts[np.argmax(summed)], pts[np.argmin(diff)],
         ],
        dtype=np.float32,
    )


def polygon_area(points: np.ndarray) -> float:
    contour = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return float(abs(cv2.contourArea(contour)))


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1 = float(a[:, 0].min()), float(a[:, 1].min())
    ax2, ay2 = float(a[:, 0].max()), float(a[:, 1].max())
    bx1, by1 = float(b[:, 0].min()), float(b[:, 1].min())
    bx2, by2 = float(b[:, 0].max()), float(b[:, 1].max())
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedupe(regions: list[Region]) -> list[Region]:
    """同一张卡片常被两条策略各检一次，按包围盒重合度去重，保留置信度高的。"""
    kept: list[Region] = []
    for region in sorted(regions, key=lambda r: -r.confidence):
        quad = np.asarray(region.quad, dtype=np.float32)
        if any(bbox_iou(quad, np.asarray(k.quad, dtype=np.float32)) > 0.6 for k in kept):
            continue
        kept.append(region)
    return kept


# ---------------------------------------------------------------- 检测主流程

def detect(image_bgr: np.ndarray, cfg: dict[str, Any]) -> tuple[list[Region], str | None]:
    """返回 (候选区域列表, 失败原因)。失败原因 None 表示检测成功。"""
    original_h, original_w = image_bgr.shape[:2]
    max_side = int(cfg["detect_max_side"])
    scale = min(1.0, max_side / float(max(original_h, original_w)))

    if scale < 1.0:
        small = cv2.resize(
            image_bgr,
            (max(1, int(round(original_w * scale))), max(1, int(round(original_h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = image_bgr.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    candidates: list[Region] = []
    candidates += _from_background_diff(small, gray, cfg)
    candidates += _from_edges(gray, cfg)
    if not candidates:
        candidates += _from_gradient(gray, cfg)

    regions = dedupe(candidates)
    # 坐标要还原到原图像素，后续裁剪一律在原分辨率的图上做
    if scale != 1.0:
        for region in regions:
            region.quad = [[x / scale, y / scale] for x, y in region.quad]

    regions = [r for r in regions if r.area_ratio <= float(cfg["max_area_ratio"])]
    regions.sort(key=lambda r: -polygon_area(np.asarray(r.quad)))
    regions = regions[: int(cfg["max_regions"])]

    if not regions:
        return [], "no_region"
    if regions[0].confidence < float(cfg["min_confidence"]):
        return regions, "low_confidence"
    return regions, None


def _background_color(bgr: np.ndarray, border_ratio: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    band = max(2, int(min(h, w) * border_ratio))
    samples = np.concatenate(
        [
            bgr[:band].reshape(-1, 3),
            bgr[-band:].reshape(-1, 3),
            bgr[:, :band].reshape(-1, 3),
            bgr[:, -band:].reshape(-1, 3),
        ]
    )
    return np.median(samples, axis=0).astype(np.float32)


def _from_background_diff(
    small: np.ndarray, gray: np.ndarray, cfg: dict[str, Any]
) -> list[Region]:
    background = _background_color(small, float(cfg["border_ratio"]))
    diff = np.abs(small.astype(np.float32) - background.reshape(1, 1, 3)).max(axis=2)
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    # 底色和内容几乎同色时这条策略不可靠，交给边缘策略
    if int(diff.max()) < 12:
        return []

    # 阈值由“画面四周底色噪声”决定，而不是 Otsu：
    # Otsu 在低对比度（白底白卡）时会把卡片内部误判为背景，导致只切到一小块。
    threshold = _noise_threshold(diff, float(cfg["border_ratio"]))
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    k = int(cfg["close_kernel"])
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _regions_from_contours(contours, gray, small.shape[:2], cfg, fill_bonus=True)


def _noise_threshold(diff: np.ndarray, border_ratio: float) -> float:
    """用四周底色区域的噪声水平估算阈值：median + 2.5 * 稳健标准差。"""
    height, width = diff.shape[:2]
    band = max(4, int(min(height, width) * max(border_ratio, 0.03)))
    border = np.concatenate(
        [
            diff[:band].ravel(), diff[-band:].ravel(),
            diff[:, :band].ravel(), diff[:, -band:].ravel(),
        ]
    ).astype(np.float32)
    median = float(np.median(border))
    mad = float(np.median(np.abs(border - median)))
    sigma = 1.4826 * mad
    return max(8.0, median + 2.5 * sigma)


def _from_gradient(gray: np.ndarray, cfg: dict[str, Any]) -> list[Region]:
    """最后兜底：直接找强梯度并闭合成环，适合底色估计失效的场景。"""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    threshold = max(18.0, float(np.percentile(magnitude, 90)))
    edges = (magnitude > threshold).astype(np.uint8) * 255
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)), iterations=2,
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _regions_from_contours(contours, gray, gray.shape[:2], cfg, fill_bonus=False)


def _from_edges(gray: np.ndarray, cfg: dict[str, Any]) -> list[Region]:
    median = float(np.median(gray))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median + 15))
    edges = cv2.Canny(gray, lower, upper, apertureSize=3, L2gradient=True)
    if float(edges.mean()) < 1.0:
        return []
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _regions_from_contours(contours, gray, gray.shape[:2], cfg, fill_bonus=False)


def _regions_from_contours(
    contours: tuple, gray: np.ndarray, shape: tuple[int, int], cfg: dict[str, Any],
    fill_bonus: bool,
) -> list[Region]:
    height, width = shape
    image_area = float(height * width)
    regions: list[Region] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue
        ratio = area / image_area
        if ratio < float(cfg["min_area_ratio"]) or ratio > float(cfg["max_area_ratio"]):
            continue

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        if hull_area <= 0:
            continue
        if area / hull_area < float(cfg["min_solidity"]):
            continue

        quad = _quad_from_contour(hull)
        if quad is None:
            continue

        quad_area = polygon_area(quad)
        if quad_area <= 0:
            continue
        fill = min(1.0, hull_area / quad_area) if fill_bonus else 0.8
        edge = _edge_support(gray, quad)
        confidence = 0.45 * fill + 0.55 * edge

        regions.append(
            Region(
                quad=order_quad(quad).tolist(),
                confidence=confidence,
                area_ratio=quad_area / image_area,
            )
        )
    return regions


def _quad_from_contour(contour: np.ndarray) -> np.ndarray | None:
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None
    for eps in np.arange(0.015, 0.16, 0.01):
        approx = cv2.approxPolyDP(contour, float(eps) * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    # 退化为最小外接旋转矩形（轮廓边缘毛刺太多时的兜底）
    rect = cv2.minAreaRect(contour)
    if min(rect[1]) < 4:
        return None
    return cv2.boxPoints(rect).astype(np.float32)


def _edge_support(gray: np.ndarray, quad: np.ndarray, samples: int = 40) -> float:
    """沿四条边采样梯度强度，作为“这条边真的是物体边界”的置信度依据。"""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    height, width = gray.shape[:2]
    scale = float(np.percentile(magnitude, 99)) + 1e-6

    q = order_quad(quad)
    values: list[float] = []
    for i in range(4):
        start, end = q[i], q[(i + 1) % 4]
        ts = np.linspace(0.03, 0.97, samples)
        xs = np.clip((start[0] + (end[0] - start[0]) * ts).astype(np.int32), 0, width - 1)
        ys = np.clip((start[1] + (end[1] - start[1]) * ts).astype(np.int32), 0, height - 1)
        values.append(float(magnitude[ys, xs].mean()))

    mean_score = float(np.mean(values)) / scale
    median_score = float(np.median(values)) / scale
    return float(np.clip(0.5 * mean_score + 0.5 * median_score, 0.0, 1.0))


# ---------------------------------------------------------------- 裁剪

def crop_region(
    image_bgr: np.ndarray,
    quad: np.ndarray | list,
    inset_px: float = 0.0,
    rotate: int = 0,
    max_long_edge: int | None = None,
) -> np.ndarray:
    """按四点做透视裁剪；inset_px 向内收缩，避免把背景带进来。"""
    height, width = image_bgr.shape[:2]
    q = order_quad(quad)
    q[:, 0] = np.clip(q[:, 0], 0, width - 1)
    q[:, 1] = np.clip(q[:, 1], 0, height - 1)

    if inset_px and inset_px > 0:
        center = q.mean(axis=0)
        distance = np.linalg.norm(q - center, axis=1)
        factor = np.clip(1.0 - float(inset_px) / np.maximum(distance, 1e-6), 0.8, 1.0)
        q = center + (q - center) * factor[:, None]

    top_left, top_right, bottom_right, bottom_left = q
    out_w = max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))
    out_h = max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))
    target_w = max(1, int(round(out_w)))
    target_h = max(1, int(round(out_h)))

    destination = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(q.astype(np.float32), destination)
    output = cv2.warpPerspective(
        image_bgr, matrix, (target_w, target_h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )

    if rotate:
        k = -((int(rotate) // 90) % 4)  # 顺时针
        if k:
            output = np.ascontiguousarray(np.rot90(output, k))

    if max_long_edge:
        output = fit_long_side(output, int(max_long_edge))
    return output
