"""Generate deterministic synthetic images and ground-truth crop labels.

Use these samples for development checks only. They are not representative of
real scans. Run with: python tools/make_demo_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "sample_data" / "images"
LABEL_DIR = ROOT / "sample_data" / "labels"


def order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    summed = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    return np.array(
        [pts[np.argmin(summed)], pts[np.argmax(diff)], pts[np.argmax(summed)], pts[np.argmin(diff)]],
        dtype=np.float32,
    )


def make_background(width: int, height: int, color: tuple, seed: int, noise: float = 5.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    canvas = np.full((height, width, 3), color, np.float32)
    canvas += rng.normal(0, noise, canvas.shape)
    # Add low-frequency texture to imitate uneven paper or fabric.
    low = cv2.GaussianBlur(rng.normal(0, 1, (height, width)).astype(np.float32), (0, 0), sigmaX=30)
    low = (low - low.mean()) / (low.std() + 1e-6)
    canvas += (low * noise * 3.0)[:, :, None]
    return np.clip(canvas, 0, 255).astype(np.uint8)


def make_card(width: int, height: int, seed: int, border_ratio: float = 0.0, hue=(190, 170, 150)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    gx = x / max(width - 1, 1)
    gy = y / max(height - 1, 1)

    card = np.zeros((height, width, 3), np.float32)
    for channel in range(3):
        card[:, :, channel] = hue[channel] * (0.78 + 0.16 * gx) + 12.0 * gy

    # Add varied content blocks so the card has visible internal edges.
    for _ in range(7):
        box_w = int(rng.integers(max(8, width // 14), max(12, width // 4)))
        box_h = int(rng.integers(max(8, height // 14), max(12, height // 4)))
        x0 = int(rng.integers(0, max(1, width - box_w)))
        y0 = int(rng.integers(0, max(1, height - box_h)))
        card[y0:y0 + box_h, x0:x0 + box_w] = rng.integers(25, 235, 3).astype(np.float32)

    cv2.putText(card, "PHOTO", (int(width * 0.06), int(height * 0.9)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.5, width / 900.0), (250, 250, 250), 2, cv2.LINE_AA)
    card += rng.normal(0, 3.0, card.shape)
    card = np.clip(card, 0, 255).astype(np.uint8)

    if border_ratio > 0:
        border = max(2, int(min(width, height) * border_ratio))
        card[:border] = 246
        card[-border:] = 246
        card[:, :border] = 246
        card[:, -border:] = 246
        cv2.rectangle(card, (border, border), (width - border - 1, height - border - 1), (175, 175, 175), 1)
    return card


def place_card(background: np.ndarray, card: np.ndarray, quad: np.ndarray) -> None:
    height, width = background.shape[:2]
    card_h, card_w = card.shape[:2]
    source = np.float32([[0, 0], [card_w - 1, 0], [card_w - 1, card_h - 1], [0, card_h - 1]])
    matrix = cv2.getPerspectiveTransform(source, np.float32(quad))
    warped = cv2.warpPerspective(card, matrix, (width, height))
    mask = cv2.warpPerspective(np.full((card_h, card_w), 255, np.uint8), matrix, (width, height))
    background[mask > 0] = warped[mask > 0]


def rotated_quad(center_x: float, center_y: float, width: float, height: float, angle: float) -> np.ndarray:
    box = cv2.boxPoints(((center_x, center_y), (width, height), angle))
    return order_points(box)


def save_jpg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError(f"Could not save image: {path}")
    buffer.tofile(str(path))


def save_label(path: Path, quad: np.ndarray | None = None, quads: list[np.ndarray] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if quads is not None:
        payload = {"quads": [[[round(float(x), 1), round(float(y), 1)] for x, y in order_points(item)] for item in quads]}
    elif quad is not None:
        payload = {"quad": [[round(float(x), 1), round(float(y), 1)] for x, y in order_points(quad)]}
    else:
        raise ValueError("Provide one quadrilateral or a list of quadrilaterals.")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_samples() -> None:
    # 1) A nearly straight photo with a white border on a light-grey surface.
    canvas = make_background(1400, 1900, (228, 228, 228), seed=1)
    card = make_card(960, 660, seed=11)
    quad = rotated_quad(700, 950, 960, 660, 0.3)
    place_card(canvas, card, quad)
    save_jpg(IMAGE_DIR / "scan_01_white_card.jpg", canvas)
    save_label(LABEL_DIR / "scan_01_white_card.json", quad)

    # 2) A white-bordered postcard rotated by 8 degrees on dark fabric.
    canvas = make_background(1400, 1900, (38, 42, 48), seed=2, noise=7)
    card = make_card(1000, 700, seed=12, border_ratio=0.06, hue=(180, 150, 120))
    quad = rotated_quad(700, 950, 1000, 700, -8.0)
    place_card(canvas, card, quad)
    save_jpg(IMAGE_DIR / "scan_02_postcard_dark.jpg", canvas)
    save_label(LABEL_DIR / "scan_02_postcard_dark.json", quad)

    # 3) Two separate photos on one page for multi-region detection.
    canvas = make_background(1700, 2100, (214, 214, 214), seed=3)
    multi_quads = [
        rotated_quad(480, 700, 760, 540, 3.0),
        rotated_quad(1140, 1420, 700, 500, -4.0),
    ]
    place_card(canvas, make_card(760, 540, seed=13), multi_quads[0])
    place_card(canvas, make_card(700, 500, seed=14, hue=(150, 175, 190)), multi_quads[1])
    save_jpg(IMAGE_DIR / "scan_03_two_photos.jpg", canvas)
    save_label(LABEL_DIR / "scan_03_two_photos.json", quads=multi_quads)

    # 4) A white card on a white background; edge detection is important.
    canvas = make_background(1500, 2000, (247, 247, 247), seed=4, noise=2.5)
    card = make_card(880, 620, seed=15, hue=(235, 235, 235))
    quad = rotated_quad(750, 1000, 880, 620, 1.5)
    place_card(canvas, card, quad)
    save_jpg(IMAGE_DIR / "scan_04_white_on_white.jpg", canvas)
    save_label(LABEL_DIR / "scan_04_white_on_white.json", quad)

    # 5) A perspective-skewed card for perspective-correction checks.
    canvas = make_background(1500, 2000, (225, 220, 210), seed=5)
    card = make_card(820, 580, seed=16)
    base = order_points(np.array([[300, 500], [1120, 440], [1180, 1020], [250, 1090]], np.float32))
    place_card(canvas, card, base)
    save_jpg(IMAGE_DIR / "scan_05_perspective.jpg", canvas)
    save_label(LABEL_DIR / "scan_05_perspective.json", base)

    # 6) A small print for minimum-area threshold checks.
    canvas = make_background(1600, 2000, (60, 64, 70), seed=6, noise=6)
    card = make_card(420, 320, seed=17, hue=(170, 160, 140))
    quad = rotated_quad(800, 1000, 420, 320, -12.0)
    place_card(canvas, card, quad)
    save_jpg(IMAGE_DIR / "scan_06_small_print.jpg", canvas)
    save_label(LABEL_DIR / "scan_06_small_print.json", quad)

    print(f"Generated sample images: {IMAGE_DIR}")
    print(f"Generated ground-truth labels: {LABEL_DIR}")


if __name__ == "__main__":
    sys.exit(build_samples())
