"""Export stage: downsample, watermark, label naming, write to disk.

All export options are handled here; the front end just passes them through.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw, ImageFont

from .detector import crop_region

POSITIONS = {
    "top-left", "top-center", "top-right",
    "middle-left", "center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
}

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",      # Microsoft YaHei, supports CJK
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "arial.ttf",
    "DejaVuSans.ttf",
]

_INVALID_CHARS = '<>:"/\\|?*\r\n\t'


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def downsample(image: Image.Image, max_long_edge: int) -> Image.Image:
    """Only shrink, never upscale; LANCZOS keeps quality."""
    width, height = image.size
    longest = max(width, height)
    if max_long_edge <= 0 or longest <= max_long_edge:
        return image
    scale = max_long_edge / float(longest)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.LANCZOS)


def add_watermark(
    image: Image.Image,
    text: str,
    position: str = "bottom-right",
    opacity: float = 0.35,
    size_ratio: float = 0.04,
) -> Image.Image:
    if position not in POSITIONS:
        position = "bottom-right"
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = max(12, int(min(image.size) * max(0.01, min(size_ratio, 0.2))))
    font = _load_font(font_size)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = right - left, bottom - top

    margin = int(min(image.size) * 0.03)
    width, height = image.size
    if position.startswith("top"):
        x, y = margin, margin - top
    elif position.startswith("bottom"):
        x, y = margin, height - text_h - margin - top
    else:
        x, y = margin, (height - text_h) // 2 - top
    if position.endswith("right"):
        x = width - text_w - margin - left
    elif position.endswith("center"):
        x = (width - text_w) // 2 - left

    alpha = int(255 * max(0.0, min(1.0, opacity)))
    # Dark stroke keeps the text readable on light photos
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0, alpha))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, alpha))

    return Image.alpha_composite(image, overlay).convert("RGB")


def sanitize_filename(text: str, max_length: int = 90) -> str:
    cleaned = "".join("_" if ch in _INVALID_CHARS else ch for ch in text).strip()
    cleaned = cleaned.replace(" ", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("._")[:max_length]


def build_stem(template: str, name: str, index: int, label: str, when: str) -> str:
    text = template or "{name}_{index}"
    replacements = {
        "{name}": name,
        "{index}": str(index),
        "{label}": label,
        "{date}": when,
    }
    for token, value in replacements.items():
        text = text.replace(token, value)
    stem = sanitize_filename(text)
    return stem or sanitize_filename(f"{name}_{index}")


def _unique_path(directory: Path, stem: str, fmt: str) -> Path:
    suffix = ".jpg" if fmt == "jpg" else ".png"
    candidate = directory / f"{stem}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = directory / f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def save_image(image: Image.Image, path: Path, fmt: str, quality: int) -> None:
    if fmt == "jpg":
        image.convert("RGB").save(path, "JPEG", quality=int(quality), optimize=True)
    else:
        image.save(path, "PNG", optimize=True)


def export_item(
    image_bgr,
    quad,
    options: dict[str, Any],
    index: int,
    source_name: str,
    out_dir: Path,
    inset_px: float,
) -> dict[str, Any]:
    """Crop -> downsample -> watermark -> name by template -> write disk -> optional sidecar."""
    cropped = crop_region(
        image_bgr,
        quad,
        inset_px=inset_px,
        rotate=int(options.get("rotate") or 0),
    )
    image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

    max_edge = options.get("max_long_edge")
    if max_edge:
        image = downsample(image, int(max_edge))

    watermark_text = (options.get("watermark_text") or "").strip()
    if watermark_text:
        image = add_watermark(
            image,
            watermark_text,
            position=options.get("watermark_position") or "bottom-right",
            opacity=float(options.get("watermark_opacity") if options.get("watermark_opacity") is not None else 0.35),
            size_ratio=float(options.get("watermark_size_ratio") if options.get("watermark_size_ratio") is not None else 0.04),
        )

    label = (options.get("label") or "").strip()
    stem = build_stem(
        options.get("filename_template") or "{name}_{index}",
        name=Path(source_name).stem,
        index=index,
        label=label,
        when=datetime.now().strftime("%Y%m%d"),
    )
    fmt = "png" if (options.get("fmt") or "jpg").lower() == "png" else "jpg"
    path = _unique_path(out_dir, stem, fmt)
    save_image(image, path, fmt, int(options.get("quality") or 92))

    if options.get("write_sidecar", True):
        sidecar = path.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                {
                    "output": path.name,
                    "source_name": source_name,
                    "label": label,
                    "quad": [[round(float(x), 1), round(float(y), 1)] for x, y in quad],
                    "rotate": int(options.get("rotate") or 0),
                    "exported_at": datetime.now().isoformat(timespec="seconds"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return {
        "name": path.name,
        "path": str(path),
        "width": image.width,
        "height": image.height,
    }
