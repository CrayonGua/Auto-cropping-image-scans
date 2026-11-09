"""Export crops with optional downsampling, watermarking, and sidecar metadata."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageColor, ImageDraw, ImageFont

from .detector import crop_region

POSITIONS = {
    "top-left", "top-center", "top-right",
    "middle-left", "center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
}

_FONT_CANDIDATES = {
    "sans": [r"C:\Windows\Fonts\arial.ttf", "arial.ttf", "DejaVuSans.ttf"],
    "serif": [r"C:\Windows\Fonts\times.ttf", "times.ttf", "DejaVuSerif.ttf"],
    "mono": [r"C:\Windows\Fonts\consola.ttf", "consola.ttf", "DejaVuSansMono.ttf"],
}

_INVALID_CHARS = '<>:"/\\|?*\r\n\t'


def _load_font(size: int, family: str = "sans") -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES.get(family, _FONT_CANDIDATES["sans"]):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def downsample(image: Image.Image, max_long_edge: int) -> Image.Image:
    """Downsample without enlarging, using the LANCZOS filter."""
    width, height = image.size
    longest = max(width, height)
    if max_long_edge <= 0 or longest <= max_long_edge:
        return image
    scale = max_long_edge / float(longest)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.LANCZOS)


def _watermark_position(position: str, image_size: tuple[int, int], mark_size: tuple[int, int], margin: int) -> tuple[int, int]:
    width, height = image_size
    mark_w, mark_h = mark_size
    horizontal = "left" if position.endswith("left") else "right" if position.endswith("right") else "center"
    vertical = "top" if position.startswith("top") else "bottom" if position.startswith("bottom") else "middle"
    x = margin if horizontal == "left" else width - mark_w - margin if horizontal == "right" else (width - mark_w) // 2
    y = margin if vertical == "top" else height - mark_h - margin if vertical == "bottom" else (height - mark_h) // 2
    return max(0, x), max(0, y)


def add_watermark(
    image: Image.Image,
    text: str,
    position: str = "bottom-right",
    opacity: float = 0.55,
    size_ratio: float = 0.05,
    color: str = "#FFFFFF",
    font_family: str = "sans",
    style: str = "outline",
) -> Image.Image:
    if position not in POSITIONS:
        position = "bottom-right"
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_size = max(12, int(min(base.size) * max(0.01, min(size_ratio, 0.2))))
    font = _load_font(font_size, font_family)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = right - left, bottom - top
    margin = max(8, int(min(base.size) * 0.03))
    x, y = _watermark_position(position, base.size, (text_w, text_h), margin)
    x -= left
    y -= top
    alpha = int(255 * max(0.0, min(1.0, opacity)))
    rgb = ImageColor.getrgb(color)
    if style == "badge":
        pad_x = max(7, font_size // 3)
        pad_y = max(4, font_size // 5)
        box = (x + left - pad_x, y + top - pad_y, x + right + pad_x, y + bottom + pad_y)
        draw.rounded_rectangle(box, radius=max(4, font_size // 5), fill=(14, 25, 43, int(alpha * 0.72)))
    if style == "outline":
        stroke = max(1, font_size // 14)
        draw.text((x, y), text, font=font, fill=(*rgb, alpha), stroke_width=stroke, stroke_fill=(8, 15, 26, int(alpha * 0.9)))
    else:
        draw.text((x, y), text, font=font, fill=(*rgb, alpha))
    return Image.alpha_composite(base, overlay).convert("RGB")


def add_image_watermark(
    image: Image.Image,
    watermark_path: Path,
    position: str = "bottom-right",
    opacity: float = 0.55,
    size_ratio: float = 0.05,
) -> Image.Image:
    if position not in POSITIONS:
        position = "bottom-right"
    base = image.convert("RGBA")
    with Image.open(watermark_path) as opened:
        mark = opened.convert("RGBA")
    longest = max(1, int(min(base.size) * max(0.01, min(size_ratio, 0.2))))
    scale = min(1.0, longest / max(mark.size))
    mark = mark.resize((max(1, round(mark.width * scale)), max(1, round(mark.height * scale))), Image.Resampling.LANCZOS)
    alpha = mark.getchannel("A").point(lambda value: round(value * max(0.0, min(1.0, opacity))))
    mark.putalpha(alpha)
    margin = max(8, int(min(base.size) * 0.03))
    x, y = _watermark_position(position, base.size, mark.size, margin)
    base.alpha_composite(mark, (x, y))
    return base.convert("RGB")


def apply_watermark(image: Image.Image, options: dict[str, Any], watermark_path: Path | str | None = None) -> Image.Image:
    if not options.get("watermark_enabled"):
        return image.convert("RGB")
    position = options.get("watermark_position") or "bottom-right"
    opacity = float(options.get("watermark_opacity") if options.get("watermark_opacity") is not None else 0.55)
    size_ratio = float(options.get("watermark_size_ratio") if options.get("watermark_size_ratio") is not None else 0.05)
    if options.get("watermark_type") == "image":
        if watermark_path:
            return add_image_watermark(image, Path(watermark_path), position, opacity, size_ratio)
        return image.convert("RGB")
    text = (options.get("watermark_text") or "").strip()
    if not text:
        return image.convert("RGB")
    return add_watermark(
        image,
        text,
        position=position,
        opacity=opacity,
        size_ratio=size_ratio,
        color=options.get("watermark_color") or "#FFFFFF",
        font_family=options.get("watermark_font") or "sans",
        style=options.get("watermark_style") or "outline",
    )


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
    """Crop, downsample, watermark, name, and save one image with optional metadata."""
    cropped = crop_region(
        image_bgr,
        quad,
        inset_px=inset_px,
        rotate=int(options.get("rotate") or 0),
        flip_h=bool(options.get("flip_h")),
        flip_v=bool(options.get("flip_v")),
    )
    image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

    max_edge = options.get("max_long_edge")
    if max_edge:
        image = downsample(image, int(max_edge))

    image = apply_watermark(image, options, options.get("watermark_path"))

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
                    "flip_h": bool(options.get("flip_h")),
                    "flip_v": bool(options.get("flip_v")),
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
