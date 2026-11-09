"""FastAPI application for the local crop editor, image processing, and export API."""

from __future__ import annotations

import io
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from .config import DATA_DIR, ROOT, load_config
from .detector import (
    crop_region, detect, fit_long_side, load_bgr, order_quad, polygon_area, save_png,
)
from .exporter import apply_watermark, export_item
from .storage import Store

STATIC_DIR = ROOT / "app" / "static"
for _sub in ("uploads", "previews", "crops", "exports", "watermarks"):
    (DATA_DIR / _sub).mkdir(parents=True, exist_ok=True)

CONFIG = load_config()
STORE = Store(DATA_DIR)

app = FastAPI(
    title="Auto Crop for Scans",
    description="Detect, review, perspective-correct, and export image regions locally.",
    version="0.2.0",
)


class DetectIn(BaseModel):
    image_id: str


class CropIn(BaseModel):
    image_id: str
    quad: list[list[float]]
    rotate: Literal[0, 90, 180, 270] = 0
    flip_h: bool = False
    flip_v: bool = False


class PreviewIn(CropIn):
    watermark_enabled: bool = False
    watermark_type: Literal["text", "image"] = "text"
    watermark_text: str = Field(default="", max_length=120)
    watermark_image_id: str | None = None
    watermark_position: str = "bottom-right"
    watermark_opacity: float = Field(default=0.55, ge=0.05, le=1.0)
    watermark_size_ratio: float = Field(default=0.05, ge=0.01, le=0.2)
    watermark_color: str = Field(default="#FFFFFF", pattern=r"^#[0-9A-Fa-f]{6}$")
    watermark_font: Literal["sans", "serif", "mono"] = "sans"
    watermark_style: Literal["plain", "outline", "badge"] = "outline"


class ExportItem(BaseModel):
    image_id: str
    quad: list[list[float]]
    rotate: Literal[0, 90, 180, 270] = 0
    flip_h: bool = False
    flip_v: bool = False


class ExportOptions(BaseModel):
    fmt: Literal["jpg", "png"] = "jpg"
    quality: int = Field(default=92, ge=40, le=100)
    max_long_edge: int | None = Field(default=None, ge=200, le=12000)
    watermark_enabled: bool = False
    watermark_type: Literal["text", "image"] = "text"
    watermark_text: str = Field(default="", max_length=120)
    watermark_image_id: str | None = None
    watermark_position: str = "bottom-right"
    watermark_opacity: float = Field(default=0.55, ge=0.05, le=1.0)
    watermark_size_ratio: float = Field(default=0.05, ge=0.01, le=0.2)
    watermark_color: str = Field(default="#FFFFFF", pattern=r"^#[0-9A-Fa-f]{6}$")
    watermark_font: Literal["sans", "serif", "mono"] = "sans"
    watermark_style: Literal["plain", "outline", "badge"] = "outline"
    label: str = Field(default="", max_length=60)
    filename_template: str = Field(default="{name}_{index}", max_length=120)
    output_dir: str | None = None
    write_sidecar: bool = False


class ExportIn(BaseModel):
    items: list[ExportItem]
    options: ExportOptions


class ClearIn(BaseModel):
    image_ids: list[str]
    watermark_ids: list[str] = Field(default_factory=list)


class RemoveWatermarkIn(BaseModel):
    watermark_id: str


def _record(image_id: str) -> dict:
    record = STORE.get(image_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Image not found: id={image_id}")
    return record


def _load_record_image(record: dict) -> np.ndarray:
    path = STORE.file_path(record)
    if not path.exists():
        raise HTTPException(status_code=410, detail="Image file is no longer available. Please import it again.")
    try:
        return load_bgr(path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image could not be decoded: {exc}") from exc


def _validated_quad(quad: list[list[float]], width: int, height: int) -> np.ndarray:
    try:
        points = np.asarray(quad, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Crop coordinates must be numeric.") from exc
    if points.shape != (4, 2):
        raise HTTPException(status_code=400, detail="quad must contain exactly four [x, y] points.")
    if not np.isfinite(points).all():
        raise HTTPException(status_code=400, detail="Crop coordinates contain an invalid number.")
    if len(np.unique(points, axis=0)) != 4:
        raise HTTPException(status_code=400, detail="Crop corners must be four distinct points.")

    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    ordered = order_quad(points)
    if len(np.unique(ordered, axis=0)) != 4:
        raise HTTPException(status_code=400, detail="Crop corners must be four distinct points inside the image.")

    edges = np.roll(ordered, -1, axis=0) - ordered
    crosses = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    if not (np.all(crosses > 1e-3) or np.all(crosses < -1e-3)):
        raise HTTPException(status_code=400, detail="Crop corners must form a convex quadrilateral.")
    if polygon_area(ordered) < 100:
        raise HTTPException(status_code=400, detail="Crop area is too small. Adjust the crop corners.")
    return ordered


async def _read_limited(upload: UploadFile, limit_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit_bytes:
            limit_mb = limit_bytes // (1024 * 1024)
            raise HTTPException(status_code=400, detail=f"{upload.filename} exceeds the {limit_mb} MB file limit.")
        chunks.append(chunk)
    return b"".join(chunks)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "time": datetime.now().isoformat(timespec="seconds")}


@app.get("/api/config")
def get_config() -> dict:
    return {
        "allowed_ext": CONFIG["upload"]["allowed_ext"],
        "max_files": CONFIG["upload"]["max_files"],
        "max_file_mb": CONFIG["upload"]["max_file_mb"],
        "defaults": CONFIG["export"],
    }


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict:
    upload_cfg = CONFIG["upload"]
    if len(files) > int(upload_cfg["max_files"]):
        raise HTTPException(status_code=400, detail=f"A single upload can contain at most {upload_cfg['max_files']} images.")
    size_limit = int(upload_cfg["max_file_mb"]) * 1024 * 1024
    preview_side = int(upload_cfg["preview_max_side"])
    allowed = {ext.lower() for ext in upload_cfg["allowed_ext"]}

    results: list[dict] = []
    for upload_file in files:
        name = Path(upload_file.filename or "").name
        suffix = Path(name).suffix.lower()
        try:
            if suffix not in allowed:
                results.append({"name": name, "error": f"Unsupported file format: {suffix or 'unknown'}"})
                continue

            try:
                raw = await _read_limited(upload_file, size_limit)
            except HTTPException as exc:
                results.append({"name": name, "error": exc.detail})
                continue

            image_id = uuid.uuid4().hex[:12]
            target = DATA_DIR / "uploads" / f"{image_id}{suffix}"
            target.write_bytes(raw)
            try:
                with Image.open(target) as opened:
                    corrected = ImageOps.exif_transpose(opened)
                    width, height = corrected.size
                bgr = load_bgr(target)
                preview_bgr = fit_long_side(bgr, preview_side)
                save_png(DATA_DIR / "previews" / f"{image_id}.png", preview_bgr)
            except Exception as exc:
                target.unlink(missing_ok=True)
                results.append({"name": name, "error": f"Image could not be decoded: {exc}"})
                continue

            record = {
                "id": image_id,
                "name": name,
                "ext": suffix,
                "width": int(width),
                "height": int(height),
                "uploaded_at": datetime.now().isoformat(timespec="seconds"),
            }
            STORE.add(record)
            results.append(
                {
                    "id": image_id,
                    "name": name,
                    "url": f"/media/previews/{image_id}.png",
                    "width": int(width),
                    "height": int(height),
                }
            )
        finally:
            await upload_file.close()
    return {"items": results}


@app.post("/api/detect")
def api_detect(payload: DetectIn) -> dict:
    record = _record(payload.image_id)
    image = _load_record_image(record)
    STORE.touch(payload.image_id)

    started = time.perf_counter()
    regions, reason = detect(image, CONFIG["detector"])
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "status": "failed" if reason else "ok",
        "reason": reason,
        "regions": [region.to_dict() for region in regions],
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "detect_ms": elapsed_ms,
    }


@app.post("/api/crop")
def api_crop(payload: CropIn) -> dict:
    record = _record(payload.image_id)
    image = _load_record_image(record)
    height, width = image.shape[:2]
    quad = _validated_quad(payload.quad, width, height)
    output = crop_region(
        image,
        quad,
        inset_px=float(CONFIG["detector"]["inset_px"]),
        rotate=payload.rotate,
        flip_h=payload.flip_h,
        flip_v=payload.flip_v,
    )
    name = f"{payload.image_id}_{uuid.uuid4().hex[:6]}.png"
    save_png(DATA_DIR / "crops" / name, output)
    return {
        "url": f"/media/crops/{name}",
        "width": int(output.shape[1]),
        "height": int(output.shape[0]),
    }


@app.post("/api/preview")
def api_preview(payload: PreviewIn) -> Response:
    record = _record(payload.image_id)
    image = _load_record_image(record)
    height, width = image.shape[:2]
    quad = _validated_quad(payload.quad, width, height)
    output = crop_region(
        image,
        quad,
        inset_px=float(CONFIG["detector"]["inset_px"]),
        rotate=payload.rotate,
        flip_h=payload.flip_h,
        flip_v=payload.flip_v,
    )
    preview = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    if payload.watermark_enabled:
        watermark_path = None
        if payload.watermark_type == "image" and payload.watermark_image_id:
            watermark_path = _watermark_path(payload.watermark_image_id, must_exist=True)
        preview = apply_watermark(preview, payload.model_dump(), watermark_path)
    stream = io.BytesIO()
    preview.save(stream, format="PNG", optimize=True)
    return Response(
        content=stream.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "no-store", "X-Preview-Size": f"{preview.width} × {preview.height}"},
    )


@app.post("/api/export")
def api_export(payload: ExportIn) -> dict:
    if not payload.items:
        raise HTTPException(status_code=400, detail="There are no accepted images to export.")

    options = payload.options.model_dump()
    raw_dir = (options.get("output_dir") or "").strip()
    if raw_dir:
        out_dir = Path(raw_dir).expanduser()
        if not out_dir.is_absolute():
            raise HTTPException(status_code=400, detail="Output folder must be an absolute path, for example D:\\photo_archive.")
    else:
        out_dir = DATA_DIR / "exports" / datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Could not create the output folder: {exc}") from exc

    inset_px = float(CONFIG["detector"]["inset_px"])
    files: list[dict] = []
    errors: list[dict] = []
    for index, item in enumerate(payload.items, start=1):
        try:
            record = _record(item.image_id)
            image = _load_record_image(record)
            height, width = image.shape[:2]
            quad = _validated_quad(item.quad, width, height)
            item_options = dict(options)
            item_options["rotate"] = item.rotate
            item_options["flip_h"] = item.flip_h
            item_options["flip_v"] = item.flip_v
            if item_options.get("watermark_enabled") and item_options.get("watermark_type") == "image":
                item_options["watermark_path"] = str(_watermark_path(item_options.get("watermark_image_id"), must_exist=True))
            files.append(
                export_item(
                    image,
                    quad,
                    item_options,
                    index=index,
                    source_name=record["name"],
                    out_dir=out_dir,
                    inset_px=inset_px,
                )
            )
        except HTTPException as exc:
            errors.append({"image_id": item.image_id, "error": exc.detail})
        except Exception as exc:
            errors.append({"image_id": item.image_id, "error": str(exc)})
    return {"output_dir": str(out_dir), "files": files, "errors": errors, "count": len(files)}


def _watermark_path(watermark_id: str | None, must_exist: bool = False) -> Path:
    if not watermark_id or not re.fullmatch(r"[0-9a-f]{12}", watermark_id):
        raise HTTPException(status_code=400, detail="Watermark image is missing or invalid. Please select it again.")
    path = DATA_DIR / "watermarks" / f"{watermark_id}.png"
    if must_exist and not path.is_file():
        raise HTTPException(status_code=404, detail="Watermark image is no longer available. Please select it again.")
    return path


@app.post("/api/browse-output")
async def browse_output() -> dict:
    root = None
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(title="Choose an output folder", mustexist=False)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"The native folder picker is unavailable: {exc}") from exc
    finally:
        if root is not None:
            root.destroy()
    return {"cancelled": not bool(selected), "path": selected or None}


@app.post("/api/watermark")
async def upload_watermark(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(status_code=400, detail="Choose a PNG, JPG, or WEBP image for the watermark.")
    raw = await _read_limited(file, 8 * 1024 * 1024)
    try:
        from PIL import Image
        from io import BytesIO
        with Image.open(BytesIO(raw)) as opened:
            opened.load()
            watermark = opened.convert("RGBA")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Watermark image could not be decoded: {exc}") from exc
    if max(watermark.size) > 4096:
        watermark.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
    watermark_id = uuid.uuid4().hex[:12]
    watermark.save(DATA_DIR / "watermarks" / f"{watermark_id}.png", format="PNG", optimize=True)
    return {"watermark_id": watermark_id, "name": Path(file.filename or "sticker.png").name}


@app.post("/api/watermark/remove")
def remove_watermark(payload: RemoveWatermarkIn) -> dict:
    _watermark_path(payload.watermark_id).unlink(missing_ok=True)
    return {"removed": True}


@app.post("/api/clear")
def clear_session(payload: ClearIn) -> dict:
    deleted = 0
    errors: list[str] = []
    for image_id in dict.fromkeys(payload.image_ids):
        record = STORE.get(image_id)
        if record is None:
            continue
        paths = [
            STORE.file_path(record),
            DATA_DIR / "previews" / f"{image_id}.png",
            *((DATA_DIR / "crops").glob(f"{image_id}_*.png")),
        ]
        error_count = len(errors)
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                errors.append(f"{path.name}: {exc}")
        if len(errors) == error_count:
            STORE.delete(image_id)
        deleted += 1
    for watermark_id in dict.fromkeys(payload.watermark_ids):
        try:
            _watermark_path(watermark_id).unlink(missing_ok=True)
        except (HTTPException, OSError) as exc:
            errors.append(f"watermark {watermark_id}: {getattr(exc, 'detail', str(exc))}")
    return {"deleted": deleted, "errors": errors}


app.mount(
    "/media/previews",
    StaticFiles(directory=str(DATA_DIR / "previews")),
    name="previews",
)
app.mount(
    "/media/crops",
    StaticFiles(directory=str(DATA_DIR / "crops")),
    name="crops",
)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
