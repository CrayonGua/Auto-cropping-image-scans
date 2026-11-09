"""FastAPI app: web UI + detection endpoints + export endpoints.

Start from the project root:
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
or double-click start.bat.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from .config import DATA_DIR, ROOT, load_config
from .detector import crop_region, detect, fit_long_side, load_bgr, save_png
from .exporter import export_item
from .storage import Store

STATIC_DIR = ROOT / "app" / "static"
for _sub in ("uploads", "previews", "crops", "exports"):
    (DATA_DIR / _sub).mkdir(parents=True, exist_ok=True)

CONFIG = load_config()
STORE = Store(DATA_DIR)

app = FastAPI(
    title="Auto Cropping of Image Scans",
    description="Detect photos/postcards in scans and auto-crop, correct perspective, and export in batch.",
    version="0.1.0",
)


# ---------------------------------------------------------------- request models

class DetectIn(BaseModel):
    image_id: str


class CropIn(BaseModel):
    image_id: str
    quad: list[list[float]]
    rotate: int = 0


class ExportItem(BaseModel):
    image_id: str
    quad: list[list[float]]
    rotate: int = 0


class ExportOptions(BaseModel):
    fmt: Literal["jpg", "png"] = "jpg"
    quality: int = Field(default=92, ge=40, le=100)
    max_long_edge: int | None = Field(default=None, ge=200, le=12000)
    watermark_text: str = ""
    watermark_position: str = "bottom-right"
    watermark_opacity: float = Field(default=0.35, ge=0.0, le=1.0)
    watermark_size_ratio: float = Field(default=0.04, ge=0.01, le=0.2)
    label: str = ""
    filename_template: str = "{name}_{index}"
    output_dir: str | None = None
    write_sidecar: bool = True


class ExportIn(BaseModel):
    items: list[ExportItem]
    options: ExportOptions


# ---------------------------------------------------------------- helpers

def _record(image_id: str) -> dict:
    record = STORE.get(image_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Image not found: id={image_id}")
    return record


def _load_record_image(record: dict) -> np.ndarray:
    path = STORE.file_path(record)
    if not path.exists():
        raise HTTPException(status_code=410, detail="Image file is missing; please re-upload")
    try:
        return load_bgr(path)
    except Exception as exc:  # corrupt image
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}") from exc


def _validated_quad(quad: list[list[float]], width: int, height: int) -> np.ndarray:
    points = np.asarray(quad, dtype=np.float32)
    if points.shape != (4, 2):
        raise HTTPException(status_code=400, detail="quad must contain exactly 4 [x, y] points")
    if not np.isfinite(points).all():
        raise HTTPException(status_code=400, detail="quad contains non-finite values")
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    xs, ys = points[:, 0], points[:, 1]
    area = abs(float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))) / 2.0
    if area < 100:
        raise HTTPException(status_code=400, detail="quad area too small; please redraw the box")
    return points


async def _read_limited(upload: UploadFile, limit_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"{upload.filename} exceeds the {limit_bytes // (1024 * 1024)}MB limit",
            )
        chunks.append(chunk)
    return b"".join(chunks)


# ---------------------------------------------------------------- endpoints

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
        raise HTTPException(
            status_code=400, detail=f"Too many files; max is {upload_cfg['max_files']} per upload"
        )
    size_limit = int(upload_cfg["max_file_mb"]) * 1024 * 1024
    preview_side = int(upload_cfg["preview_max_side"])
    allowed = {ext.lower() for ext in upload_cfg["allowed_ext"]}

    results: list[dict] = []
    for upload_file in files:
        name = Path(upload_file.filename or "").name
        suffix = Path(name).suffix.lower()
        if suffix not in allowed:
            results.append({"name": name, "error": f"Unsupported format: {suffix or 'unknown'}"})
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
            with Image.open(target) as im:
                im = ImageOps.exif_transpose(im)
                width, height = im.size
                preview = im.convert("RGB")
            bgr = load_bgr(target)
            preview_bgr = fit_long_side(bgr, preview_side)
            save_png(DATA_DIR / "previews" / f"{image_id}.png", preview_bgr)
        except Exception as exc:
            target.unlink(missing_ok=True)
            results.append({"name": name, "error": f"Cannot decode image: {exc}"})
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
    )
    name = f"{payload.image_id}_{uuid.uuid4().hex[:6]}.png"
    save_png(DATA_DIR / "crops" / name, output)
    return {
        "url": f"/media/crops/{name}",
        "width": int(output.shape[1]),
        "height": int(output.shape[0]),
    }


@app.post("/api/export")
def api_export(payload: ExportIn) -> dict:
    if not payload.items:
        raise HTTPException(status_code=400, detail="No images to export")

    options = payload.options.model_dump()
    raw_dir = (options.get("output_dir") or "").strip()
    if raw_dir:
        out_dir = Path(raw_dir).expanduser()
        if not out_dir.is_absolute():
            raise HTTPException(
                status_code=400, detail=r"Output directory must be an absolute path, e.g. D:\photo_archive"
            )
    else:
        out_dir = DATA_DIR / "exports" / datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Cannot create output directory: {exc}") from exc

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
        except Exception as exc:  # a single failure must not block the rest
            errors.append({"image_id": item.image_id, "error": str(exc)})

    return {
        "output_dir": str(out_dir),
        "files": files,
        "errors": errors,
        "count": len(files),
    }


# Static: /media serves the data folder (previews, crops), / serves the web UI
app.mount("/media", StaticFiles(directory=str(DATA_DIR)), name="media")
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
