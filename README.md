# Auto Cropping of Image Scans (Project 17)

A local tool that takes a scanned or photographed page containing photos or postcards, automatically finds the photo region, performs **rectangular cropping + perspective correction**, and exports in batch. When auto-detection is inaccurate, you can drag the four corners to adjust manually.

Stack: `HTML + CSS + vanilla JS` frontend / `FastAPI` API layer / `OpenCV + NumPy + Pillow` image processing.

> Note: this project performs **rectangular region cropping**, not cutout/background removal. Output keeps the rectangular boundary; no transparency matting is applied.

---

## 1. Feature list

| Feature | Required? | Implementation |
| --- | --- | --- |
| Multi-image import (drag & drop supported) | Required | `app/static/app.js` `uploadFiles()` / `app/main.py` `POST /api/upload` |
| Auto-detect target region | Required | `app/detector.py` `detect()` |
| Auto rectangular crop + perspective correction | Required | `app/detector.py` `crop_region()` |
| Clear failure hint on detection failure | Required | `detect()` returns `reason`; shown in the front-end status bar |
| Manual fallback (drag corners / pan / rotate 90° / switch candidates) | Bonus | Canvas drag logic in `app/static/app.js` |
| Result comparison & browsing | Bonus | `app/static/app.js` `renderResults()` |
| Batch export | Bonus | `app/main.py` `POST /api/export` |
| Downsampling (optional long-edge cap, shrink only) | Project brief | `app/exporter.py` `downsample()` |
| Watermark (text, 9-grid position, opacity, font size ratio) | Project brief | `app/exporter.py` `add_watermark()` |
| Custom label + filename template | Project brief | `app/exporter.py` `build_stem()` |
| Sidecar JSON export (source + corner coordinates) | Bonus | `app/exporter.py` `export_item()` |
| Quantitative detection evaluation (IoU) | Bonus | `tools/eval.py` + `sample_data/` |
| End-to-end smoke test | Bonus | `tools/smoke_test.py` |

---

## 2. Directory structure

```
project-root/
├── config.json              # all tunable parameters (detection thresholds, upload limits, export defaults)
├── requirements.txt
├── start.bat                # one-click launch (install deps + start server + open browser)
├── README.md
├── app/
│   ├── config.py            # load config.json (falls back to defaults for missing keys)
│   ├── storage.py           # upload registry persistence (data/manifest.json)
│   ├── detector.py          # detection + perspective crop (core algorithm)
│   ├── exporter.py          # downsample / watermark / naming / write to disk
│   ├── main.py              # FastAPI routes and static file mounting
│   └── static/              # index.html / style.css / app.js
├── tools/
│   ├── make_demo_data.py    # generate synthetic scan samples + ground truth
│   ├── eval.py              # evaluate detection accuracy via IoU, writes eval_report.csv
│   └── smoke_test.py        # HTTP-layer end-to-end self-test
├── sample_data/
│   ├── images/              # 6 synthetic samples
│   └── labels/              # 5 ground-truth files (scan_03 is a two-region sample, no label)
└── data/                    # generated at runtime: uploads / previews / crops / exports / manifest.json
```

---

## 3. Quick start

### Option 1: double-click `start.bat`

It will: switch to UTF-8 code page -> install dependencies -> open `http://127.0.0.1:8000` after a 3-second delay -> start the server.

### Option 2: manually

```bash
cd project-root
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000> in your browser.

Requirements: Python 3.10+ (developed and tested on 3.13); works on Windows / macOS / Linux. The first `opencv-python` install is a few dozen MB.

---

## 4. Workflow

1. Click **Import Images** or drag files into the window (supports jpg/jpeg/png/webp/bmp; max 60MB per file, max 30 files per import).
2. Each image is auto-detected; the left list shows status: green = detected, orange = low confidence / not found, grey = discarded.
3. The middle canvas shows the detected quadrilateral (cyan box + four corner handles):
   - Drag a corner to fine-tune; drag inside the box to pan the whole region;
   - For multi-region images, switch candidates in the chip bar at the top;
   - Click **Rotate 90°** to fix orientation.
4. Click **Accept** to produce the crop; the result appears in the bottom-right "crop" preview and the result grid.
5. Set export options at the bottom: format (jpg/png), quality, long-edge downsample, watermark text/position/opacity, label, filename template, output directory.
6. Click **Batch Export** to write all accepted files (defaults to `data/exports/<timestamp>/`).

The filename template supports four placeholders: `{name}` original filename, `{index}` sequence number, `{label}` custom label, `{date}` export date (e.g. `20261102`).

---

## 5. Algorithm

Detection runs on a grayscale image scaled to long side <= `detect_max_side` (default 1600). Coordinates are scaled back to the original image, and cropping always happens at **original resolution**.

Three candidate strategies run in parallel; results are merged and deduplicated by bounding-box overlap (IoU > 0.6 keeps the higher-confidence one):

| Strategy | Approach | Best for |
| --- | --- | --- |
| A Background diff | Estimate background from the median color of a 5% border band, compute per-pixel max channel difference, threshold at `max(8, median + 2.5 * 1.4826 * MAD)` (adaptive noise), then morphology (close/open) and outer contours | Card differs from background in color |
| B Canny edges | Adaptive thresholds from the grayscale median, dilate + close edges into loops, find outer contours | Low contrast such as white card on white |
| C Gradient fallback | When A and B return nothing, take Sobel magnitude > `max(18, p90)`, heavy close into loops | Background estimation completely fails |

Contour -> quadrilateral rules:

- Area ratio must lie in `[min_area_ratio, max_area_ratio]` = `[1.5%, 92%]`;
- Convex-hull solidity (contour area / hull area) >= `min_solidity` = 0.85, to reject tree-like or hollow contours;
- `cv2.approxPolyDP` with epsilon increasing from 0.015 to 0.16 to find a convex 4-point quad; falls back to `minAreaRect` if none found;
- Confidence = `0.45 * fill + 0.55 * edge_support` (sample Sobel magnitude along the four edges, normalized by p99);
- Candidates below `min_confidence` = 0.45 are still returned but flagged `low_confidence` so the UI prompts the user.

Cropping: `order_quad` arranges corners as TL->TR->BR->BL via sum/diff; `getPerspectiveTransform` + `warpPerspective` output an upright rectangle; output width/height take the larger of opposite-side lengths; an extra inward inset of `inset_px` (default 2px) avoids background bleed.

Images are read with Pillow (`ImageOps.exif_transpose` corrects phone orientation) and written with `cv2.imencode` + `tofile`, avoiding OpenCV's inability to read non-ASCII paths on Windows.

---

## 6. config.json parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `detector.detect_max_side` | 1600 | Long-side cap during detection; larger = more accurate but slower |
| `detector.border_ratio` | 0.05 | Border band width (fraction of short side) used to estimate background color |
| `detector.min_area_ratio` | 0.015 | Minimum candidate area ratio |
| `detector.max_area_ratio` | 0.92 | Maximum candidate area ratio (avoids treating the whole image as target) |
| `detector.min_solidity` | 0.85 | Minimum solidity, filters irregular contours |
| `detector.close_kernel` | 9 | Close-operation kernel size for strategy A |
| `detector.max_regions` | 4 | Maximum number of candidate regions per image |
| `detector.min_confidence` | 0.45 | Below this -> "low confidence" |
| `detector.inset_px` | 2 | Pixels to shrink the quad inward on crop |
| `upload.max_files` | 30 | Max files per upload |
| `upload.max_file_mb` | 60 | Max size per file |
| `upload.preview_max_side` | 1600 | Long side of front-end preview images |
| `upload.allowed_ext` | jpg/jpeg/png/webp/bmp | Allowed file extensions |
| `export.default_format` | jpg | Default export format |
| `export.default_quality` | 92 | Default JPEG quality |
| `export.write_sidecar` | true | Whether to write a sidecar .json by default |

Restart the server after editing `config.json` for changes to take effect.

---

## 7. API endpoints

| Method + path | Request | Response |
| --- | --- | --- |
| `GET /api/health` | — | `{ok, time}` |
| `GET /api/config` | — | Upload limits and export defaults (for front-end init) |
| `POST /api/upload` | multipart, field `files` (multiple allowed) | `{items: [{id, name, url, width, height}]}`, failed items include `error` |
| `POST /api/detect` | `{image_id}` | `{status: ok/failed, reason: null/no_region/low_confidence, regions: [{quad, confidence, area_ratio}], width, height, detect_ms}` |
| `POST /api/crop` | `{image_id, quad: [[x,y]*4], rotate: 0/90/180/270}` | `{url, width, height}` |
| `POST /api/export` | `{items: [{image_id, quad, rotate}], options: {...}}` | `{output_dir, files: [{name, path, width, height}], errors, count}` |

`quad` uses **original-image pixel coordinates** in TL->TR->BR->BL order (the server reorders and clips). See `ExportOptions` in `app/main.py` for the `options` fields.

Static files: `/media/*` serves `data/` (previews, crops), `/` serves the web UI.

---

## 8. Evaluation method and results

`tools/eval.py`:

- Ground-truth format: a `.json` with the same name as the image, content `{"quad": [[x,y]*4]}` in original-image pixel coordinates;
- Prediction = the candidate with the highest confidence / largest area;
- Both prediction and truth polygons are rasterized onto a mask with long side <= 1200, and **polygon IoU** is computed via `cv2.fillPoly`;
- Decision: IoU >= 0.85 = success, 0.70-0.85 = partial, else fail.

Results on `sample_data/` (**6 synthetic scans, 5 with ground truth**):

| Image | IoU | Status |
| --- | --- | --- |
| scan_01_white_card | 0.9954 | ok |
| scan_02_postcard_dark | 0.9775 | ok |
| scan_04_white_on_white | 0.9834 | ok |
| scan_05_perspective | 0.9979 | ok |
| scan_06_small_print | 0.9854 | ok |
| **Summary** | **mean 0.9879** | **ok=5, partial=0, fail=0** |

`scan_03_two_photos` is a two-region sample (detection correctly returns 2 regions); it has no ground truth so it is excluded from the IoU statistics.

Reproduce:

```bash
python tools/make_demo_data.py    # regenerate synthetic samples and labels
python tools/eval.py              # print the table and write eval_report.csv
python tools/smoke_test.py        # end-to-end HTTP self-test, should be 11/11 PASS
```

> **Important disclaimer**: the numbers above come from **synthetic samples** whose background/card are generated procedurally with gradients and noise; they do not represent the difficulty of real scans. To draw trustworthy conclusions in your report, build a real labelled set in the same format: put real scans in `sample_data/images/` (or a separate directory), use any labelling tool (LabelMe, CVAT, or this tool itself by dragging corners and saving the quad as json) to produce same-name `.json` files, then point `tools/eval.py` at that directory. Real data is expected to surface failures such as white-on-white cards, overlapping cards, or non-rectangular targets.

---

## 9. Known limitations

- **Not supported**: TIFF / HEIC / RAW; 60MB per-file cap.
- White-on-white cards, very low contrast, or strong reflections may cause missed or shifted detections; manual corner dragging is the fallback.
- Overlapping cards may yield only the outermost contour.
- Non-rectangular targets (discs, irregular photos) fall back to the minimum-area rotated rectangle.
- Cards with a thick white border may have the boundary land inside or outside the border depending on contrast; expand manually before accepting.
- No content-level restoration (descreen, dust removal, auto color correction).
- This is a **single-machine local tool**: no user system, no concurrency control, `manifest.json` is a simple JSON store, not suitable for multi-user concurrent use.
- Output directory must be an absolute path (safety check to prevent accidental overwrites).

---

## 10. Possible extensions

1. OCR / content recognition to auto-classify card type and tag.
2. Train a lightweight segmentation model (e.g. U-Net / MobileNet segmentation head) on real labels to replace the geometry-only detector.
3. Batch task queue + progress bar + resume (currently synchronous, one image at a time).
4. Drag-and-drop whole folders, preserve subdirectory structure in output.
5. Export PDF imposition / contact sheet.
6. Package as a single-file desktop app (PyInstaller + local port auto-start).

---

## 11. Handover notes

- All tunable parameters live in `config.json`; adapting to a different scanner background does not require code changes.
- Algorithm entry point is `app/detector.py::detect`, taking a BGR array + config dict and returning `(regions, reason)`, decoupled from the web layer so it can be called or replaced standalone.
- API contract in section 7; the front-end talks to the back-end through exactly these four endpoints, so front-end changes do not affect the algorithm.
- All runtime data lives in `data/`; deleting it clears all upload and export records (files written to a manually specified output directory under `exports/` are not deleted).
- Verification commands: `tools/eval.py` (accuracy) and `tools/smoke_test.py` (pipeline). Re-run both after changing the algorithm.
