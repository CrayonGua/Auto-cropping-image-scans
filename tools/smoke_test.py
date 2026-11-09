"""Exercise the local HTTP workflow and remove its temporary working files."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((label, bool(condition), detail))
    print(f"[{'PASS' if condition else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))


def summarize() -> int:
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    print("-" * 60)
    print(f"Passed {passed}/{len(CHECKS)} checks")
    return 0 if passed == len(CHECKS) else 1


def main() -> int:
    client = TestClient(app)
    image_id: str | None = None

    health = client.get("/api/health")
    check("GET /api/health", health.status_code == 200)

    sample_dir = ROOT / "sample_data" / "images"
    samples = sorted(sample_dir.glob("*.jpg"))
    if not samples:
        print("No sample images found. Run tools/make_demo_data.py first.")
        return 1

    try:
        with samples[0].open("rb") as handle:
            upload = client.post(
                "/api/upload",
                files=[("files", (samples[0].name, handle, "image/jpeg"))],
            )
        check("POST /api/upload", upload.status_code == 200, f"HTTP {upload.status_code}")
        payload = upload.json()
        item = (payload.get("items") or [{}])[0]
        image_id = item.get("id")
        check("Upload returns an ID and preview URL", bool(image_id) and bool(item.get("url")), item.get("name", ""))

        if image_id:
            detect_response = client.post("/api/detect", json={"image_id": image_id})
            check("POST /api/detect", detect_response.status_code == 200, f"HTTP {detect_response.status_code}")
            result = detect_response.json() if detect_response.status_code == 200 else {}
            check(
                "Detection returns a candidate or an explicit reason",
                bool(result.get("regions")) or bool(result.get("reason")),
                f"candidates={len(result.get('regions') or [])}, reason={result.get('reason')}",
            )

            if result.get("regions"):
                quad = result["regions"][0]["quad"]
            else:
                width, height = result.get("width", 100), result.get("height", 100)
                quad = [[width * 0.15, height * 0.15], [width * 0.85, height * 0.15],
                        [width * 0.85, height * 0.85], [width * 0.15, height * 0.85]]

            crop = client.post("/api/crop", json={"image_id": image_id, "quad": quad, "rotate": 0})
            check("POST /api/crop", crop.status_code == 200, f"HTTP {crop.status_code}")
            crop_body = crop.json() if crop.status_code == 200 else {}
            check(
                "Crop returns a preview URL and positive dimensions",
                bool(crop_body.get("url")) and crop_body.get("width", 0) > 0 and crop_body.get("height", 0) > 0,
                f"{crop_body.get('width')}x{crop_body.get('height')}",
            )

            output_dir = tempfile.TemporaryDirectory(prefix="auto_crop_smoke_")
            try:
                export = client.post(
                    "/api/export",
                    json={
                        "items": [{"image_id": image_id, "quad": quad, "rotate": 0}],
                        "options": {
                            "fmt": "jpg", "quality": 88, "max_long_edge": 800,
                            "watermark_text": "SMOKE TEST", "watermark_position": "bottom-right",
                            "watermark_opacity": 0.35, "label": "smoke",
                            "filename_template": "{name}_{index}_{label}",
                            "output_dir": output_dir.name, "write_sidecar": True,
                        },
                    },
                )
                check("POST /api/export", export.status_code == 200, f"HTTP {export.status_code}")
                export_body = export.json() if export.status_code == 200 else {}
                files = export_body.get("files") or []
                check("Export creates one image", len(files) == 1 and Path(files[0]["path"]).exists(),
                      files[0]["name"] if files else "")
                if files:
                    check("Downsampling respects the 800-pixel limit",
                          max(files[0]["width"], files[0]["height"]) <= 800,
                          f"{files[0]['width']}x{files[0]['height']}")
                    check("Sidecar metadata is written",
                          Path(files[0]["path"]).with_suffix(".json").exists())
            finally:
                output_dir.cleanup()

            check("Uploaded original is not web-accessible",
                  client.get(f"/media/uploads/{image_id}{Path(item['name']).suffix}").status_code == 404)

    finally:
        if image_id:
            cleanup = client.post("/api/clear", json={"image_ids": [image_id]})
            cleanup_body = cleanup.json() if cleanup.status_code == 200 else {}
            suffix = Path(item["name"]).suffix
            leftovers = [
                ROOT / "data" / "uploads" / f"{image_id}{suffix}",
                ROOT / "data" / "previews" / f"{image_id}.png",
                *(ROOT / "data" / "crops").glob(f"{image_id}_*.png"),
            ]
            remaining = [path.name for path in leftovers if path.exists()]
            check(
                "POST /api/clear removes temporary files",
                cleanup.status_code == 200
                and cleanup_body.get("deleted") == 1
                and not cleanup_body.get("errors")
                and not remaining,
                ", ".join(remaining),
            )

    return summarize()


if __name__ == "__main__":
    sys.exit(main())
