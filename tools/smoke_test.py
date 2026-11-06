"""接口冒烟测试：上传 -> 检测 -> 裁剪 -> 导出，全程走 HTTP 接口。

不需要先启动服务。运行：
    python tools/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((label, bool(condition), detail))
    print(f"[{'PASS' if condition else 'FAIL'}] {label}" + (f" · {detail}" if detail else ""))


def main() -> int:
    client = TestClient(app)

    health = client.get("/api/health")
    check("GET /api/health", health.status_code == 200)

    sample_dir = ROOT / "sample_data" / "images"
    samples = sorted(
        path for path in sample_dir.glob("*.jpg")
    )
    if not samples:
        print("缺少示例图片，请先运行：python tools/make_demo_data.py")
        return 1

    with samples[0].open("rb") as handle:
        upload = client.post(
            "/api/upload",
            files=[("files", (samples[0].name, handle, "image/jpeg"))],
        )
    check("POST /api/upload", upload.status_code == 200, f"HTTP {upload.status_code}")
    payload = upload.json()
    item = (payload.get("items") or [{}])[0]
    check("上传返回 id 与预览地址", bool(item.get("id")) and bool(item.get("url")), item.get("name", ""))

    if not item.get("id"):
        return summarize()

    detect = client.post("/api/detect", json={"image_id": item["id"]})
    check("POST /api/detect", detect.status_code == 200, f"HTTP {detect.status_code}")
    result = detect.json() if detect.status_code == 200 else {}
    check("检测返回区域或明确的失败原因", bool(result.get("regions")) or bool(result.get("reason")),
          f"regions={len(result.get('regions') or [])}, reason={result.get('reason')}")

    if result.get("regions"):
        quad = result["regions"][0]["quad"]
    else:
        width, height = result.get("width", 100), result.get("height", 100)
        quad = [[width * 0.15, height * 0.15], [width * 0.85, height * 0.15],
                [width * 0.85, height * 0.85], [width * 0.15, height * 0.85]]

    crop = client.post("/api/crop", json={"image_id": item["id"], "quad": quad, "rotate": 0})
    check("POST /api/crop", crop.status_code == 200, f"HTTP {crop.status_code}")
    crop_body = crop.json() if crop.status_code == 200 else {}
    check("裁剪返回图片地址与尺寸", bool(crop_body.get("url")) and crop_body.get("width", 0) > 0,
          f"{crop_body.get('width')}×{crop_body.get('height')}")

    output_dir = (ROOT / "data" / "exports" / "smoke_test").resolve()
    export = client.post(
        "/api/export",
        json={
            "items": [{"image_id": item["id"], "quad": quad, "rotate": 0}],
            "options": {
                "fmt": "jpg",
                "quality": 88,
                "max_long_edge": 800,
                "watermark_text": "SMOKE TEST",
                "watermark_position": "bottom-right",
                "watermark_opacity": 0.35,
                "label": "smoke",
                "filename_template": "{name}_{index}_{label}",
                "output_dir": str(output_dir),
                "write_sidecar": True,
            },
        },
    )
    check("POST /api/export", export.status_code == 200, f"HTTP {export.status_code}")
    export_body = export.json() if export.status_code == 200 else {}
    files = export_body.get("files") or []
    check("导出落盘成功", len(files) == 1 and Path(files[0]["path"]).exists(),
          files[0]["name"] if files else "")
    if files:
        check("降采样生效（长边 <= 800）", max(files[0]["width"], files[0]["height"]) <= 800,
              f"{files[0]['width']}×{files[0]['height']}")
        check("sidecar 元数据已写出", Path(files[0]["path"]).with_suffix(".json").exists())

    return summarize()


def summarize() -> int:
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    print("-" * 60)
    print(f"通过 {passed}/{len(CHECKS)}")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
