"""Uploaded-file registry.

Images are stored under data/uploads keyed by uuid, with metadata
persisted to data/manifest.json so previously uploaded files survive
a backend restart.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class Store:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.manifest_path = self.data_dir / "manifest.json"
        self._lock = threading.Lock()
        self._items: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.manifest_path.exists():
            return
        try:
            self._items = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._items = {}

    def _save(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(self._items, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add(self, record: dict) -> None:
        with self._lock:
            self._items[record["id"]] = record
            self._save()

    def get(self, image_id: str) -> dict | None:
        return self._items.get(image_id)

    def all(self) -> list[dict]:
        return list(self._items.values())

    def touch(self, image_id: str) -> None:
        record = self._items.get(image_id)
        if record is None:
            return
        record["last_used_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            self._save()

    # Paths are always built from id + controlled extension, never from
    # user-supplied paths.
    def file_path(self, record: dict) -> Path:
        return self.data_dir / "uploads" / f"{record['id']}{record['ext']}"

    @staticmethod
    def preview_url(image_id: str) -> str:
        return f"/media/previews/{image_id}.png"
