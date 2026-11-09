"""Thread-safe manifest storage for uploaded image metadata."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class Store:
    """Keep upload metadata between local application restarts."""

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
            loaded = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._items = {}
            return
        if isinstance(loaded, dict):
            self._items = {
                key: value
                for key, value in loaded.items()
                if isinstance(key, str) and isinstance(value, dict)
            }

    def _save(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self._items, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.manifest_path)

    def add(self, record: dict) -> None:
        with self._lock:
            self._items[record["id"]] = record
            self._save()

    def get(self, image_id: str) -> dict | None:
        with self._lock:
            record = self._items.get(image_id)
            return dict(record) if record is not None else None

    def delete(self, image_id: str) -> dict | None:
        with self._lock:
            record = self._items.pop(image_id, None)
            if record is not None:
                self._save()
                return dict(record)
            return None

    def all(self) -> list[dict]:
        with self._lock:
            return [dict(record) for record in self._items.values()]

    def touch(self, image_id: str) -> None:
        with self._lock:
            record = self._items.get(image_id)
            if record is None:
                return
            record["last_used_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._save()

    def file_path(self, record: dict) -> Path:
        """Build an upload path from its stored identifier and extension."""
        return self.data_dir / "uploads" / f"{record['id']}{record['ext']}"

    @staticmethod
    def preview_url(image_id: str) -> str:
        return f"/media/previews/{image_id}.png"
