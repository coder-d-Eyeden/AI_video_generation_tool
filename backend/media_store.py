"""
File-backed registry for generated/uploaded clips, audio assets and final renders.
Each media file has a JSON sidecar so the library survives server restarts.
"""
import json
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from backend import config
from backend.video_editor import extract_thumbnail, probe

_lock = threading.Lock()

_KIND_DIRS = {"clip": "CLIPS_DIR", "asset": "ASSETS_DIR", "final": "FINAL_DIR"}


def _dir(kind: str) -> str:
    return getattr(config, _KIND_DIRS[kind])


def _url(path: str) -> str:
    rel = os.path.relpath(path, config.MEDIA_ROOT).replace(os.sep, "/")
    return f"/media/{rel}"


def new_media_path(kind: str, ext: str) -> str:
    config.ensure_media_dirs()
    return os.path.join(_dir(kind), f"{uuid.uuid4().hex[:12]}{ext}")


def register(kind: str, path: str, **meta: Any) -> Dict[str, Any]:
    """Registers a media file already written at `path` and returns its record."""
    media_id = os.path.splitext(os.path.basename(path))[0]
    record: Dict[str, Any] = {
        "id": media_id,
        "kind": kind,
        "filename": os.path.basename(path),
        "url": _url(path),
        "created_at": time.time(),
    }
    try:
        record.update(probe(path))
    except Exception:
        pass
    if kind in ("clip", "final") and record.get("has_video"):
        thumb = os.path.splitext(path)[0] + ".jpg"
        try:
            extract_thumbnail(path, thumb)
            record["thumbnail_url"] = _url(thumb)
        except Exception:
            pass
    record.update({k: v for k, v in meta.items() if v is not None})
    with _lock:
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    return record


def save_bytes(kind: str, data: bytes, ext: str, **meta: Any) -> Dict[str, Any]:
    path = new_media_path(kind, ext)
    with open(path, "wb") as f:
        f.write(data)
    return register(kind, path, **meta)


def get(kind: str, media_id: str) -> Optional[Dict[str, Any]]:
    sidecar = os.path.join(_dir(kind), f"{os.path.basename(media_id)}.json")
    if not os.path.exists(sidecar):
        return None
    with open(sidecar, encoding="utf-8") as f:
        record = json.load(f)
    record["path"] = os.path.join(_dir(kind), record["filename"])
    return record


def list_media(kind: str) -> List[Dict[str, Any]]:
    config.ensure_media_dirs()
    out = []
    for name in os.listdir(_dir(kind)):
        if name.endswith(".json"):
            rec = get(kind, name[:-5])
            if rec and os.path.exists(rec["path"]):
                rec.pop("path", None)
                out.append(rec)
    return sorted(out, key=lambda r: r.get("created_at", 0))


def delete(kind: str, media_id: str) -> bool:
    rec = get(kind, media_id)
    if not rec:
        return False
    base = os.path.splitext(rec["path"])[0]
    for p in (rec["path"], base + ".json", base + ".jpg"):
        if os.path.exists(p):
            os.remove(p)
    return True


def public(record: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in record.items() if k != "path"}
