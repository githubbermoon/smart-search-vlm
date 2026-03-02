from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .config import IMAGE_EXTENSIONS, StackConfig
from .db import (
    check_stale_files,
    connect_sqlite,
    ensure_schema,
    get_image_by_hash,
    get_image_by_id,
    get_image_by_path,
    list_enabled_watched_folders,
    mark_file_removed,
    mark_file_removed_by_path,
    update_image_file_location,
)
from .preprocess import preprocess_image


HashResolver = Callable[[Path], str]


class LivePathWatcher:
    """
    Low-RAM realtime watcher for image file path changes.
    It updates DB path metadata and stale flags but never runs model inference.
    """

    def __init__(
        self,
        cfg: StackConfig | None = None,
        *,
        debounce_ms: int = 1200,
        move_grace_sec: float = 5.0,
        hourly_refresh_min: int = 60,
        hash_resolver: HashResolver | None = None,
    ) -> None:
        self.cfg = cfg or StackConfig()
        self.debounce_sec = max(0.1, float(debounce_ms) / 1000.0)
        self.move_grace_sec = max(0.0, float(move_grace_sec))
        self.hourly_refresh_sec = max(60.0, float(hourly_refresh_min) * 60.0)
        self.hash_resolver = hash_resolver or self._default_hash_resolver
        self.stop_event = threading.Event()
        self.pending_created: dict[str, float] = {}
        self.pending_deleted: dict[str, float] = {}
        self.logger = logging.getLogger(__name__)
        self.stats: dict[str, int] = {
            "relinked": 0,
            "marked_stale": 0,
            "fallback_matched": 0,
            "ignored_new_files": 0,
            "hourly_refresh_runs": 0,
        }

    def _default_hash_resolver(self, path: Path) -> str:
        prepared = preprocess_image(path, self.cfg)
        return prepared.sha256_hash

    def _resolve_hash(self, path: Path) -> str:
        return self.hash_resolver(path)

    @staticmethod
    def _as_path_str(path: str | Path) -> str:
        return str(Path(path))

    @staticmethod
    def _stat_or_zero(path: Path) -> tuple[int, int, float]:
        try:
            st = os.stat(path)
            return int(st.st_ino), int(st.st_size), float(st.st_mtime)
        except OSError:
            return 0, 0, 0.0

    @staticmethod
    def _is_image_path(path: Path) -> bool:
        return path.suffix.lower() in IMAGE_EXTENSIONS

    def stop(self) -> None:
        self.stop_event.set()

    def enqueue_created(self, path: str | Path, *, now: float | None = None) -> None:
        ts = time.time() if now is None else float(now)
        self.pending_created[self._as_path_str(path)] = ts + self.debounce_sec

    def enqueue_deleted(self, path: str | Path, *, now: float | None = None) -> None:
        ts = time.time() if now is None else float(now)
        self.pending_deleted[self._as_path_str(path)] = ts + self.move_grace_sec

    def handle_created(self, path: str | Path, *, now: float | None = None) -> None:
        self.enqueue_created(path, now=now)

    def handle_deleted(self, path: str | Path, *, now: float | None = None) -> None:
        self.enqueue_deleted(path, now=now)

    def handle_moved(self, src_path: str | Path, dst_path: str | Path, *, now: float | None = None) -> None:
        src = self._as_path_str(src_path)
        dst = self._as_path_str(dst_path)
        if not src or not dst or src == dst:
            return

        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        try:
            row = get_image_by_path(conn, src)
            if row is not None:
                dst_obj = Path(dst)
                if dst_obj.exists() and dst_obj.is_file():
                    inode, size, mtime = self._stat_or_zero(dst_obj)
                    update_image_file_location(
                        conn,
                        image_id=str(row["id"]),
                        file_path=str(dst_obj),
                        file_inode=inode,
                        file_size=size,
                        file_mtime=mtime,
                    )
                    self.stats["relinked"] += 1
                    self.logger.info("relinked image_id=%s from=%s to=%s", str(row["id"]), src, str(dst_obj))
                    self.pending_deleted.pop(src, None)
                    self.pending_created.pop(dst, None)
                    return
        finally:
            conn.close()

        # Fall back for missed move pairing: treat as delete+create.
        self.enqueue_deleted(src, now=now)
        self.enqueue_created(dst, now=now)

    def process_pending(self, *, now: float | None = None) -> dict[str, int]:
        ts = time.time() if now is None else float(now)
        out = {"created_processed": 0, "deleted_processed": 0}

        due_created = [p for p, due in self.pending_created.items() if due <= ts]
        for p in due_created:
            self.pending_created.pop(p, None)
            self._process_created_path(Path(p))
            out["created_processed"] += 1

        due_deleted = [p for p, due in self.pending_deleted.items() if due <= ts]
        for p in due_deleted:
            self.pending_deleted.pop(p, None)
            self._process_deleted_path(Path(p))
            out["deleted_processed"] += 1

        return out

    def _process_created_path(self, path: Path) -> None:
        # Low-RAM mode: path-only reconciliation; ignore unknown files.
        if not path.exists() or (not path.is_file()):
            return
        if not self._is_image_path(path):
            self.stats["ignored_new_files"] += 1
            self.logger.info("ignored_new_files path=%s reason=unsupported_ext", str(path))
            return

        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        try:
            by_path = get_image_by_path(conn, str(path))
            if by_path is not None:
                inode, size, mtime = self._stat_or_zero(path)
                update_image_file_location(
                    conn,
                    image_id=str(by_path["id"]),
                    file_path=str(path),
                    file_inode=inode,
                    file_size=size,
                    file_mtime=mtime,
                )
                self.stats["relinked"] += 1
                self.logger.info("relinked image_id=%s path=%s", str(by_path["id"]), str(path))
                return

            try:
                sha = self._resolve_hash(path)
            except Exception as exc:
                self.stats["ignored_new_files"] += 1
                self.logger.warning("ignored_new_files path=%s reason=hash_error err=%s", str(path), str(exc))
                return

            by_hash = get_image_by_hash(conn, sha)
            if by_hash is None:
                self.stats["ignored_new_files"] += 1
                self.logger.info("ignored_new_files path=%s reason=unknown_hash", str(path))
                return

            inode, size, mtime = self._stat_or_zero(path)
            update_image_file_location(
                conn,
                image_id=str(by_hash["id"]),
                file_path=str(path),
                file_inode=inode,
                file_size=size,
                file_mtime=mtime,
            )
            self.stats["fallback_matched"] += 1
            self.logger.info(
                "fallback_matched image_id=%s old_path=%s new_path=%s",
                str(by_hash["id"]),
                str(by_hash["file_path"] or ""),
                str(path),
            )
        finally:
            conn.close()

    def _process_deleted_path(self, path: Path) -> None:
        # If a moved file already updated to another location, this path lookup will no-op.
        if path.exists():
            return
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        try:
            if mark_file_removed_by_path(conn, str(path)):
                self.stats["marked_stale"] += 1
                self.logger.info("marked_stale path=%s", str(path))
        finally:
            conn.close()

    def run_hourly_refresh(self) -> dict[str, int]:
        """
        Lightweight refresh only: reconcile missing indexed paths by hash
        and mark unresolved paths stale. No model inference.
        """
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        try:
            changed = check_stale_files(conn)
            watched_dirs = [Path(p) for p in list_enabled_watched_folders(conn)]
            watched_dirs = [p for p in watched_dirs if p.is_dir()]
            resolved = 0
            missing = 0
            unresolved = 0
            hash_cache: dict[str, str | None] = {}

            for entry in changed:
                if str(entry.get("reason", "")) != "missing":
                    continue
                missing += 1
                image_id = str(entry["image_id"])
                row = get_image_by_id(conn, image_id)
                if row is None:
                    continue
                old_path = str(row["file_path"] or "")
                target_hash = str(row["sha256_hash"] or "")
                if not target_hash:
                    mark_file_removed(conn, image_id)
                    unresolved += 1
                    continue

                if target_hash in hash_cache:
                    found = hash_cache[target_hash]
                else:
                    found = self._find_path_for_hash(
                        target_hash=target_hash,
                        old_name=Path(old_path).name,
                        watched_dirs=watched_dirs,
                    )
                    hash_cache[target_hash] = found

                if found:
                    inode, size, mtime = self._stat_or_zero(Path(found))
                    update_image_file_location(
                        conn,
                        image_id=image_id,
                        file_path=found,
                        file_inode=inode,
                        file_size=size,
                        file_mtime=mtime,
                    )
                    resolved += 1
                    self.stats["fallback_matched"] += 1
                    self.logger.info("fallback_matched image_id=%s old_path=%s new_path=%s", image_id, old_path, found)
                else:
                    mark_file_removed(conn, image_id)
                    unresolved += 1
                    self.stats["marked_stale"] += 1
                    self.logger.info("marked_stale path=%s", old_path)

            self.stats["hourly_refresh_runs"] += 1
            return {"missing_seen": missing, "resolved": resolved, "unresolved": unresolved}
        finally:
            conn.close()

    def _find_path_for_hash(self, *, target_hash: str, old_name: str, watched_dirs: list[Path]) -> str | None:
        for folder in watched_dirs:
            candidates: list[Path] = []
            if old_name:
                candidates.extend([p for p in folder.rglob(old_name) if p.is_file()])
            if not candidates:
                candidates.extend(
                    p
                    for p in folder.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                )

            for candidate in candidates:
                if not self._is_image_path(candidate):
                    continue
                try:
                    sha = self._resolve_hash(candidate)
                except Exception:
                    continue
                if sha == target_hash:
                    return str(candidate)
        return None

    def _load_enabled_watch_paths(self) -> set[str]:
        conn = connect_sqlite(self.cfg)
        ensure_schema(conn)
        try:
            paths = list_enabled_watched_folders(conn)
        finally:
            conn.close()
        out: set[str] = set()
        for p in paths:
            path = Path(p)
            if path.is_dir():
                out.add(str(path))
        return out

    def run(self, *, initial_refresh: bool = False) -> dict[str, Any]:
        """
        Run live watcher loop until interrupted (Ctrl+C) or stop() is called.
        """
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except Exception as exc:
            raise RuntimeError("watchdog is required. Install with: uv pip install watchdog") from exc

        class _Handler(FileSystemEventHandler):
            def __init__(self, owner: LivePathWatcher) -> None:
                self.owner = owner

            def on_created(self, event: Any) -> None:
                if getattr(event, "is_directory", False):
                    return
                self.owner.handle_created(str(event.src_path))

            def on_deleted(self, event: Any) -> None:
                if getattr(event, "is_directory", False):
                    return
                self.owner.handle_deleted(str(event.src_path))

            def on_moved(self, event: Any) -> None:
                if getattr(event, "is_directory", False):
                    return
                self.owner.handle_moved(str(event.src_path), str(event.dest_path))

        observer = Observer()
        handler = _Handler(self)
        scheduled: dict[str, Any] = {}
        desired_paths = self._load_enabled_watch_paths()
        for path in sorted(desired_paths):
            scheduled[path] = observer.schedule(handler, path, recursive=True)
            self.logger.info("watching path=%s", path)

        observer.start()
        try:
            if initial_refresh:
                self.run_hourly_refresh()
            next_refresh = time.time() + self.hourly_refresh_sec

            while not self.stop_event.is_set():
                self.process_pending()
                now = time.time()
                if now >= next_refresh:
                    # Refresh watch targets and run lightweight reconciliation.
                    desired = self._load_enabled_watch_paths()
                    for stale in sorted(set(scheduled.keys()) - desired):
                        observer.unschedule(scheduled.pop(stale))
                        self.logger.info("stopped_watching path=%s", stale)
                    for added in sorted(desired - set(scheduled.keys())):
                        scheduled[added] = observer.schedule(handler, added, recursive=True)
                        self.logger.info("watching path=%s", added)
                    self.run_hourly_refresh()
                    next_refresh = now + self.hourly_refresh_sec
                time.sleep(0.2)
        except KeyboardInterrupt:
            self.logger.info("watch-live interrupted by user")
        finally:
            observer.stop()
            observer.join(timeout=5.0)

        return {
            "status": "stopped",
            "watched_paths": sorted(scheduled.keys()),
            "stats": dict(self.stats),
        }


def watch_live(
    *,
    cfg: StackConfig | None = None,
    debounce_ms: int = 1200,
    move_grace_sec: float = 5.0,
    hourly_refresh_min: int = 60,
    initial_refresh: bool = False,
) -> dict[str, Any]:
    watcher = LivePathWatcher(
        cfg=cfg,
        debounce_ms=debounce_ms,
        move_grace_sec=move_grace_sec,
        hourly_refresh_min=hourly_refresh_min,
    )
    return watcher.run(initial_refresh=bool(initial_refresh))
