from __future__ import annotations

import json
import queue
import sys
import threading
import urllib.request
from dataclasses import dataclass
from typing import Any

from .utils import utc_now_iso


@dataclass(frozen=True)
class TelemetryOptions:
    command: str
    emit_to_stderr: bool = False
    webhook_url: str = ""
    webhook_timeout_sec: float = 2.0


class IngestTelemetry:
    """
    Best-effort telemetry sink for ingest/rescan flows.
    - stderr emission is line-delimited JSON
    - webhook emission is async and non-blocking for ingestion
    """

    def __init__(
        self,
        *,
        run_id: str,
        options: TelemetryOptions,
    ) -> None:
        self.run_id = run_id
        self.options = options
        self._webhook_queue: queue.Queue[dict[str, Any] | None] | None = None
        self._webhook_thread: threading.Thread | None = None

        url = self.options.webhook_url.strip()
        if url:
            self._webhook_queue = queue.Queue(maxsize=2048)
            self._webhook_thread = threading.Thread(target=self._webhook_worker, daemon=True)
            self._webhook_thread.start()

    def emit(self, event: str, **payload: Any) -> None:
        item = {
            "ts": utc_now_iso(),
            "run_id": self.run_id,
            "command": self.options.command,
            "event": str(event),
            **payload,
        }

        if self.options.emit_to_stderr:
            line = json.dumps(item, ensure_ascii=False)
            try:
                sys.stderr.write(f"[IngestEvent] {line}\n")
                sys.stderr.flush()
            except Exception:
                pass

        if self._webhook_queue is not None:
            try:
                self._webhook_queue.put_nowait(item)
            except queue.Full:
                # Drop instead of blocking ingest.
                pass

    def close(self) -> None:
        if self._webhook_queue is None:
            return
        try:
            self._webhook_queue.put_nowait(None)
        except queue.Full:
            pass
        thread = self._webhook_thread
        if thread is not None:
            thread.join(timeout=2.0)

    def _webhook_worker(self) -> None:
        q = self._webhook_queue
        if q is None:
            return
        url = self.options.webhook_url.strip()
        timeout = max(0.2, float(self.options.webhook_timeout_sec))

        while True:
            item = q.get()
            if item is None:
                return
            try:
                data = json.dumps(item, ensure_ascii=False).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout):
                    pass
            except Exception:
                # Best effort: never fail ingest due to webhook issues.
                continue
