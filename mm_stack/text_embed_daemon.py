from __future__ import annotations

import argparse
import json
import os
import socketserver
from pathlib import Path
from typing import Any

from .text_embedder import TextEmbedder


class _EmbedUnixServer(socketserver.UnixStreamServer):
    allow_reuse_address = True

    def __init__(self, socket_path: str, model_name: str):
        if os.path.exists(socket_path):
            try:
                os.remove(socket_path)
            except OSError:
                pass
        super().__init__(socket_path, _EmbedRequestHandler)
        self.socket_path = socket_path
        self.model_name = model_name
        self.embedder = TextEmbedder(model_name)
        self.embedder.load()

    def server_close(self) -> None:
        try:
            self.embedder.unload()
        except Exception:
            pass
        super().server_close()
        try:
            os.remove(self.socket_path)
        except OSError:
            pass


class _EmbedRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        line = self.rfile.readline()
        if not line:
            return

        payload: dict[str, Any] = {}
        try:
            payload = json.loads(line.decode("utf-8"))
        except Exception:
            self._write({"ok": False, "error": "invalid_json"})
            return

        text = str(payload.get("text", "")).strip()
        is_query = bool(payload.get("is_query", True))
        requested_model = str(payload.get("model_name", "")).strip()
        server_model = str(getattr(self.server, "model_name", "")).strip()
        if requested_model and server_model and requested_model != server_model:
            self._write({"ok": False, "error": "model_mismatch"})
            return
        if not text:
            self._write({"ok": False, "error": "empty_text"})
            return

        try:
            embedder = getattr(self.server, "embedder")
            vector = embedder.encode([text], is_query=is_query)[0]
            self._write({"ok": True, "vector": vector})
        except Exception as exc:
            self._write({"ok": False, "error": f"encode_failed:{exc}"})

    def _write(self, payload: dict[str, Any]) -> None:
        self.wfile.write((json.dumps(payload) + "\n").encode("utf-8"))
        self.wfile.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent text embedding daemon for Smart Stack")
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--pid-file", default="")
    args = parser.parse_args()

    Path(args.socket_path).parent.mkdir(parents=True, exist_ok=True)
    pid_file = args.pid_file.strip() or f"{args.socket_path}.pid"
    Path(pid_file).parent.mkdir(parents=True, exist_ok=True)

    if os.path.exists(pid_file):
        try:
            old_pid = int(Path(pid_file).read_text(encoding="utf-8").strip())
            os.kill(old_pid, 0)
            return
        except Exception:
            pass

    Path(pid_file).write_text(str(os.getpid()), encoding="utf-8")
    server = _EmbedUnixServer(args.socket_path, args.model_name)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        try:
            if Path(pid_file).exists():
                Path(pid_file).unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
